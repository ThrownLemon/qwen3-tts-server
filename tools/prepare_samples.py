#!/usr/bin/env python3
"""
Voice Sample Preparation Pipeline

Extracts clean voice samples from raw audio files for use with Qwen3-TTS voice cloning.

Pipeline:
  1. Vocal separation (Demucs) - removes music/background noise
  2. Segmentation (silence-based) - splits into 5-15 second clips
  3. Transcription (Whisper) - generates text transcripts for each clip

Output:
  <output_dir>/
    clip_000.wav
    clip_001.wav
    ...
    manifest.json   # maps each clip to its transcript

Usage:
  python tools/prepare_samples.py input_audio.mp3 --output samples/speaker_name/
  python tools/prepare_samples.py input_audio.mp3 --skip-separation  # if audio is already clean vocals
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def separate_vocals(input_path: str, output_dir: str) -> str:
    """Use Demucs to separate vocals from background audio."""
    print(f"[1/3] Separating vocals with Demucs...")
    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-o", output_dir,
            input_path,
        ],
        check=True,
    )
    # Demucs outputs to <output_dir>/htdemucs/<stem_name>/vocals.wav
    stem = Path(input_path).stem
    vocals_path = os.path.join(output_dir, "htdemucs", stem, "vocals.wav")
    if not os.path.exists(vocals_path):
        # Try alternate structure
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f == "vocals.wav":
                    vocals_path = os.path.join(root, f)
                    break
    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Demucs did not produce vocals.wav. Check {output_dir}")
    print(f"  Vocals extracted: {vocals_path}")
    return vocals_path


def slice_audio(
    audio_path: str,
    output_dir: str,
    min_length_ms: int = 5000,
    max_length_ms: int = 15000,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 300,
) -> list[str]:
    """Slice audio into clips based on silence detection."""
    print(f"[2/3] Slicing audio into {min_length_ms//1000}-{max_length_ms//1000}s clips...")

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono

    # Simple RMS-based silence detection
    hop_size = int(sr * 0.01)  # 10ms hops
    frame_length = int(sr * 0.025)  # 25ms frames
    min_silence_frames = int(min_silence_ms / 10)
    min_length_samples = int(min_length_ms / 1000 * sr)
    max_length_samples = int(max_length_ms / 1000 * sr)

    # Compute RMS energy per frame
    n_frames = (len(audio) - frame_length) // hop_size + 1
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start : start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    # Convert threshold from dB to linear
    silence_thresh = 10 ** (silence_thresh_db / 20)

    # Find silent regions
    is_silent = rms < silence_thresh

    # Find split points (centers of silence regions longer than min_silence_frames)
    split_points = [0]
    silence_start = None
    for i, silent in enumerate(is_silent):
        if silent and silence_start is None:
            silence_start = i
        elif not silent and silence_start is not None:
            duration = i - silence_start
            if duration >= min_silence_frames:
                center = (silence_start + i) // 2
                split_sample = center * hop_size
                split_points.append(split_sample)
            silence_start = None
    split_points.append(len(audio))

    # Merge short segments, split long ones
    clips = []
    current_start = split_points[0]
    for sp in split_points[1:]:
        segment_len = sp - current_start
        if segment_len < min_length_samples:
            continue  # merge with next
        if segment_len > max_length_samples:
            # Force-split at max_length intervals
            pos = current_start
            while pos + min_length_samples <= sp:
                end = min(pos + max_length_samples, sp)
                clips.append((pos, end))
                pos = end
            current_start = sp
        else:
            clips.append((current_start, sp))
            current_start = sp

    # Handle remaining audio
    if current_start < len(audio):
        remaining = len(audio) - current_start
        if remaining >= min_length_samples:
            clips.append((current_start, len(audio)))
        elif clips:
            # Extend last clip
            start, _ = clips[-1]
            clips[-1] = (start, len(audio))

    # Write clips
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for i, (start, end) in enumerate(clips):
        clip = audio[start:end]
        clip_path = os.path.join(output_dir, f"clip_{i:03d}.wav")
        sf.write(clip_path, clip, sr)
        clip_paths.append(clip_path)
        duration = (end - start) / sr
        print(f"  clip_{i:03d}.wav  ({duration:.1f}s)")

    print(f"  {len(clip_paths)} clips created")
    return clip_paths


def transcribe_clips(clip_paths: list[str], model_size: str = "base") -> dict[str, str]:
    """Transcribe clips using Whisper."""
    print(f"[3/3] Transcribing {len(clip_paths)} clips with Whisper ({model_size})...")

    try:
        import whisper
    except ImportError:
        print("  whisper not installed. Install with: pip install openai-whisper")
        print("  Skipping transcription. You can add transcripts manually to manifest.json")
        return {p: "" for p in clip_paths}

    model = whisper.load_model(model_size)
    transcripts = {}
    for i, path in enumerate(clip_paths):
        result = model.transcribe(path, language="en")
        text = result["text"].strip()
        transcripts[path] = text
        print(f"  clip_{i:03d}: {text[:80]}{'...' if len(text) > 80 else ''}")

    return transcripts


def write_manifest(output_dir: str, clip_paths: list[str], transcripts: dict[str, str]):
    """Write manifest.json mapping clips to transcripts."""
    entries = []
    for path in clip_paths:
        filename = os.path.basename(path)
        audio, sr = sf.read(path)
        duration = len(audio) / sr
        entries.append({
            "filename": filename,
            "path": os.path.abspath(path),
            "duration_seconds": round(duration, 2),
            "transcript": transcripts.get(path, ""),
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"samples": entries, "count": len(entries)}, f, indent=2, ensure_ascii=False)

    print(f"\nManifest written to {manifest_path}")
    print(f"Total: {len(entries)} clips")


def main():
    parser = argparse.ArgumentParser(description="Prepare voice samples for Qwen3-TTS voice cloning")
    parser.add_argument("input", help="Input audio file (mp3, wav, flac, etc.)")
    parser.add_argument("--output", "-o", default="samples/output", help="Output directory (default: samples/output)")
    parser.add_argument("--skip-separation", action="store_true", help="Skip vocal separation (use if audio is already clean vocals)")
    parser.add_argument("--min-length", type=int, default=5000, help="Minimum clip length in ms (default: 5000)")
    parser.add_argument("--max-length", type=int, default=15000, help="Maximum clip length in ms (default: 15000)")
    parser.add_argument("--silence-thresh", type=float, default=-40.0, help="Silence threshold in dB (default: -40)")
    parser.add_argument("--min-silence", type=int, default=300, help="Minimum silence duration in ms for splitting (default: 300)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size: tiny, base, small, medium, large (default: base)")
    parser.add_argument("--skip-transcription", action="store_true", help="Skip Whisper transcription")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Vocal separation
    if args.skip_separation:
        print("[1/3] Skipping vocal separation")
        audio_path = input_path
    else:
        tmp_dir = tempfile.mkdtemp(prefix="tts_prep_")
        audio_path = separate_vocals(input_path, tmp_dir)

    # Step 2: Slice into clips
    clip_paths = slice_audio(
        audio_path,
        output_dir,
        min_length_ms=args.min_length,
        max_length_ms=args.max_length,
        silence_thresh_db=args.silence_thresh,
        min_silence_ms=args.min_silence,
    )

    if not clip_paths:
        print("No clips generated. Try adjusting --silence-thresh or --min-length.")
        sys.exit(1)

    # Step 3: Transcribe
    if args.skip_transcription:
        print("[3/3] Skipping transcription")
        transcripts = {p: "" for p in clip_paths}
    else:
        transcripts = transcribe_clips(clip_paths, model_size=args.whisper_model)

    # Write manifest
    write_manifest(output_dir, clip_paths, transcripts)
    print("\nDone! Use these samples with the /v1/audio/clone endpoint.")


if __name__ == "__main__":
    main()
