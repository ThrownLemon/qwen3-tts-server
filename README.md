# qwen3-tts-server

Self-hosted Qwen3-TTS server with OpenAI-compatible API, voice cloning, and voice design.
Runs on an NVIDIA GPU and serves TTS over your local network.

Based on [groxaxo/Qwen3-TTS-Openai-Fastapi](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi) with added voice cloning, voice design, and sample preparation tools.

## Features

- **OpenAI-compatible** `/v1/audio/speech` endpoint (drop-in replacement)
- **Voice cloning** from 3-second audio samples (`/v1/audio/clone`)
- **Voice design** from natural language descriptions (`/v1/audio/design`)
- **Style/emotion control** via `instruct` parameter ("excited", "whisper", "angry", etc.)
- **Voice sample preparation** pipeline (Demucs + silence slicer + Whisper transcription)
- GPU-optimized with Flash Attention 2, torch.compile(), TF32, BF16
- MP3, WAV, FLAC, Opus, AAC, PCM output formats

## Requirements

- NVIDIA GPU (RTX 3060+ recommended, RTX 4090 ideal)
- Python 3.12
- CUDA toolkit
- ~7GB VRAM per model (up to 3 models: CustomVoice, Base, VoiceDesign)

## Setup

```bash
# 1. Create conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# 2. Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu131

# 3. Install the project
pip install -e ".[api]"

# 4. Install ffmpeg and sox (for audio format conversion)
conda install -y -c conda-forge ffmpeg sox

# 5. (Optional) Install Flash Attention 2 for faster inference
pip install flash-attn --no-build-isolation

# 6. Download models to local directory
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='models/CustomVoice')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='models/Base')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', local_dir='models/VoiceDesign')"

# 7. (Optional) Install voice sample prep tools
pip install demucs openai-whisper
```

### Starting the Server

```bash
./start.sh        # start server
./stop.sh         # stop server
```

Or manually:
```bash
conda activate qwen3-tts
python -m api.main
```

The server starts on `http://0.0.0.0:8880`.

### WSL2 Notes

If running in WSL2 and accessing from the Windows host, the server is available at `http://localhost:8880` from Windows (WSL2 auto-forwards ports). To access from other LAN machines, use the Windows host IP.

## Usage

### Text-to-Speech (OpenAI-compatible)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Ryan",
    input="Hello from Qwen3 TTS!",
    response_format="mp3",
)
response.stream_to_file("output.mp3")
```

### With emotion/style control

```python
import requests

resp = requests.post("http://localhost:8880/v1/audio/speech", json={
    "model": "qwen3-tts",
    "voice": "Ryan",
    "input": "I can't believe we won!",
    "instruct": "very excited and happy",
})
with open("excited.mp3", "wb") as f:
    f.write(resp.content)
```

### Voice Cloning

```python
import base64, requests

with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8880/v1/audio/clone", json={
    "input": "Hello in a cloned voice.",
    "ref_audio": ref_b64,
    "ref_text": "Transcript of the reference audio.",
})
with open("cloned.mp3", "wb") as f:
    f.write(resp.content)
```

### Voice Design

```python
resp = requests.post("http://localhost:8880/v1/audio/design", json={
    "input": "Welcome to the evening news.",
    "voice_description": "Deep male voice, British accent, calm and authoritative",
})
with open("designed.mp3", "wb") as f:
    f.write(resp.content)
```

## Voice Sample Preparation

Extract clean voice samples from any audio file:

```bash
python tools/prepare_samples.py recording.mp3 --output samples/speaker/
```

This runs:
1. **Demucs** — separates vocals from music/background
2. **Slicer** — splits into 5-15 second clips
3. **Whisper** — transcribes each clip

Output: numbered `.wav` files + `manifest.json` with transcripts.

Use `--skip-separation` if audio is already clean vocals.
Use `--skip-transcription` to skip Whisper.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | Generate speech (OpenAI-compatible) |
| POST | `/v1/audio/clone` | Clone voice from reference audio |
| POST | `/v1/audio/design` | Design voice from description |
| GET | `/v1/voices` | List available voices |
| GET | `/v1/models` | List available models |
| GET | `/health` | Server health check |
| GET | `/docs` | Swagger API documentation |

## Network Access

To access from another machine on your LAN:

1. Find your IP: `ip addr show` (or `hostname -I`)
2. Open firewall port if needed: `sudo ufw allow 8880/tcp`
3. Use `http://<YOUR_IP>:8880` from other devices

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8880` | Server port |
| `TTS_BACKEND` | `official` | Backend: `official` or `vllm_omni` |
| `TTS_WARMUP_ON_START` | `false` | Pre-warm models on startup |
| `CORS_ORIGINS` | `*` | CORS allowed origins |

## Documentation

- **[Agent API Guide](docs/AGENT_API_GUIDE.md)** — Comprehensive endpoint docs with examples for AI agents
- **Swagger UI** — Interactive docs at `http://localhost:8880/docs`
