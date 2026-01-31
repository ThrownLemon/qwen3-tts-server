# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-hosted TTS server providing an OpenAI-compatible API (`/v1/audio/speech`), voice cloning (`/v1/audio/clone`), and voice design (`/v1/audio/design`) built on Alibaba's Qwen3-TTS models. Supports multiple audio formats (MP3, WAV, FLAC, Opus, AAC, PCM).

## Commands

```bash
# Setup
conda create -n qwen3-tts python=3.12 -y && conda activate qwen3-tts
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu131
pip install -e ".[api]"
conda install -y -c conda-forge ffmpeg sox

# Run server
python -m api.main                    # starts on http://0.0.0.0:8880
./start.sh                            # Linux/WSL2 shortcut
docker-compose up qwen3-tts-gpu       # Docker (GPU)

# Tests
pytest tests/                         # all tests
pytest tests/test_api.py              # API tests only
pytest tests/test_backends.py         # backend tests only
pytest tests/test_api.py::test_health # single test

# Benchmarks
python bench_tts.py
python benchmark_official.py
```

## Architecture

**Two-backend system** selected via `TTS_BACKEND` env var (`official` or `vllm_omni`):
- `api/backends/base.py` — abstract `TTSBackend` base class defining the interface
- `api/backends/official_qwen3_tts.py` — default backend using Qwen3-TTS directly
- `api/backends/vllm_omni_qwen3_tts.py` — alternative backend using vLLM for inference
- `api/backends/factory.py` — instantiates the correct backend

**Request flow:** FastAPI router (`api/routers/openai_compatible.py`) → backend → audio encoding (`api/services/audio_encoding.py`). Text is normalized before synthesis via `api/services/text_processing.py`.

**Core model code** lives in `qwen_tts/core/models/` (model definition + processor) with a high-level wrapper at `qwen_tts/inference/qwen3_tts_model.py`. The 12Hz speech tokenizer is at `qwen_tts/core/tokenizer_12hz/`.

**Models** are stored locally in `models/` (CustomVoice, Base, VoiceDesign, Tokenizer-12Hz) for faster loading and offline use.

**Voice mapping:** alloy→Vivian, echo→Ryan, fable→Sophia, nova→Isabella, onyx→Evan, shimmer→Lily.

## Key Environment Variables

- `HOST` / `PORT` (default: 0.0.0.0:8880)
- `TTS_BACKEND` — `official` (default) or `vllm_omni`
- `TTS_WARMUP_ON_START` — pre-warm models on startup
- `TTS_MODEL_NAME` — override default model path

## Request/Response Schemas

Defined in `api/structures/schemas.py` as Pydantic models: `OpenAISpeechRequest`, `VoiceCloneRequest`, `VoiceDesignRequest`.
