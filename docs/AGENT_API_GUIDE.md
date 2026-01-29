# Qwen3-TTS Server - Agent API Guide

This document is a reference for AI agents (Claude, etc.) to use the Qwen3-TTS API server.

**Base URL:** `http://<SERVER_IP>:8880`

---

## Endpoints

### 1. `POST /v1/audio/speech` — Text-to-Speech (OpenAI-compatible)

Generate speech from text using a preset voice with optional style/emotion control.

**Request:**
```json
{
  "model": "qwen3-tts",
  "voice": "Ryan",
  "input": "Hello, this is a test of the text to speech system.",
  "response_format": "mp3",
  "speed": 1.0,
  "instruct": "calm and professional"
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | no | `"qwen3-tts"` | Model name. Also accepts `tts-1`, `tts-1-hd`. |
| `input` | string | **yes** | — | Text to speak (max 4096 chars). |
| `voice` | string | no | `"Vivian"` | Voice name (see table below). |
| `response_format` | string | no | `"mp3"` | One of: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`. |
| `speed` | float | no | `1.0` | Speed multiplier: `0.25` to `4.0`. |
| `instruct` | string | no | `null` | Natural language style/emotion instruction. |

**Response:** Binary audio file with appropriate Content-Type header.

**curl example:**
```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","voice":"Ryan","input":"Hello world!","instruct":"excited"}' \
  --output speech.mp3
```

**Python (OpenAI client):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Ryan",
    input="Hello world!",
    response_format="mp3",
    speed=1.0,
)
response.stream_to_file("output.mp3")
```

**Python (requests, with instruct):**
```python
import requests

resp = requests.post("http://localhost:8880/v1/audio/speech", json={
    "model": "qwen3-tts",
    "voice": "Ryan",
    "input": "I'm so excited to tell you about this!",
    "instruct": "very excited and enthusiastic",
    "response_format": "wav",
})
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

---

### 2. `POST /v1/audio/clone` — Voice Cloning

Clone a voice from a reference audio sample and generate speech in that voice.

**Request (JSON):**
```json
{
  "input": "Text to speak in the cloned voice.",
  "ref_audio": "<base64-encoded audio OR URL OR file path>",
  "ref_text": "Transcript of the reference audio.",
  "x_vector_only_mode": false,
  "response_format": "mp3",
  "speed": 1.0
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | **yes** | — | Text to speak. |
| `ref_audio` | string | **yes** | — | Reference audio: base64 string, URL, or local file path. |
| `ref_text` | string | no | `null` | Transcript of ref audio. Improves quality significantly. |
| `x_vector_only_mode` | bool | no | `false` | Use speaker embedding only (faster, lower quality). |
| `response_format` | string | no | `"mp3"` | Audio format. |
| `speed` | float | no | `1.0` | Speed multiplier. |

**Tips for best results:**
- Use 5-15 seconds of clean reference audio
- Provide `ref_text` (the transcript) for best quality
- Set `x_vector_only_mode: true` for faster but lower-quality cloning

**curl example:**
```bash
# Using base64
REF_B64=$(base64 -i reference.wav)
curl -X POST http://localhost:8880/v1/audio/clone \
  -H "Content-Type: application/json" \
  -d "{\"input\":\"Hello in a cloned voice.\",\"ref_audio\":\"$REF_B64\",\"ref_text\":\"This is the reference text.\"}" \
  --output cloned.mp3
```

**Python example:**
```python
import base64, requests

with open("reference.wav", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8880/v1/audio/clone", json={
    "input": "Hello, this is my cloned voice speaking.",
    "ref_audio": ref_b64,
    "ref_text": "This is what I said in the reference audio.",
    "response_format": "wav",
})
with open("cloned.wav", "wb") as f:
    f.write(resp.content)
```

---

### 3. `POST /v1/audio/design` — Voice Design

Create a custom voice from a natural language description and generate speech.

**Request:**
```json
{
  "input": "Text to speak with the designed voice.",
  "voice_description": "Deep male voice with a British accent, calm and authoritative, slight warmth",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Fields:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input` | string | **yes** | — | Text to speak. |
| `voice_description` | string | **yes** | — | Natural language description of the desired voice. |
| `response_format` | string | no | `"mp3"` | Audio format. |
| `speed` | float | no | `1.0` | Speed multiplier. |

**Voice description examples:**
- `"Deep male voice with British accent, calm and authoritative"`
- `"Young female voice, bright and energetic, American accent"`
- `"Elderly male, warm and gentle, slow speaking pace"`
- `"Professional female news anchor voice, clear and crisp"`
- `"Raspy male voice, slight southern drawl, relaxed tone"`

**curl example:**
```bash
curl -X POST http://localhost:8880/v1/audio/design \
  -H "Content-Type: application/json" \
  -d '{"input":"Welcome to the show.","voice_description":"Deep male radio announcer voice"}' \
  --output designed.mp3
```

---

### 4. `GET /v1/voices` — List Available Voices

Returns the list of preset voices and supported languages.

```bash
curl http://localhost:8880/v1/voices
```

### 5. `GET /v1/models` — List Models

```bash
curl http://localhost:8880/v1/models
```

### 6. `GET /health` — Health Check

```bash
curl http://localhost:8880/health
```

Returns backend status, GPU info, and model readiness.

---

## Available Preset Voices

| Voice | Gender | Style | Best For |
|-------|--------|-------|----------|
| **Ryan** | Male | Strong rhythm, dynamic | General English narration |
| **Aiden** | Male | Sunny, clear, American | Casual, friendly content |
| **Vivian** | Female | Bright, slightly edgy | Energetic presentations |
| **Sophia** | Female | — | General female voice |
| **Isabella** | Female | — | General female voice |
| **Evan** | Male | — | General male voice |
| **Lily** | Female | — | General female voice |

**OpenAI alias mapping** (for drop-in compatibility):
- `alloy` → Vivian
- `echo` → Ryan
- `fable` → Sophia
- `nova` → Isabella
- `onyx` → Evan
- `shimmer` → Lily

---

## The `instruct` Parameter

The `instruct` field on `/v1/audio/speech` controls emotion, style, and delivery using natural language. Examples:

**Emotions:**
- `"very happy and excited"`
- `"sad and melancholic"`
- `"angry and frustrated"`
- `"calm and relaxed"`
- `"fearful and anxious"`

**Delivery style:**
- `"whisper softly"`
- `"shout loudly"`
- `"speak formally"`
- `"speak casually"`
- `"slow, deliberate pace with dramatic pauses"`
- `"fast-paced and energetic"`

**Character/persona:**
- `"professional news anchor"`
- `"friendly customer service agent"`
- `"storyteller reading to children"`
- `"stern military commander"`

You can combine multiple instructions: `"excited and happy, speaking quickly"`

---

## Voice Cloning Workflow

1. **Prepare samples** using the included tool:
   ```bash
   python tools/prepare_samples.py recording.mp3 --output samples/my_voice/
   ```
   This produces numbered `.wav` clips and a `manifest.json` with transcripts.

2. **Pick the best clip** — choose a clean 5-15 second sample with clear speech.

3. **Clone the voice:**
   ```python
   import base64, requests, json

   # Load manifest to get transcript
   with open("samples/my_voice/manifest.json") as f:
       manifest = json.load(f)

   best_sample = manifest["samples"][0]  # pick the best one

   with open(best_sample["path"], "rb") as f:
       ref_b64 = base64.b64encode(f.read()).decode()

   resp = requests.post("http://SERVER:8880/v1/audio/clone", json={
       "input": "This is my cloned voice speaking new text.",
       "ref_audio": ref_b64,
       "ref_text": best_sample["transcript"],
   })
   with open("result.mp3", "wb") as f:
       f.write(resp.content)
   ```

---

## Error Handling

All endpoints return JSON errors:
```json
{
  "detail": {
    "error": "processing_error",
    "message": "Description of what went wrong",
    "type": "server_error"
  }
}
```

Common HTTP status codes:
- `400` — Invalid input (empty text, bad model name, etc.)
- `500` — Server processing error
- `501` — Feature not available with current backend (e.g. clone/design with vLLM backend)

---

## Audio Formats

| Format | MIME Type | Notes |
|--------|-----------|-------|
| `mp3` | `audio/mpeg` | Default. Good compression, universal support. |
| `wav` | `audio/wav` | Uncompressed. Largest file size. |
| `flac` | `audio/flac` | Lossless compression. |
| `opus` | `audio/opus` | Efficient compression. |
| `aac` | `audio/aac` | Good compression. |
| `pcm` | `audio/pcm` | Raw 16-bit PCM. No header. |

---

## Server Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8880` | Server port |
| `TTS_BACKEND` | `official` | Backend engine (`official` or `vllm_omni`) |
| `TTS_WARMUP_ON_START` | `false` | Pre-warm models on startup |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
