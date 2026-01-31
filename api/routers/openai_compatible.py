# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import base64
import logging
import os
import tempfile
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..structures.schemas import OpenAISpeechRequest, VoiceCloneRequest, VoiceDesignRequest, ModelInfo, VoiceInfo
from ..services.text_processing import normalize_text
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Serena",
    "nova": "Sohee",
    "onyx": "Eric",
    "shimmer": "Ono_anna",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """
    Extract language from model name if it has a language suffix.
    
    Args:
        model_name: Model name (e.g., "tts-1-es", "tts-1-hd-fr")
    
    Returns:
        Language name if suffix found, None otherwise
    """
    # Check if model ends with a language code
    # Only extract language if the model follows the expected pattern
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            # Verify it's a valid language-specific model variant
            # Should be either tts-1-{lang} or tts-1-hd-{lang}
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend
    
    backend = get_backend()
    
    if not backend.is_ready():
        await initialize_backend()
    
    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    # Check OpenAI voice mapping first
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    # Otherwise use the voice name directly
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text using the configured TTS backend.
    
    Args:
        text: The text to synthesize
        voice: Voice name to use
        language: Language code
        instruct: Optional instruction for voice style
        speed: Speech speed multiplier
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    backend = await get_tts_backend()
    
    # Map voice name
    voice_name = get_voice_name(voice)
    
    # Generate speech using the backend
    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )
        
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """
    OpenAI-compatible endpoint for text-to-speech.
    
    Generates audio from the input text using the specified voice and model.
    """
    # Validate model
    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )
    
    try:
        start_time = time.time()

        logger.debug(
            "[TTS Request] model=%s voice=%s format=%s speed=%.2f language=%s stream=%s instruct=%s input_length=%d",
            request.model, request.voice, request.response_format,
            request.speed, request.language, request.stream,
            request.instruct, len(request.input),
        )
        logger.debug("[TTS Request] input_text=%.200s", request.input)

        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        logger.debug("[TTS Normalize] normalized_length=%d text=%.200s", len(normalized_text), normalized_text)

        # Validate voice
        voice_name = get_voice_name(request.voice)
        backend = await get_tts_backend()
        supported_voices = backend.get_supported_voices()
        supported_lower = {v.lower(): v for v in supported_voices}
        if voice_name.lower() not in supported_lower:
            openai_aliases = list(VOICE_MAPPING.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_voice",
                    "message": f"Unsupported voice: '{request.voice}'. Supported voices: {sorted(supported_voices)} or OpenAI aliases: {openai_aliases}",
                    "type": "invalid_request_error",
                },
            )

        logger.debug("[TTS Voice] requested=%s mapped=%s", request.voice, voice_name)

        # Extract language from model name if present, otherwise use request language
        model_language = extract_language_from_model(request.model)
        language = model_language if model_language else (request.language or "Auto")

        # Generate speech
        gen_start = time.time()
        audio, sample_rate = await generate_speech(
            text=normalized_text,
            voice=request.voice,
            language=language,
            instruct=request.instruct,
            speed=request.speed,
        )
        gen_elapsed = time.time() - gen_start

        audio_duration = len(audio) / sample_rate
        logger.debug(
            "[TTS Generate] elapsed=%.3fs sample_rate=%d samples=%d audio_duration=%.2fs rtf=%.2f",
            gen_elapsed, sample_rate, len(audio), audio_duration,
            gen_elapsed / audio_duration if audio_duration > 0 else 0,
        )

        # Encode audio to requested format
        encode_start = time.time()
        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        encode_elapsed = time.time() - encode_start

        logger.debug(
            "[TTS Encode] format=%s elapsed=%.3fs output_size=%d bytes",
            request.response_format, encode_elapsed, len(audio_bytes),
        )

        # Get content type
        content_type = get_content_type(request.response_format)

        total_elapsed = time.time() - start_time
        logger.debug(
            "[TTS Done] total=%.3fs (generate=%.3fs encode=%.3fs) audio=%.2fs size=%d bytes",
            total_elapsed, gen_elapsed, encode_elapsed, audio_duration, len(audio_bytes),
        )

        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    # Default voices (always available)
    default_voices = [
        VoiceInfo(id="Vivian", name="Vivian", language="English", description="Female voice"),
        VoiceInfo(id="Ryan", name="Ryan", language="English", description="Male voice"),
        VoiceInfo(id="Sophia", name="Sophia", language="English", description="Female voice"),
        VoiceInfo(id="Isabella", name="Isabella", language="English", description="Female voice"),
        VoiceInfo(id="Evan", name="Evan", language="English", description="Male voice"),
        VoiceInfo(id="Lily", name="Lily", language="English", description="Female voice"),
    ]
    
    # OpenAI-compatible voice aliases
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Serena)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Sohee)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Eric)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Ono_anna)"),
    ]
    
    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]
    
    try:
        backend = await get_tts_backend()
        
        # Get supported speakers from the backend
        speakers = backend.get_supported_voices()
        
        # Get supported languages
        languages = backend.get_supported_languages()
        
        # Build voice list from backend
        if speakers:
            voices = []
            for speaker in speakers:
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=f"Qwen3-TTS voice: {speaker}",
                )
                voices.append(voice_info.model_dump())
        else:
            voices = [v.model_dump() for v in default_voices]
        
        return {
            "voices": voices + [v.model_dump() for v in openai_voices],
            "languages": languages if languages else default_languages,
        }
        
    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        # Return default voices if backend is not loaded
        return {
            "voices": [v.model_dump() for v in default_voices] + [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }


@router.post("/audio/clone")
async def create_voice_clone(request: VoiceCloneRequest):
    """
    Clone a voice from reference audio and generate speech.

    Provide a base64-encoded audio sample (or URL/file path) and the text to speak.
    Optionally include the transcript of the reference audio for better quality.
    """
    try:
        start_time = time.time()
        logger.debug(
            "[Clone Request] format=%s speed=%.2f x_vector_only=%s ref_text=%s input_length=%d",
            request.response_format, request.speed, request.x_vector_only_mode,
            request.ref_text is not None, len(request.input),
        )
        backend = await get_tts_backend()

        normalized_text = normalize_text(request.input, request.normalization_options)
        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_input", "message": "Input text is empty after normalization", "type": "invalid_request_error"},
            )

        # Decode base64 ref_audio to a temp file if it's not already a file path
        ref_audio_path = request.ref_audio
        temp_file = None
        if not os.path.exists(ref_audio_path):
            try:
                audio_data = base64.b64decode(ref_audio_path)
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(audio_data)
                temp_file.close()
                ref_audio_path = temp_file.name
            except Exception:
                pass  # Not base64, let the backend handle it

        try:
            audio, sample_rate = await backend.generate_voice_clone(
                text=normalized_text,
                ref_audio=ref_audio_path,
                ref_text=request.ref_text,
                x_vector_only_mode=request.x_vector_only_mode,
                speed=request.speed,
            )
        finally:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        content_type = get_content_type(request.response_format)

        logger.debug("[Clone Done] total=%.3fs size=%d bytes", time.time() - start_time, len(audio_bytes))

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=clone.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail={"error": "not_implemented", "message": str(e), "type": "server_error"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "processing_error", "message": str(e), "type": "server_error"})


@router.post("/audio/design")
async def create_voice_design(request: VoiceDesignRequest):
    """
    Design a custom voice from a natural language description and generate speech.

    Describe the voice you want (e.g. "Deep male voice with British accent, calm and authoritative")
    and provide the text to speak.
    """
    try:
        start_time = time.time()
        logger.debug(
            "[Design Request] format=%s speed=%.2f description=%.100s input_length=%d",
            request.response_format, request.speed, request.voice_description, len(request.input),
        )
        backend = await get_tts_backend()

        normalized_text = normalize_text(request.input, request.normalization_options)
        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_input", "message": "Input text is empty after normalization", "type": "invalid_request_error"},
            )

        audio, sample_rate = await backend.generate_voice_design(
            text=normalized_text,
            voice_description=request.voice_description,
            speed=request.speed,
        )

        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        content_type = get_content_type(request.response_format)

        logger.debug("[Design Done] total=%.3fs size=%d bytes", time.time() - start_time, len(audio_bytes))

        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=design.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail={"error": "not_implemented", "message": str(e), "type": "server_error"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "processing_error", "message": str(e), "type": "server_error"})
