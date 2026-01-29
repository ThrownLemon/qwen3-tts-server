# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Pydantic schemas for OpenAI-compatible TTS API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class NormalizationOptions(BaseModel):
    """Options for the text normalization system."""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=True,
        description="Transforms units like 10KB to 10 kilobytes",
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes URLs so they can be properly pronounced",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronounced",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronounced",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replaces the remaining symbols after normalization with their words",
    )


class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint."""

    model: str = Field(
        default="qwen3-tts",
        description="The model to use for generation. Supported models: qwen3-tts, tts-1, tts-1-hd",
    )
    input: str = Field(
        ...,
        description="The text to generate audio for. Maximum length is 4096 characters.",
        max_length=4096,
    )
    voice: str = Field(
        default="Vivian",
        description="The voice to use for generation. Available voices depend on the loaded model.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, aac, flac, wav, pcm.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=False,
        description="If true, audio will be streamed as it's generated.",
    )
    language: Optional[str] = Field(
        default="Auto",
        description="Optional language code for TTS. If not provided, will auto-detect.",
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Optional instruction for voice style/emotion control.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default_factory=NormalizationOptions,
        description="Options for the text normalization system",
    )


class VoiceCloneRequest(BaseModel):
    """Request schema for voice cloning endpoint."""

    input: str = Field(
        ...,
        description="The text to generate audio for.",
        max_length=4096,
    )
    ref_audio: str = Field(
        ...,
        description="Reference audio as a base64-encoded string, URL, or local file path.",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of the reference audio. Improves cloning quality when provided.",
    )
    x_vector_only_mode: bool = Field(
        default=False,
        description="Use speaker embedding only (faster but lower quality). ref_text not needed in this mode.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default_factory=NormalizationOptions,
        description="Options for the text normalization system",
    )


class VoiceDesignRequest(BaseModel):
    """Request schema for voice design endpoint."""

    input: str = Field(
        ...,
        description="The text to generate audio for.",
        max_length=4096,
    )
    voice_description: str = Field(
        ...,
        description="Natural language description of the desired voice. E.g. 'Deep male voice with British accent, calm and authoritative'.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default_factory=NormalizationOptions,
        description="Options for the text normalization system",
    )


class ModelInfo(BaseModel):
    """Model information schema."""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = Field(..., description="Model owner/organization")


class VoiceInfo(BaseModel):
    """Voice information schema."""

    id: str = Field(..., description="Voice identifier")
    name: str = Field(..., description="Voice display name")
    language: Optional[str] = Field(None, description="Primary language of the voice")
    description: Optional[str] = Field(None, description="Voice description")
