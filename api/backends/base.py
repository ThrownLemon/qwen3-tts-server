# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Base class for TTS backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""
    
    def __init__(self):
        """Initialize the backend."""
        self.model = None
        self.device = None
        self.dtype = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend and load the model.
        
        This method should:
        - Load the model
        - Set up device and dtype
        - Perform any necessary warmup
        """
        pass
    
    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: The text to synthesize
            voice: Voice name/identifier to use
            language: Language code (e.g., "English", "Chinese", "Auto")
            instruct: Optional instruction for voice style/emotion
            speed: Speech speed multiplier (0.25 to 4.0)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass
    
    @abstractmethod
    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech by cloning a reference voice.

        Args:
            text: The text to synthesize
            ref_audio: Reference audio (base64 string, URL, or file path)
            ref_text: Transcript of the reference audio
            x_vector_only_mode: Use speaker embedding only (faster, lower quality)
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass

    @abstractmethod
    async def generate_voice_design(
        self,
        text: str,
        voice_description: str,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech with a voice designed from a natural language description.

        Args:
            text: The text to synthesize
            voice_description: Natural language description of the desired voice
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Return device information.
        
        Returns:
            Dict with keys: device, gpu_available, gpu_name, vram_total, vram_used
        """
        pass
