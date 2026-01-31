# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Official Qwen3-TTS backend implementation.

This backend uses the official Qwen3-TTS Python implementation
from the qwen_tts package.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_MODELS_DIR = _PROJECT_ROOT / "models"

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def _load_cloned_voices(samples_dir: Path) -> Dict[str, Dict[str, str]]:
    """Scan samples/ for cloned voice directories containing manifest.json.

    Returns a dict mapping lowercase voice name to {"ref_audio": path, "ref_text": transcript}.
    Picks the clip with the longest duration that has a non-empty transcript.
    """
    voices: Dict[str, Dict[str, str]] = {}
    if not samples_dir.is_dir():
        return voices

    for voice_dir in samples_dir.iterdir():
        manifest_path = voice_dir / "clips" / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            # Pick best clip: longest duration with a transcript
            best = None
            for sample in manifest.get("samples", []):
                if not sample.get("transcript", "").strip():
                    continue
                if best is None or sample["duration_seconds"] > best["duration_seconds"]:
                    best = sample
            if best is None:
                continue
            clip_path = str(voice_dir / "clips" / best["filename"])
            voices[voice_dir.name.lower()] = {
                "ref_audio": clip_path,
                "ref_text": best["transcript"],
                "display_name": voice_dir.name.capitalize(),
            }
            logger.info(f"Loaded cloned voice '{voice_dir.name}' from {clip_path}")
        except Exception as e:
            logger.warning(f"Failed to load cloned voice from {voice_dir}: {e}")

    return voices


class OfficialQwen3TTSBackend(TTSBackend):
    """Official Qwen3-TTS backend using the qwen_tts package."""

    # Default to local model path if it exists, otherwise use HuggingFace ID
    _DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        """
        Initialize the official backend.

        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__()
        self.model_name = model_name
        self._ready = False
        self._clone_model = None
        self._design_model = None
        self._clone_ready = False
        self._design_ready = False
        # Discover cloned voices from samples/ directory
        self._cloned_voices = _load_cloned_voices(_PROJECT_ROOT / "samples")
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("Official backend already initialized")
            return
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.dtype = torch.bfloat16
            else:
                self.device = "cpu"
                self.dtype = torch.float32
            
            logger.info(f"Loading Qwen3-TTS model '{self.model_name}' on {self.device}...")

            # Try loading with Flash Attention 2, fallback to SDPA or eager if not supported
            # (e.g., RTX 5090/Blackwell GPUs don't have pre-built flash-attn wheels yet)
            attn_implementations = ["flash_attention_2", "sdpa", "eager"]
            model_loaded = False

            last_error = None
            for attn_impl in attn_implementations:
                try:
                    logger.info(f"Attempting to load model with attention: {attn_impl}")
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.model_name,
                        device_map=self.device,
                        dtype=self.dtype,
                        attn_implementation=attn_impl,
                    )
                    logger.info(f"Successfully loaded model with {attn_impl} attention")
                    model_loaded = True
                    break
                except Exception as attn_error:
                    last_error = attn_error
                    logger.warning(f"Could not load with {attn_impl}: {attn_error}")
                    if attn_impl != attn_implementations[-1]:
                        logger.info(f"Falling back to next attention implementation...")

            if not model_loaded:
                # If GPU loading failed completely, try CPU as last resort
                if self.device != "cpu":
                    logger.warning("All GPU attention implementations failed. Falling back to CPU...")
                    self.device = "cpu"
                    self.dtype = torch.float32
                    try:
                        self.model = Qwen3TTSModel.from_pretrained(
                            self.model_name,
                            device_map=self.device,
                            dtype=self.dtype,
                            attn_implementation="eager",
                        )
                        logger.info("Successfully loaded model on CPU (GPU not compatible)")
                        model_loaded = True
                    except Exception as cpu_error:
                        raise RuntimeError(f"Failed to load model on CPU: {cpu_error}")
                else:
                    raise RuntimeError(f"Failed to load model with any attention implementation. Last error: {last_error}")

            # Apply torch.compile() optimization for faster inference
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                logger.info("Applying torch.compile() optimization...")
                try:
                    # Compile the model with reduce-overhead mode for faster inference
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",  # Optimize for inference speed
                        fullgraph=False,  # Allow graph breaks for compatibility
                    )
                    logger.info("torch.compile() optimization applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply torch.compile(): {e}")
            
            # Enable cuDNN benchmarking for optimal convolution algorithms
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
            
            # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for faster matmul")
            
            self._ready = True
            logger.info(f"Official Qwen3-TTS backend loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load official TTS backend: {e}")
            raise RuntimeError(f"Failed to initialize official TTS backend: {e}")
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using the official Qwen3-TTS model.
        
        Args:
            text: The text to synthesize
            voice: Voice name to use
            language: Language code
            instruct: Optional instruction for voice style
            speed: Speech speed multiplier
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()

        # Check if this is a cloned voice (only if the model doesn't have it as a built-in speaker)
        voice_key = voice.lower()
        builtin_speakers = set()
        try:
            if self.model and hasattr(self.model.model, 'get_supported_speakers'):
                speakers = self.model.model.get_supported_speakers()
                if speakers:
                    builtin_speakers = {s.lower() for s in speakers}
        except Exception:
            pass

        if voice_key in self._cloned_voices and voice_key not in builtin_speakers:
            clone_info = self._cloned_voices[voice_key]
            logger.debug(
                "[Backend] generate_speech routed to CLONE: voice=%s ref_audio=%s speed=%.2f text_length=%d",
                voice, clone_info["ref_audio"], speed, len(text),
            )
            return await self.generate_voice_clone(
                text=text,
                ref_audio=clone_info["ref_audio"],
                ref_text=clone_info["ref_text"],
                speed=speed,
            )

        try:
            logger.debug(
                "[Backend] generate_speech called: voice=%s language=%s speed=%.2f instruct=%s text_length=%d",
                voice, language, speed, instruct, len(text),
            )

            # Generate speech
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=instruct,
            )

            audio = wavs[0]
            logger.debug("[Backend] raw audio: samples=%d sample_rate=%d duration=%.2fs", len(audio), sr, len(audio) / sr)

            # Log VRAM after generation
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.debug("[Backend] VRAM: allocated=%.2f GB reserved=%.2f GB", allocated, reserved)
            except Exception:
                pass

            # Apply speed adjustment if needed
            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
                logger.debug("[Backend] speed adjusted to %.2fx, new samples=%d", speed, len(audio))
            elif speed != 1.0:
                logger.warning("Speed adjustment requested but librosa not available")

            return audio, sr
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")
    
    async def _ensure_clone_model(self):
        """Lazy-load the Base model for voice cloning."""
        if self._clone_ready:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            logger.info("Loading Qwen3-TTS Base model for voice cloning...")
            device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = self.dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

            # Prefer local model if available
            local_base = _LOCAL_MODELS_DIR / "Base"
            model_id = str(local_base) if (local_base / "config.json").exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

            for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
                try:
                    self._clone_model = Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=device,
                        dtype=dtype,
                        attn_implementation=attn_impl,
                    )
                    logger.info(f"Clone model loaded with {attn_impl}")
                    self._clone_ready = True
                    return
                except Exception as e:
                    logger.warning(f"Clone model failed with {attn_impl}: {e}")
            raise RuntimeError("Failed to load clone model with any attention implementation")
        except Exception as e:
            logger.error(f"Failed to load clone model: {e}")
            raise

    async def _ensure_design_model(self):
        """Lazy-load the VoiceDesign model."""
        if self._design_ready:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            logger.info("Loading Qwen3-TTS VoiceDesign model...")
            device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = self.dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

            # Prefer local model if available
            local_vd = _LOCAL_MODELS_DIR / "VoiceDesign"
            model_id = str(local_vd) if (local_vd / "config.json").exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

            for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
                try:
                    self._design_model = Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=device,
                        dtype=dtype,
                        attn_implementation=attn_impl,
                    )
                    logger.info(f"Design model loaded with {attn_impl}")
                    self._design_ready = True
                    return
                except Exception as e:
                    logger.warning(f"Design model failed with {attn_impl}: {e}")
            raise RuntimeError("Failed to load design model with any attention implementation")
        except Exception as e:
            logger.error(f"Failed to load design model: {e}")
            raise

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech by cloning a reference voice."""
        await self._ensure_clone_model()

        try:
            wavs, sr = self._clone_model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            audio = wavs[0]

            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)

            return audio, sr
        except Exception as e:
            logger.error(f"Voice clone generation failed: {e}")
            raise RuntimeError(f"Voice clone generation failed: {e}")

    async def generate_voice_design(
        self,
        text: str,
        voice_description: str,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech with a designed voice from a natural language description."""
        await self._ensure_design_model()

        try:
            wavs, sr = self._design_model.generate_voice_design(
                text=text,
                language="English",
                instruct=voice_description,
            )
            audio = wavs[0]

            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)

            return audio, sr
        except Exception as e:
            logger.error(f"Voice design generation failed: {e}")
            raise RuntimeError(f"Voice design generation failed: {e}")

    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "official"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names, including cloned voices."""
        if not self._ready or not self.model:
            voices = ["Vivian", "Ryan", "Serena", "Sohee", "Eric", "Ono_anna", "Aiden", "Dylan", "Uncle_fu"]
        else:
            try:
                if hasattr(self.model.model, 'get_supported_speakers'):
                    speakers = self.model.model.get_supported_speakers()
                    if speakers:
                        voices = list(speakers)
                    else:
                        voices = ["Vivian", "Ryan", "Serena", "Sohee", "Eric", "Ono_anna", "Aiden", "Dylan", "Uncle_fu"]
                else:
                    voices = ["Vivian", "Ryan", "Serena", "Sohee", "Eric", "Ono_anna", "Aiden", "Dylan", "Uncle_fu"]
            except Exception as e:
                logger.warning(f"Could not get speakers from model: {e}")
                voices = ["Vivian", "Ryan", "Serena", "Sohee", "Eric", "Ono_anna", "Aiden", "Dylan", "Uncle_fu"]

        # Add cloned voices
        for info in self._cloned_voices.values():
            display = info["display_name"]
            if display not in voices:
                voices.append(display)

        return voices
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        if not self._ready or not self.model:
            # Return default languages when model is not loaded
            return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                    "Spanish", "Russian", "Portuguese", "Italian"]
        
        try:
            if hasattr(self.model.model, 'get_supported_languages'):
                languages = self.model.model.get_supported_languages()
                if languages:
                    return list(languages)
        except Exception as e:
            logger.warning(f"Could not get languages from model: {e}")
        
        # Fallback to default languages
        return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                "Spanish", "Russian", "Portuguese", "Italian"]
    
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                info["gpu_available"] = True
                if torch.cuda.current_device() >= 0:
                    device_idx = torch.cuda.current_device()
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                    
                    # Get VRAM info
                    props = torch.cuda.get_device_properties(device_idx)
                    info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"
                    
                    if self._ready:
                        allocated = torch.cuda.memory_allocated(device_idx)
                        info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        return info
