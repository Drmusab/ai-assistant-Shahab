from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable
import tempfile
from datetime import datetime
import asyncio
import time

import numpy as np
import torch
import whisper
import sounddevice as sd
import soundfile as sf

from src.core.exceptions import STTError
from src.core.config import Config
from src.utils.logger import get_logger
from src.speech.voice_activity import VoiceActivityDetector


class WhisperTranscriber:
    """
    Enhanced Speech recognition system using OpenAI's Whisper model with real-time VAD support.
    Last updated: 2025-03-01 by Drmusab
    """

    # Supported Whisper models and their configurations
    AVAILABLE_MODELS = {
        "tiny": {"size": 39, "multilingual": False},
        "base": {"size": 74, "multilingual": False},
        "small": {"size": 244, "multilingual": False},
        "medium": {"size": 769, "multilingual": False},
        "large": {"size": 1550, "multilingual": True},
        "large-v2": {"size": 1550, "multilingual": True},
    }

    def __init__(self, config: Config) -> None:
        """
        Initialize the Whisper-based speech recognition system.
        
        Args:
            config: Application configuration instance.
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        # Device selection with M1/M2 Mac support
        self.device = self._get_optimal_device()
        
        # Audio settings initialization and model loading
        self._setup_audio_config()
        self._load_model()
        
        # Initialize VAD for improved voice detection
        self.vad = VoiceActivityDetector(config)
        
        # Performance monitoring
        self.last_transcription_time: float = 0
        self.total_transcriptions: int = 0
        
        self.logger.info(
            f"WhisperTranscriber initialized (Model: {self.model_name}, "
            f"Device: {self.device}, VAD: Enabled)"
        )

    def _get_optimal_device(self) -> torch.device:
        """
        Determine the optimal device for model inference.
        Supports CUDA, MPS (M1/M2 Macs), and CPU.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_audio_config(self) -> None:
        """Configure audio recording settings with enhanced error checking."""
        try:
            self.sample_rate: int = 16000  # Whisper requires 16kHz
            self.channels: int = 1         # Mono audio
            self.chunk_size: int = int(self.config.get("speech.input.chunk_size", 1024))
            self.audio_format = np.float32

            # Validate audio configuration
            sd.check_input_settings(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.audio_format
            )

            # Create dedicated temporary directory with cleanup
            self.temp_dir = Path(tempfile.gettempdir()) / "whisper_audio"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Cleanup old temporary files
            self._cleanup_old_files()

        except Exception as e:
            raise STTError(f"Audio configuration failed: {str(e)}") from e

    def _cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """Clean up old temporary files."""
        try:
            current_time = time.time()
            for file in self.temp_dir.glob("*.wav"):
                if (current_time - file.stat().st_mtime) > (max_age_hours * 3600):
                    file.unlink()
        except Exception as e:
            self.logger.warning(f"Cleanup of old files failed: {str(e)}")

    def _load_model(self) -> None:
        """Load and initialize the Whisper model with optimization options."""
        try:
            self.model_name: str = self.config.get("speech.models.whisper.model", "base")
            if self.model_name not in self.AVAILABLE_MODELS:
                valid_models = ", ".join(self.AVAILABLE_MODELS.keys())
                raise ValueError(f"Invalid model name. Available models: {valid_models}")

            # Load model with appropriate compute type
            compute_type = "float16" if self.device.type in ["cuda", "mps"] else "float32"
            self.model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=self.config.get("speech.models.whisper.path", None),
                in_memory=self.config.get("speech.models.whisper.in_memory", True)
            ).to(self.device)

            # Set model attributes
            self.multilingual = self.AVAILABLE_MODELS[self.model_name]["multilingual"]
            
        except Exception as e:
            raise STTError(f"Failed to load Whisper model: {str(e)}") from e

    async def listen(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.0,
        energy_threshold: float = 0.005,
        on_speech_detected: Optional[Callable[[float], None]] = None
    ) -> np.ndarray:
        """
        Record audio with enhanced voice activity detection.
        
        Args:
            duration: Maximum recording duration in seconds.
            silence_threshold: Volume threshold to determine silence.
            silence_duration: Duration (in seconds) of silence to stop recording.
            energy_threshold: Minimum volume level to start recording.
            on_speech_detected: Optional callback when speech is detected.
        
        Returns:
            Recorded audio as a numpy array.
        """
        # Use VAD for improved voice detection
        voice_detected, audio_data = await self.vad.listen_for_voice(
            duration=duration,
            silence_threshold=silence_threshold,
            min_speech_duration=0.3
        )
        
        if voice_detected and on_speech_detected:
            on_speech_detected(time.time())
            
        return audio_data

    async def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced transcribe function with beam search and temperature control.
        
        Args:
            audio: Audio data as numpy array or path to the audio file.
            language: Language code for transcription.
            task: Whisper task - 'transcribe' or 'translate'.
            beam_size: Beam size for beam search decoding.
            temperature: Temperature for sampling. None uses config value.
        
        Returns:
            Dictionary containing transcription results with metadata.
        """
        start_time = time.time()
        temp_file_created = False
        
        try:
            # Handle audio input
            if isinstance(audio, (str, Path)):
                audio_path = Path(audio)
            else:
                audio_path = self._write_temp_audio(audio)
                temp_file_created = True

            # Configure transcription options
            options = whisper.DecodingOptions(
                language=language if self.multilingual else "en",
                task=task,
                beam_size=beam_size,
                fp16=self.device.type in ["cuda", "mps"],
                temperature=temperature or self.config.get(
                    "speech.models.whisper.temperature", 
                    0
                )
            )

            # Perform transcription
            result = self.model.transcribe(
                str(audio_path),
                **options.__dict__
            )
            
            # Update metrics
            self.last_transcription_time = time.time() - start_time
            self.total_transcriptions += 1

            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", "en"),
                "task": task,
                "metadata": {
                    "processing_time": self.last_transcription_time,
                    "model": self.model_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "audio_duration": result.get("duration", 0),
                    "confidence": result.get("confidence", 0)
                }
            }

        except Exception as e:
            raise STTError(f"Transcription failed: {str(e)}") from e
        finally:
            if temp_file_created and audio_path.exists():
                audio_path.unlink()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    async def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        language: Optional[str] = None,
        task: str = "transcribe",
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Batch transcribe multiple audio files with concurrency control.
        
        Args:
            audio_files: List of paths to audio files.
            language: Language code for transcription.
            task: Whisper task - 'transcribe' or 'translate'.
            max_concurrent: Maximum number of concurrent transcriptions.
        
        Returns:
            List of transcription result dictionaries.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def transcribe_with_semaphore(file: Union[str, Path]) -> Dict[str, Any]:
            async with semaphore:
                return await self.transcribe(file, language, task)

        tasks = [transcribe_with_semaphore(file) for file in audio_files]
        return await asyncio.gather(*tasks)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the transcriber."""
        return {
            "total_transcriptions": self.total_transcriptions,
            "last_transcription_time": self.last_transcription_time,
            "device": str(self.device),
            "model": self.model_name,
            "multilingual": self.multilingual
        }

    def get_available_languages(self) -> List[str]:
        """Get the list of supported languages based on the current model."""
        return whisper.tokenizer.LANGUAGES if self.multilingual else ["en"]