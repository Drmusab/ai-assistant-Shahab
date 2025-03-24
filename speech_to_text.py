"""
Speech to Text Transcription Module using Whisper
Author: Drmusab
Last Modified: 2025-03-24 00:35:12 UTC
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import tempfile
from datetime import datetime
import asyncio

import numpy as np
import torch
import whisper
import sounddevice as sd
import soundfile as sf

from src.speech.audio_utils import (
    AudioProcessor,
    AudioProcessingError,
    AudioProcessorProtocol,  # Import instead of redefining
    AudioData  # Import instead of redefining
)
from src.core.exceptions import STTError
from src.core.config import Config
from src.utils.logger import get_logger

class WhisperTranscriber:
    """Optimized Speech recognition system using OpenAI's Whisper model."""

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Audio settings initialization
        self._setup_audio_config()
        
        # Initialize audio processor with matching sample rate
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)
        # Removed redundant logger handler assignment
        
        # Load model
        self._load_model()
        
        # Configure preprocessing settings
        self._setup_preprocessing()

        self.logger.info(
            f"WhisperTranscriber initialized (Model: {self.model_name}, Device: {self.device})"
        )

    def _setup_preprocessing(self) -> None:
        """Configure audio preprocessing settings from config."""
        self.trim_enabled = self.config.get("speech.preprocessing.trim_silence", True)
        self.normalize_enabled = self.config.get("speech.preprocessing.normalize", True)
        self.noise_reduction_enabled = self.config.get("speech.preprocessing.noise_reduction", True)
        self.trim_threshold_db = self.config.get("speech.preprocessing.trim_threshold_db", -50.0)

    def _setup_audio_config(self) -> None:
        """Configure audio recording settings."""
        self.sample_rate = self.config.get("speech.input.sample_rate", 16000)
        self.channels = self.config.get("speech.input.channels", 1)
        self.chunk_size = int(self.config.get("speech.input.chunk_size", 1024))
        self.audio_format = np.float32

        # Create a dedicated temporary directory for audio files
        self.temp_dir = Path(tempfile.gettempdir()) / "whisper_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        """Load and initialize the Whisper model."""
        try:
            self.model_name = self.config.get("speech.models.whisper.model", "base")
            if self.model_name not in self.AVAILABLE_MODELS:
                valid_models = ", ".join(self.AVAILABLE_MODELS.keys())
                raise ValueError(f"Invalid model name. Available models: {valid_models}")

            self.model = whisper.load_model(self.model_name).to(self.device)
            self.multilingual = self.AVAILABLE_MODELS[self.model_name]["multilingual"]
        except Exception as e:
            raise STTError(f"Failed to load Whisper model: {str(e)}") from e

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps based on configuration settings.
        
        Args:
            audio: Input audio data
        
        Returns:
            Processed audio data
            
        Raises:
            STTError: If audio preprocessing fails
        """
        try:
            processed_audio = audio.copy()

            if self.trim_enabled:
                processed_audio = self.audio_processor.trim_silence(
                    processed_audio,
                    threshold_db=self.trim_threshold_db
                )

            if self.normalize_enabled:
                processed_audio = self.audio_processor.normalize_audio(processed_audio)

            if self.noise_reduction_enabled:
                processed_audio = self.audio_processor.apply_noise_reduction(
                    processed_audio,
                    self.sample_rate
                )

            return processed_audio
        except (AudioProcessingError, ValueError) as e:
            raise STTError(f"Audio preprocessing failed: {str(e)}") from e

    def process_large_audio(self, audio: np.ndarray, chunk_size: int = 32000) -> np.ndarray:
        """
        Process large audio files in chunks using AudioProcessor's process_chunks method.
        
        Args:
            audio: Input audio data
            chunk_size: Size of chunks to process
            
        Returns:
            Processed audio data
        """
        return self.audio_processor.process_chunks(
            audio,
            chunk_size=chunk_size,
            process_fn=self._preprocess_audio
        )

    async def listen(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.0,
        energy_threshold: float = 0.005
    ) -> np.ndarray:
        """
        Record audio with voice activity detection.
        
        Args:
            duration: Maximum recording duration in seconds
            silence_threshold: Threshold for detecting silence
            silence_duration: Duration of silence to stop recording
            energy_threshold: Threshold for detecting speech
            
        Returns:
            Recorded audio data
            
        Raises:
            STTError: If recording fails
        """
        audio_chunks: List[np.ndarray] = []
        is_recording = False
        silent_chunks = 0
        max_chunks = int((duration or float('inf')) * self.sample_rate / self.chunk_size)

        def audio_callback(indata: np.ndarray, *_):
            nonlocal is_recording, silent_chunks
            energy = np.mean(np.abs(indata))
            if not is_recording and energy > energy_threshold:
                is_recording = True
                silent_chunks = 0
            if is_recording:
                audio_chunks.append(indata.copy())
                silent_chunks = silent_chunks + 1 if energy < silence_threshold else 0

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.audio_format,
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                while len(audio_chunks) < max_chunks:
                    if is_recording and silent_chunks > int(silence_duration * self.sample_rate / self.chunk_size):
                        break
                    await asyncio.sleep(0.01)

            recorded_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([])
            return self._preprocess_audio(recorded_audio)
        except Exception as e:
            raise STTError(f"Recording failed: {str(e)}") from e

    def _write_temp_audio(self, audio: np.ndarray) -> Path:
        """
        Write audio data to a temporary WAV file.
        
        Args:
            audio: Audio data as a numpy array
            
        Returns:
            Path to the temporary audio file
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        temp_path = self.temp_dir / f"audio_{timestamp}.wav"
        sf.write(temp_path, audio, self.sample_rate)
        return temp_path

    async def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data or path to audio file
            language: Language code for transcription
            task: Whisper task - 'transcribe' or 'translate'
            
        Returns:
            Dictionary containing transcription results
            
        Raises:
            STTError: If transcription fails
        """
        temp_file_created = False
        audio_path = None
        
        try:
            if isinstance(audio, (str, Path)):
                # Pass target_sr during loading to avoid redundant resampling
                audio_data, file_sr = self.audio_processor.load_audio(
                    audio,
                    target_sr=self.sample_rate
                )
                processed_audio = self.process_large_audio(audio_data)
                audio_path = self._write_temp_audio(processed_audio)
                temp_file_created = True
            else:
                processed_audio = self.process_large_audio(audio)
                audio_path = self._write_temp_audio(processed_audio)
                temp_file_created = True

            options = whisper.DecodingOptions(
                language=language if self.multilingual else "en",
                task=task,
                fp16=torch.cuda.is_available()
            )

            result = self.model.transcribe(
                str(audio_path),
                **options.__dict__,
                temperature=self.config.get("speech.models.whisper.temperature", 0)
            )

            return {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", "en"),
                "task": task
            }

        except Exception as e:
            raise STTError(f"Transcription failed: {str(e)}") from e
        finally:
            if temp_file_created and audio_path and audio_path.exists():
                audio_path.unlink()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    async def batch_transcribe(
        self,
        audio_files: List[Union[str, Path]],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> List[Dict[str, Any]]:
        """
        Batch transcribe multiple audio files.
        
        Args:
            audio_files: List of paths to audio files
            language: Language code for transcription
            task: Whisper task - 'transcribe' or 'translate'
            
        Returns:
            List of transcription result dictionaries
        """
        transcriptions = []
        for file in audio_files:
            transcription = await self.transcribe(file, language, task)
            transcriptions.append(transcription)
        return transcriptions

    def get_available_languages(self) -> List[str]:
        """
        Get the list of supported languages based on the current model.
        
        Returns:
            List of supported language codes
        """
        return whisper.tokenizer.LANGUAGES if self.multilingual else ["en"]

    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            for file in self.temp_dir.glob('*.wav'):
                try:
                    file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {file}: {str(e)}")
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary directory: {str(e)}")
        
        # Call the audio processor's cleanup method
        if hasattr(self, 'audio_processor'):
            self.audio_processor.cleanup()
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
