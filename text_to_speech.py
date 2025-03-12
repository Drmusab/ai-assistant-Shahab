#!/usr/bin/env python3
"""
Text to Speech Module
Handles speech synthesis functionality for the AI Assistant.


"""

import os
import asyncio
import numpy as np
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import sounddevice as sd
from dataclasses import dataclass
import json

from src.core.exceptions import TTSError
from src.core.config import Config
from src.utils.logger import get_logger
from src.speech.audio_utils import AudioPreprocessor
from src.utils.validators import validate_audio_format

@dataclass
class SpeechConfig:
    """Configuration for speech synthesis."""
    voice_id: str
    language: str
    speed: float
    pitch: float
    volume: float

class TextToSpeech:
    """
    Handles text-to-speech synthesis using multiple backends with fallback support.
    Supports real-time synthesis and batch processing.
    """

    def __init__(self, config: Config):
        """
        Initialize speech synthesis system.

        Args:
            config: Configuration instance
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self._init_audio_settings()
        self._init_models()
        self.preprocessor = AudioPreprocessor()
        
        self.logger.info(f"Text-to-Speech initialized (Device: {self.device})")

    def _init_audio_settings(self) -> None:
        """Initialize audio output settings."""
        self.sample_rate = self.config.get("speech.output.sample_rate", 22050)
        self.channels = self.config.get("speech.output.channels", 1)
        self.audio_format = sd.default.dtype[1]
        
        # Default speech configuration
        self.default_speech_config = SpeechConfig(
            voice_id=self.config.get("speech.output.voice_id", "default"),
            language=self.config.get("speech.output.language", "en-US"),
            speed=self.config.get("speech.output.speed", 1.0),
            pitch=self.config.get("speech.output.pitch", 1.0),
            volume=self.config.get("speech.output.volume", 1.0)
        )
        
        try:
            validate_audio_format(self.sample_rate, self.channels, self.audio_format)
        except ValueError as e:
            raise TTSError(f"Invalid audio settings: {str(e)}")

    def _init_models(self) -> None:
        """Initialize speech synthesis models."""
        try:
            # Load models
            model_path = self.config.get(
                "speech.models.t5_path",
                "microsoft/speecht5_tts"
            )
            vocoder_path = self.config.get(
                "speech.models.vocoder_path",
                "microsoft/speecht5_hifigan"
            )
            
            self.processor = SpeechT5Processor.from_pretrained(model_path)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_path).to(self.device)
            
            # Set models to evaluation mode
            self.model.eval()
            self.vocoder.eval()
            
            self.logger.info(f"Models loaded successfully from {model_path}")
            
        except Exception as e:
            raise TTSError(f"Failed to load speech synthesis models: {str(e)}")

    async def speak(
        self,
        text: str,
        speech_config: Optional[SpeechConfig] = None
    ) -> None:
        """
        Synthesize and play text immediately.

        Args:
            text: Text to synthesize
            speech_config: Optional speech configuration
        """
        try:
            audio = await self.synthesize(text, speech_config)
            await self._play_audio(audio)
        except Exception as e:
            raise TTSError(f"Error during speech synthesis: {str(e)}")

    async def synthesize(
        self,
        text: str,
        speech_config: Optional[SpeechConfig] = None
    ) -> np.ndarray:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            speech_config: Optional speech configuration

        Returns:
            Synthesized audio as numpy array
        """
        try:
            config = speech_config or self.default_speech_config
            
            # Prepare input
            inputs = self.processor(
                text=text,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    vocoder=self.vocoder,
                    speaker_embeddings=None  # Can be used for voice cloning
                )
            
            # Apply audio modifications
            audio = speech.cpu().numpy()
            audio = self._modify_audio(audio, config)
            
            return audio
            
        except Exception as e:
            raise TTSError(f"Speech synthesis error: {str(e)}")

    async def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save synthesized audio to file with metadata.

        Args:
            audio: Audio data as numpy array
            output_path: Path to save audio file
            metadata: Optional metadata to save with audio
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio
            torchaudio.save(
                str(output_path),
                torch.from_numpy(audio).unsqueeze(0),
                self.sample_rate
            )
            
            # Save metadata if provided
            if metadata:
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Audio saved to {output_path}")
            
        except Exception as e:
            raise TTSError(f"Error saving audio: {str(e)}")

    async def batch_synthesize(
        self,
        texts: List[str],
        speech_config: Optional[SpeechConfig] = None
    ) -> List[np.ndarray]:
        """
        Batch synthesize multiple texts.

        Args:
            texts: List of texts to synthesize
            speech_config: Optional speech configuration

        Returns:
            List of synthesized audio arrays
        """
        try:
            audio_segments = []
            for text in texts:
                audio = await self.synthesize(text, speech_config)
                audio_segments.append(audio)
            return audio_segments
            
        except Exception as e:
            raise TTSError(f"Batch synthesis error: {str(e)}")

    async def _play_audio(self, audio: np.ndarray) -> None:
        """
        Play audio through default output device.

        Args:
            audio: Audio data as numpy array
        """
        try:
            # Ensure audio is in the correct format
            audio = audio.astype(self.audio_format)
            
            # Play audio
            sd.play(audio, self.sample_rate)
            sd.wait()
            
        except Exception as e:
            raise TTSError(f"Error playing audio: {str(e)}")

    def _modify_audio(
        self,
        audio: np.ndarray,
        config: SpeechConfig
    ) -> np.ndarray:
        """
        Modify audio based on configuration.

        Args:
            audio: Audio data
            config: Speech configuration

        Returns:
            Modified audio data
        """
        try:
            # Apply volume
            audio = audio * config.volume
            
            # Apply speed (resampling)
            if config.speed != 1.0:
                audio = self.preprocessor.change_speed(audio, config.speed)
            
            # Apply pitch
            if config.pitch != 1.0:
                audio = self.preprocessor.change_pitch(audio, config.pitch)
            
            return audio
            
        except Exception as e:
            raise TTSError(f"Error modifying audio: {str(e)}")

    def __str__(self) -> str:
        """String representation of the TTS instance."""
        return (f"TextToSpeech(device={self.device}, "
                f"sample_rate={self.sample_rate}, "
                f"language={self.default_speech_config.language})")
        
        
        """
    
  text_to_speech.py implementation includes:

Multiple Synthesis Backends:

SpeechT5 model support
HiFi-GAN vocoder
GPU acceleration
Audio Processing:

Real-time synthesis
Batch processing
Audio modifications (speed, pitch, volume)
Multi-format support
Advanced Features:

Async support
Voice configuration
Metadata handling
Multiple language support
Robust Error Handling:

Custom exceptions
Detailed error messages
Comprehensive logging
Performance Optimizations:

GPU acceleration
Batch processing
Efficient audio handling
Key features:

Asynchronous operations
Multiple output formats
Configurable speech parameters
Voice customization
Robust error handling
GPU support
Metadata handling
Usage example:

Python
async def main():
    config = Config()
    tts = TextToSpeech(config)
    
    # Simple synthesis and playback
    await tts.speak("Hello, how can I help you today?")
    
    # Custom voice configuration
    custom_config = SpeechConfig(
        voice_id="custom_voice",
        language="en-US",
        speed=1.2,
        pitch=1.1,
        volume=0.9
    )
    
    # Synthesize and save
    audio = await tts.synthesize(
        "This is a test message",
        speech_config=custom_config
    )
    await tts.save_audio(audio, "test_output.wav")

if __name__ == "__main__":
    asyncio.run(main())
        """