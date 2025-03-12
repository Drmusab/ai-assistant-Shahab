#!/usr/bin/env python3
"""
Voice Activity Detection Module
Handles voice activity detection and speech segmentation for the AI Assistant.

"""
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import sounddevice as sd
from datetime import datetime
import webrtcvad
import asyncio

from src.core.exceptions import VADError
from src.core.config import Config
from src.utils.logger import get_logger


class VoiceActivityDetector:
    """Voice Activity Detection (VAD) system using WebRTC VAD."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the Voice Activity Detection system.
        
        Args:
            config: Application configuration instance.
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        # Audio settings
        self.sample_rate: int = 16000  # WebRTC VAD requires 16kHz
        self.channels: int = 1         # Mono audio required for VAD
        self.frame_duration: int = 30  # Frame duration in milliseconds (10, 20, or 30)
        self.vad_mode: int = 3        # Aggressiveness mode (0-3)
        
        # Initialize WebRTC VAD
        self._setup_vad()
        
        self.logger.info(
            f"VoiceActivityDetector initialized (Mode: {self.vad_mode}, "
            f"Frame Duration: {self.frame_duration}ms)"
        )

    def _setup_vad(self) -> None:
        """Configure and initialize the WebRTC VAD instance."""
        try:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(self.vad_mode)
            
            # Calculate frame size based on sample rate and frame duration
            self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
            self.logger.debug(f"VAD frame size: {self.frame_size} samples")
            
        except Exception as e:
            raise VADError(f"Failed to initialize WebRTC VAD: {str(e)}") from e

    async def detect_voice_activity(
        self,
        audio: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Detect voice activity in an audio frame.
        
        Args:
            audio: Audio data as numpy array.
            threshold: Minimum ratio of voice frames to consider as speech.
            
        Returns:
            Tuple containing:
            - bool: True if voice activity detected, False otherwise
            - float: Confidence score (0.0 to 1.0)
        """
        try:
            # Ensure audio is in the correct format
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)
            
            # Split audio into frames
            frames = self._frame_audio(audio)
            
            # Process each frame
            voice_frames = 0
            total_frames = len(frames)
            
            for frame in frames:
                try:
                    if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                        voice_frames += 1
                except Exception as e:
                    self.logger.warning(f"Error processing VAD frame: {str(e)}")
                    continue
                
                # Give control back to event loop periodically
                await asyncio.sleep(0)
            
            if total_frames == 0:
                return False, 0.0
                
            confidence = voice_frames / total_frames
            is_speech = confidence >= threshold
            
            return is_speech, confidence
            
        except Exception as e:
            raise VADError(f"Voice activity detection failed: {str(e)}") from e

    def _frame_audio(self, audio: np.ndarray) -> list:
        """
        Split audio into frames suitable for WebRTC VAD.
        
        Args:
            audio: Audio data as numpy array.
            
        Returns:
            List of audio frames.
        """
        # Ensure audio length is multiple of frame size
        remainder = len(audio) % self.frame_size
        if remainder != 0:
            padding = np.zeros(self.frame_size - remainder, dtype=np.int16)
            audio = np.concatenate([audio, padding])
        
        # Split into frames
        frames = []
        for i in range(0, len(audio), self.frame_size):
            frames.append(audio[i:i + self.frame_size])
            
        return frames

    async def listen_for_voice(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.1,
        min_speech_duration: float = 0.3
    ) -> Tuple[bool, np.ndarray]:
        """
        Listen for voice activity in real-time audio stream.
        
        Args:
            duration: Maximum recording duration in seconds.
            silence_threshold: Volume threshold to determine silence.
            min_speech_duration: Minimum duration of speech to trigger detection.
            
        Returns:
            Tuple containing:
            - bool: True if voice detected within duration
            - np.ndarray: Recorded audio data
        """
        audio_chunks = []
        voice_detected = False
        speech_duration = 0.0
        
        def audio_callback(indata: np.ndarray, frames: int, time_info: Dict, status: Any) -> None:
            nonlocal speech_duration
            if status:
                self.logger.warning(f"Audio input status: {status}")
            
            # Add audio chunk to buffer
            audio_chunks.append(indata.copy())
            
            # Quick energy-based voice detection for immediate feedback
            energy = np.mean(np.abs(indata))
            if energy > silence_threshold:
                speech_duration += frames / self.sample_rate
                if speech_duration >= min_speech_duration:
                    voice_detected = True

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=audio_callback
            )
            
            with stream:
                await asyncio.sleep(duration if duration else 5.0)
                
            # Combine all audio chunks
            audio_data = np.concatenate(audio_chunks) if audio_chunks else np.array([])
            
            return voice_detected, audio_data
            
        except Exception as e:
            raise VADError(f"Audio recording failed: {str(e)}") from e

    def save_audio(self, audio: np.ndarray, file_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save recorded audio to a file.
        
        Args:
            audio: Audio data as numpy array.
            file_path: Optional path to save the file.
            
        Returns:
            Path to the saved audio file.
        """
        import soundfile as sf
        
        if file_path is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
            file_path = Path(self.config.get(
                "speech.vad.recordings_dir",
                "data/recordings"
            )) / f"vad_recording_{timestamp}.wav"
            
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            sf.write(str(file_path), audio, self.sample_rate)
            self.logger.info(f"Audio saved to: {file_path}")
            return file_path
        except Exception as e:
            raise VADError(f"Failed to save audio file: {str(e)}") from e
        
        """Uses WebRTC VAD for robust voice activity detection
Integrates with the existing project structure using Config and logging
Provides async methods for real-time voice detection
Includes:
Frame-based voice activity detection
Real-time audio monitoring
Audio file saving capabilities
Configurable sensitivity and thresholds
Key features:

WebRTC VAD integration for industry-standard voice detection
Async support for non-blocking operation
Configurable parameters (VAD mode, frame duration, thresholds)
Error handling and logging
Audio file management
Integration with the existing audio pipeline
Dependencies needed to be added to requirements.txt:

plaintext
webrtcvad>=2.0.10
The code uses the same audio configuration (16kHz, mono) as the WhisperTranscriber class to ensure compatibility. It provides both high-level methods for simple voice detection and lower-level access to the VAD engine for more complex use cases.

You would typically use it like this:

Python
# Initialize
vad = VoiceActivityDetector(config)

# Real-time voice detection
voice_detected, audio = await vad.listen_for_voice(duration=5.0)

if voice_detected:
    # Detailed analysis of the recorded audio
    is_speech, confidence = await vad.detect_voice_activity(audio)
    if is_speech:
        # Save the audio
        file_path = vad.save_audio(audio)
        # Process with speech recognition
        # ...
        """