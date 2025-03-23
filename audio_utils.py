from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Protocol, TypeVar
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from scipy import signal
from src.utils.logger import get_logger
from src.core.exceptions import AudioProcessingError

AudioData = TypeVar('AudioData', bound=np.ndarray)

class AudioProcessorProtocol(Protocol):
    """Protocol defining the interface for audio processing operations."""
    def normalize_audio(self, audio: AudioData, target_db: float = -20.0) -> AudioData: ...
    def trim_silence(self, audio: AudioData, threshold_db: float = -50.0) -> AudioData: ...
    def apply_noise_reduction(self, audio: AudioData, sample_rate: int) -> AudioData: ...

class AudioProcessor:
    """Utility class for audio processing operations."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the AudioProcessor with settings.
        
        Args:
            sample_rate: The default sample rate to use for processing
        """
        self.logger = get_logger(__name__)
        self.default_sample_rate = sample_rate
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Set default audio processing parameters."""
        self.default_channels: int = 1  # Mono audio
        self.default_dtype = np.float32
        self.min_silence_duration: float = 0.1
        self.frame_length: int = 2048
        self.hop_length: int = 512

    def load_audio(
        self,
        file_path: Union[str, Path],
        target_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and optionally resample it.
        
        Args:
            file_path: Path to the audio file
            target_sr: Target sampling rate (None to keep original)
        
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioProcessingError: If loading fails
        """
        try:
            file_path = Path(file_path)
            audio, sr = librosa.load(str(file_path), sr=target_sr)
            return audio, sr
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio file: {str(e)}") from e

    def save_audio(
        self,
        audio: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: Optional[int] = None
    ) -> Path:
        """
        Save audio data to a file.
        
        Args:
            audio: Audio data as numpy array
            file_path: Output file path
            sample_rate: Sampling rate of the audio (defaults to default_sample_rate)
        
        Returns:
            Path to the saved file
            
        Raises:
            AudioProcessingError: If saving fails
        """
        try:
            file_path = Path(file_path)
            sf.write(file_path, audio, sample_rate or self.default_sample_rate)
            return file_path
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio file: {str(e)}") from e

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to a target sampling rate.
        
        Args:
            audio: Input audio data
            orig_sr: Original sampling rate
            target_sr: Target sampling rate
        
        Returns:
            Resampled audio data
            
        Raises:
            AudioProcessingError: If resampling fails
        """
        try:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            raise AudioProcessingError(f"Resampling failed: {str(e)}") from e

    def normalize_audio(
        self,
        audio: np.ndarray,
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio to a target dB level.
        
        Args:
            audio: Input audio data
            target_db: Target dB level
        
        Returns:
            Normalized audio data
            
        Raises:
            AudioProcessingError: If normalization fails
        """
        try:
            return librosa.util.normalize(audio) * (10 ** (target_db / 20.0))
        except Exception as e:
            raise AudioProcessingError(f"Normalization failed: {str(e)}") from e

    def apply_noise_reduction(
        self,
        audio: np.ndarray,
        sample_rate: int,
        noise_clip: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply noise reduction to audio data.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sampling rate
            noise_clip: Optional noise profile for reduction
        
        Returns:
            Noise-reduced audio data
            
        Raises:
            AudioProcessingError: If noise reduction fails
        """
        try:
            if noise_clip is None:
                # Estimate noise from silent regions
                S = librosa.stft(
                    audio,
                    n_fft=self.frame_length,
                    hop_length=self.hop_length
                )
                mag = np.abs(S)
                power = mag ** 2
                noise_power = np.mean(power[:, :10], axis=1, keepdims=True)
                
                # Apply soft thresholding
                mask = (power > noise_power).astype(np.float32)
                S_clean = S * mask
                
                return librosa.istft(S_clean, hop_length=self.hop_length)
            else:
                # Use provided noise profile
                noise_power = np.mean(np.abs(librosa.stft(noise_clip)) ** 2, axis=1)
                return self._spectral_subtraction(audio, noise_power, sample_rate)
        except Exception as e:
            raise AudioProcessingError(f"Noise reduction failed: {str(e)}") from e

    def _spectral_subtraction(
        self,
        audio: np.ndarray,
        noise_power: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.
        
        Args:
            audio: Input audio data
            noise_power: Pre-computed noise power spectrum
            sample_rate: Audio sampling rate
        
        Returns:
            Processed audio data
        """
        S = librosa.stft(
            audio,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        mag = np.abs(S)
        phase = np.angle(S)
        
        # Apply spectral subtraction
        power = mag ** 2
        power_clean = np.maximum(power - noise_power[:, np.newaxis], 0)
        mag_clean = np.sqrt(power_clean)
        
        # Reconstruct signal
        S_clean = mag_clean * np.exp(1j * phase)
        return librosa.istft(S_clean, hop_length=self.hop_length)

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -50.0,
        min_silence_duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of the audio.
        
        Args:
            audio: Input audio data
            threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration in seconds
        
        Returns:
            Trimmed audio data
            
        Raises:
            AudioProcessingError: If trimming fails
        """
        try:
            return librosa.effects.trim(
                audio,
                top_db=-threshold_db,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
        except Exception as e:
            raise AudioProcessingError(f"Audio trimming failed: {str(e)}") from e

    def split_on_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        min_segment_length: float = 0.5,
        silence_threshold: float = 0.01
    ) -> list[np.ndarray]:
        """
        Split audio into segments based on silence.
        
        Args:
            audio: Input audio data
            sample_rate: Audio sampling rate
            min_segment_length: Minimum segment length in seconds
            silence_threshold: Threshold for silence detection
            
        Returns:
            List of audio segments
            
        Raises:
            AudioProcessingError: If splitting fails
        """
        try:
            # Convert parameters to samples
            min_samples = int(min_segment_length * sample_rate)
            
            # Find silence intervals
            energy = np.abs(audio)
            silence_mask = energy < silence_threshold
            
            # Find silence boundaries
            silence_starts = np.where(np.diff(silence_mask.astype(int)) == 1)[0]
            silence_ends = np.where(np.diff(silence_mask.astype(int)) == -1)[0]
            
            if len(silence_starts) == 0 or len(silence_ends) == 0:
                return [audio]
                
            # Ensure matching starts and ends
            if silence_starts[0] > silence_ends[0]:
                silence_starts = np.concatenate(([0], silence_starts))
            if silence_ends[-1] < silence_starts[-1]:
                silence_ends = np.concatenate((silence_ends, [len(audio)]))
                
            # Split audio
            segments = []
            for start, end in zip(silence_ends, silence_starts[1:]):
                if end - start >= min_samples:
                    segments.append(audio[start:end])
            
            return segments
        except Exception as e:
            raise AudioProcessingError(f"Audio splitting failed: {str(e)}") from e

    def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Union[int, float]]:
        """
        Get audio file information.
        
        Args:
            file_path: Path to the audio file
        
        Returns:
            Dictionary containing audio properties
            
        Raises:
            AudioProcessingError: If getting info fails
        """
        try:
            with sf.SoundFile(file_path) as audio_file:
                return {
                    "sample_rate": audio_file.samplerate,
                    "channels": audio_file.channels,
                    "duration": len(audio_file) / audio_file.samplerate,
                    "format": audio_file.format,
                    "subtype": audio_file.subtype
                }
        except Exception as e:
            raise AudioProcessingError(f"Failed to get audio info: {str(e)}") from e

    def process_chunks(
        self,
        audio: np.ndarray,
        chunk_size: int = 32000,
        process_fn: callable = None
    ) -> np.ndarray:
        """
        Process audio in chunks to handle large files.
        
        Args:
            audio: Input audio data
            chunk_size: Size of chunks to process
            process_fn: Optional function to process each chunk
            
        Returns:
            Processed audio data
        """
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
        processed_chunks = []
        
        for chunk in chunks:
            if process_fn:
                chunk = process_fn(chunk)
            processed_chunks.append(chunk)
        
        return np.concatenate(processed_chunks)

    def cleanup(self) -> None:
        """Clean up any resources held by the audio processor."""
        # Currently no cleanup needed, but method included for future use
        pass
