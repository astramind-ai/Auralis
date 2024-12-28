import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Tuple, List
import sounddevice as sd

from IPython.display import Audio, display


import numpy as np
import torch
import torchaudio
from torio.io import CodecConfig


@dataclass
class TTSOutput:
    """Container for TTS inference output with integrated audio utilities"""
    array: Union[np.ndarray, bytes]
    sample_rate: int = 24000
    bit_depth: int = 32
    bit_rate: int = 192 # kbps
    compression: int = 10 #
    channel: int = 1

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    token_length: Optional[int] = None


    def __post_init__(self):
        if isinstance(self.array, bytes):
            self.array = np.frombuffer(self.array, dtype=np.int16)
            #normalize in the range
            self.array = self.array.astype(np.float32) / 32768.0
            fade_length = 100
            fade_in = np.linspace(0, 1, fade_length)
            self.array[:fade_length] *= fade_in

    def change_speed(self, speed_factor: float) -> 'TTSOutput':
        """
        Change audio speed while preserving quality and minimizing distortion.
        Uses phase vocoder for better quality at extreme speed changes.

        Args:
            speed_factor (float): Speed modification factor:
                                 > 1.0: speeds up (e.g. 1.2 = 20% faster)
                                 < 1.0: slows down (e.g. 0.8 = 20% slower)
                                 = 1.0: no change

        Returns:
            TTSOutput: New instance with speed-modified audio

        Example:
            # Speed up 20%
            faster = audio.change_speed(1.2)

            # Slow down 20%
            slower = audio.change_speed(0.8)

        Raises:
            ValueError: If speed_factor is <= 0
        """
        import librosa

        # Validate input
        if speed_factor <= 0:
            raise ValueError("Speed factor must be positive")

        if speed_factor == 1.0:
            return self

        # Ensure float32
        wav = self.array.astype(np.float32) if self.array.dtype != np.float32 else self.array

        # Parameters for STFT
        n_fft = 2048
        hop_length = 512

        # Compute STFT
        D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)

        # Time-stretch using phase vocoder
        modified_stft = librosa.phase_vocoder(
            D,
            rate=speed_factor,
            hop_length=hop_length
        )

        # Inverse STFT
        modified = librosa.istft(
            modified_stft,
            hop_length=hop_length,
            length=len(wav)
        )

        # Normalize to prevent clipping
        modified = librosa.util.normalize(modified, norm=np.inf)

        return TTSOutput(
            array=modified,
            sample_rate=self.sample_rate
        )

    @staticmethod
    def combine_outputs(outputs: List['TTSOutput']) -> 'TTSOutput':
        """Combine multiple TTSOutput instances into a single instance.

        Args:
            outputs: List of TTSOutput instances

        Returns:
            New TTSOutput instance with concatenated audio
        """
        # Concatenate audio
        combined_audio = np.concatenate([out.array for out in outputs])

        # Use sample rate of first output
        return TTSOutput(
            array=combined_audio,
            sample_rate=outputs[0].sample_rate
        )

    def to_tensor(self) -> Union[torch.Tensor, np.ndarray]:
        """Convert numpy array to torch tensor"""
        if isinstance(self.array, np.ndarray):
            return torch.from_numpy(self.array)
        return self.array

    def to_bytes(self, format: str = 'wav', sample_width: int = 2) -> bytes:
        """Convert audio to bytes format.

        Args:
            format: Output format ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')
            sample_width: Bit depth (1, 2, or 4 bytes per sample)

        Returns:
            Audio data as bytes
        """
        # Convert to tensor if needed
        wav_tensor = self.to_tensor().to(torch.float32)

        # Ensure correct shape (1, N) for torchaudio
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Normalize to [-1, 1]
        wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)

        buffer = io.BytesIO()

        if format in ['wav', 'flac']:
            torchaudio.save(
                buffer,
                wav_tensor,
                self.sample_rate,
                format=format,
                encoding="PCM_S" if sample_width == 2 else "PCM_F",
                bits_per_sample=sample_width * 8,
                compression = CodecConfig(compression_level=min(8, self.compression)) if format == 'flac' else None
            )
        elif format == 'mp3':
            torchaudio.save(
                buffer,
                wav_tensor,
                self.sample_rate,
                format="mp3",
                compression = CodecConfig(bit_rate=self.bit_rate)
            )
        elif format == 'opus':
            torchaudio.save(
                buffer,
                wav_tensor,
                self.sample_rate,
                format="opus",
                compression = CodecConfig(compression_level=self.compression)
            )
        elif format == 'aac':
            torchaudio.save(
                buffer,
                wav_tensor,
                self.sample_rate,
                format="adts",
                compression = CodecConfig(bit_rate=self.bit_rate)
            )
        elif format == 'pcm':
            # Scale to appropriate range based on sample width
            if sample_width == 2:  # 16-bit
                wav_tensor = (wav_tensor * 32767).to(torch.int16)
            elif sample_width == 4:  # 32-bit
                wav_tensor = (wav_tensor * 2147483647).to(torch.int32)
            else:  # 8-bit
                wav_tensor = (wav_tensor * 127).to(torch.int8)
            return wav_tensor.cpu().numpy().tobytes()
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats are: mp3, opus, aac, flac, wav, pcm")

        return buffer.getvalue()

    def save(self,
             filename: Union[str, Path],
             sample_rate: Optional[int] = None,
             format: Optional[str] = None) -> None:
        """Save audio to file.

        Args:
            filename: Output filename
            sample_rate: Optional new sample rate for resampling
            format: Optional format override (default: inferred from extension)
        """
        wav_tensor = self.to_tensor()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Resample if needed
        if sample_rate and sample_rate != self.sample_rate:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor,
                orig_freq=self.sample_rate,
                new_freq=sample_rate
            )
        else:
            sample_rate = self.sample_rate
        if wav_tensor.dtype != torch.float32:
            wav_tensor = wav_tensor.to(torch.float32)
        torchaudio.save(
            filename,
            wav_tensor,
            sample_rate,
            format=format,
            bits_per_sample=self.bit_depth,
            channels_first=self.channel
        )

    def resample(self, new_sample_rate: int) -> 'TTSOutput':
        """Create new TTSOutput with resampled audio.

        Args:
            new_sample_rate: Target sample rate

        Returns:
            New TTSOutput instance with resampled audio
        """
        wav_tensor = self.to_tensor()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        resampled = torchaudio.functional.resample(
            wav_tensor,
            orig_freq=self.sample_rate,
            new_freq=new_sample_rate
        )

        return TTSOutput(
            array=resampled.squeeze().numpy(),
            sample_rate=new_sample_rate
        )

    def get_info(self) -> Tuple[int, int, float]:
        """Get audio information.

        Returns:
            Tuple of (number of samples, sample rate, duration in seconds)
        """
        n_samples = len(self.array)
        duration = n_samples / self.sample_rate
        return n_samples, self.sample_rate, duration

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, sample_rate: int = 24000) -> 'TTSOutput':
        """Create TTSOutput from torch tensor.

        Args:
            tensor: Audio tensor
            sample_rate: Sample rate of the audio

        Returns:
            New TTSOutput instance
        """
        return cls(
            array=tensor.squeeze().cpu().numpy(),
            sample_rate=sample_rate
        )

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'TTSOutput':
        """Create TTSOutput from audio file.

        Args:
            filename: Path to audio file

        Returns:
            New TTSOutput instance
        """
        wav_tensor, sample_rate = torchaudio.load(filename)
        return cls.from_tensor(wav_tensor, sample_rate)

    def play(self) -> None:
        """Play the audio through the default sound device.
        For use in regular Python scripts/applications."""
        # Ensure the audio is in the correct format
        if isinstance(self.array, torch.Tensor):
            audio_data = self.array.cpu().numpy()
        else:
            audio_data = self.array

        # Ensure float32 and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Play the audio
        sd.play(audio_data, self.sample_rate, blocksize=2048)
        sd.wait()  # Wait until the audio is finished playing

    def display(self) -> Optional[Audio]:
        """Display audio player in Jupyter notebook.
        Returns Audio widget if in notebook, None otherwise."""
        try:
            # Convert to bytes
            audio_bytes = self.to_bytes(format='wav')

            # Create and display audio widget
            audio_widget = Audio(audio_bytes, rate=self.sample_rate, autoplay=False)
            display(audio_widget)
            return audio_widget
        except Exception as e:
            print(f"Could not display audio widget: {str(e)}")
            print("Try using .play() method instead")
            return None

    def preview(self) -> None:
        """Smart play method that chooses appropriate playback method."""
        try:
            # Try notebook display first
            if self.display() is None:
                # Fall back to sounddevice if not in notebook
                self.play()
        except Exception as e:
            print(f"Error playing audio: {str(e)}")