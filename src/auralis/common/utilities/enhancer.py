import uuid
from dataclasses import dataclass, field
from typing import Union, AsyncGenerator, Optional, List, Literal, get_args
from functools import lru_cache
import numpy as np
import librosa
import torch
import torchaudio
import pyloudnorm

@dataclass
class AudioPreprocessingConfig:
    sample_rate: int = 22050
    normalize: bool = True
    trim_silence: bool = True
    remove_noise: bool = True
    enhance_speech: bool = True

    # VAD parameters
    vad_threshold: float = 0.02
    vad_frame_length: int = 1024*4

    # Noise reduction
    noise_reduce_margin: float = 1.0
    noise_reduce_frames: int = 25

    # Enhancement
    enhance_amount: float = 1.0

    # Normalization target
    target_lufs: float = -18.0


class EnhancedAudioProcessor:
    def __init__(self, config: AudioPreprocessingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    @torch.no_grad()
    def get_mel_spectrogram(audio: np.ndarray, sr: int) -> torch.Tensor:
        """Compute mel spectrogram efficiently using torch."""
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )(audio_tensor)
        return torch.log(torch.clamp(mel_spec, min=1e-5))

    def vad_split(self, audio: np.ndarray) -> np.ndarray:
        """Enhanced Voice Activity Detection using energy and spectral features."""
        # Compute short-time energy
        frame_length = self.config.vad_frame_length
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_length // 2)
        energy = np.sum(frames ** 2, axis=0)
        energy = energy / np.max(energy)

        # Compute spectral features
        mel_spec = self.get_mel_spectrogram(audio, self.config.sample_rate)
        spectral_sum = torch.sum(mel_spec, dim=1).numpy().squeeze()
        spectral_sum = spectral_sum / np.max(spectral_sum)

        # Resize signals to match
        if len(energy) > len(spectral_sum):
            # Interpolate spectral_sum to match energy length
            spectral_sum = np.interp(
                np.linspace(0, 1, len(energy)),
                np.linspace(0, 1, len(spectral_sum)),
                spectral_sum
            )
        else:
            # Interpolate energy to match spectral_sum length
            energy = np.interp(
                np.linspace(0, 1, len(spectral_sum)),
                np.linspace(0, 1, len(energy)),
                energy
            )

        # Combine features
        vad_signal = (energy + spectral_sum) / 2
        vad_mask = np.absolute(vad_signal) > self.config.vad_threshold

        # Apply mask (resizing to audio length)
        mask_upsampled = np.interp(
            np.linspace(0, 1, len(audio)),
            np.linspace(0, 1, len(vad_mask)),
            vad_mask.astype(float)
        )

        return audio * mask_upsampled

    def spectral_gating(self, audio: np.ndarray) -> np.ndarray:
        """Enhanced spectral noise reduction."""
        # Compute STFT
        D = librosa.stft(audio)
        mag, phase = librosa.magphase(D)

        # Estimate noise profile from lowest energy frames
        noise_profile = np.mean(np.sort(mag, axis=1)[:, :self.config.noise_reduce_frames], axis=1)
        noise_profile = noise_profile[:, None]

        # Create mask
        mask = (mag - noise_profile * self.config.noise_reduce_margin).clip(min=0)
        mask = mask / (mask + noise_profile)

        # Apply mask
        return librosa.istft(mask * D)

    def enhance_clarity(self, audio: np.ndarray) -> np.ndarray:
        """Enhance speech clarity using spectral shaping."""
        # Convert to frequency domain
        D = librosa.stft(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0))
        mag, phase = librosa.magphase(D)

        # Apply mild spectral shaping to enhance clarity
        freq_bins = np.fft.fftfreq(D.shape[0], 1 / self.config.sample_rate)
        clarity_boost = np.exp(-np.abs(freq_bins - 2000) / 1000) * self.config.enhance_amount
        clarity_boost = clarity_boost[:, None]

        mag_enhanced = mag * (1 + clarity_boost)

        return librosa.istft(mag_enhanced * phase)

    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Improved loudness normalization targeting LUFS."""
        # Compute current loudness
        meter = pyloudnorm.Meter(self.config.sample_rate)
        current_loudness = meter.integrated_loudness(audio)

        # Compute gain needed
        gain_db = self.config.target_lufs - current_loudness
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain with soft clipping
        audio_normalized = audio * gain_linear
        return np.tanh(audio_normalized)  # Soft clipping

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply all processing steps efficiently."""
        if self.config.trim_silence:
            audio = self.vad_split(audio)

        if self.config.remove_noise:
            audio = self.spectral_gating(audio)

        if self.config.enhance_speech:
            audio = self.enhance_clarity(audio)

        if self.config.normalize:
            audio = self.normalize_loudness(audio)

        return audio