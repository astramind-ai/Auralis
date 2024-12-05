#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from typing import Union, Callable, Dict, Any

import fsspec
import torch
import torchaudio


def wav_to_mel_cloning(
        wav,
        mel_norms_file="../experiments/clips_mel_norms.pth",
        mel_norms=None,
        device=torch.device("cpu"),
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        power=2,
        normalized=False,
        sample_rate=22050,
        f_min=0,
        f_max=8000,
        n_mels=80,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    audio, lsr = torchaudio.load(audiopath)

    # Stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Clip audio invalid values
    audio.clip_(-1, 1)
    return audio

def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)
