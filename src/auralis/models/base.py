from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, List, Union, Tuple, Optional

import torch
import torchaudio
from dataclasses import dataclass

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
from vllm import RequestOutput

from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest

Token = Union[int, List[int]]

AudioTokenGenerator = AsyncGenerator[RequestOutput, None]
AudioOutputGenerator = AsyncGenerator[TTSOutput, None]

SpeakerEmbeddings = torch.Tensor
GPTLikeDecoderConditioning = torch.Tensor
RequestsIds = List

TokenGeneratorsAndPossiblyConditioning = Union[
    Tuple[
        List[AudioTokenGenerator],
        RequestsIds,
        SpeakerEmbeddings,
        Union[List[GPTLikeDecoderConditioning], GPTLikeDecoderConditioning]
    ],
    Tuple[
        List[AudioTokenGenerator],
        RequestsIds,
        SpeakerEmbeddings
    ],
    Tuple[
        List[AudioTokenGenerator],
        RequestsIds,
        GPTLikeDecoderConditioning
    ],
    List[AudioTokenGenerator],
    RequestsIds
    ]

@dataclass
class ConditioningConfig:
    """Conditioning configuration for the model.
    
    Attributes:
        speaker_embeddings (bool): Whether the model uses speaker embeddings for voice cloning.
        gpt_like_decoder_conditioning (bool): Whether the model uses GPT-like decoder conditioning.
    """
    speaker_embeddings: bool = False
    gpt_like_decoder_conditioning: bool = False


class BaseAsyncTTSEngine(ABC, torch.nn.Module):
    """Base interface for asynchronous text-to-speech engines.
    
    This abstract class defines the interface for TTS engines that follow a two-phase generation process:
    1. Token generation: Converting text to intermediate tokens
    2. Audio generation: Converting tokens to speech waveforms
    
    The class supports both speaker conditioning and GPT-like decoder conditioning for enhanced control
    over the generated speech. It inherits from torch.nn.Module for neural network functionality.
    """

    @abstractmethod
    async def get_generation_context(
            self,
            request: TTSRequest,
    ) -> TokenGeneratorsAndPossiblyConditioning:
        """Get token generators and conditioning for audio generation.

        This method prepares the generation context by processing the input text and any
        conditioning signals (speaker embeddings, GPT conditioning) specified in the request.

        Args:
            request (TTSRequest): The TTS request containing input text and optional speaker files.

        Returns:
            TokenGeneratorsAndPossiblyConditioning: A tuple containing token generators and optional
                conditioning tensors (speaker embeddings and/or GPT conditioning).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    async def process_tokens_to_speech(
            self,
            generator: AudioTokenGenerator,
            speaker_embeddings: SpeakerEmbeddings,
            multimodal_data: GPTLikeDecoderConditioning = None,
            request: TTSRequest = None,
    ) -> AudioOutputGenerator:
        """Generate speech from tokens with optional conditioning.

        This method converts the generated tokens into speech waveforms, applying any
        specified conditioning signals to control the voice characteristics.

        Args:
            generator (AudioTokenGenerator): Token generator from the first phase.
            speaker_embeddings (SpeakerEmbeddings): Speaker embeddings for voice cloning.
            multimodal_data (GPTLikeDecoderConditioning, optional): GPT conditioning data.
            request (TTSRequest, optional): Original TTS request for reference.

        Returns:
            AudioOutputGenerator: An async generator yielding TTSOutput objects containing
                audio chunks.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def conditioning_config(self) -> ConditioningConfig:
        """Get the model's conditioning configuration.

        Returns:
            ConditioningConfig: Configuration specifying which conditioning types are supported.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def device(self):
        """Get the current device of the model.

        Returns:
            torch.device: The device (CPU/GPU) where the model parameters reside.
        """
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the current data type of the model parameters.

        Returns:
            torch.dtype: The data type of the model parameters.
        """
        return next(self.parameters()).dtype

    @abstractmethod
    def get_memory_usage_curve(self):
        """Get memory usage curve for different concurrency levels.

        This method tests VLLM memory usage at different concurrency levels to help
        optimize resource allocation.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def get_memory_percentage(memory: int) -> Optional[float]:
        """Calculate the percentage of GPU memory that would be used.

        Args:
            memory (int): The amount of memory in bytes to check.

        Returns:
            Optional[float]: The fraction of total GPU memory that would be used,
                or None if no suitable GPU is found.
        """
        for i in range(torch.cuda.device_count()):
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            used_memory = total_memory - free_memory
            estimated_mem_occupation = (memory + used_memory) / total_memory
            if estimated_mem_occupation > 0 and estimated_mem_occupation < 1:
                return estimated_mem_occupation
        return None

    @classmethod
    def from_pretrained(
            cls,
            *args,
            **kwargs
    )-> 'BaseAsyncTTSEngine':
        """Load a pretrained model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            BaseAsyncTTSEngine: An instance of the model loaded with pretrained weights.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def load_audio(audio_path: Union[str, Path], sampling_rate: int = 22050) -> torch.Tensor:
        """Load and preprocess an audio file.

        This method loads an audio file, converts it to mono if needed, resamples to the
        target sampling rate, and ensures valid amplitude range.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            sampling_rate (int, optional): Target sampling rate. Defaults to 22050.

        Returns:
            torch.Tensor: Preprocessed audio tensor with shape (1, samples).
        """
        audio, lsr = torchaudio.load(audio_path)

        # Stereo to mono if needed
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

        # Clip audio invalid values
        audio.clip_(-1, 1)
        return audio