import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Union, Optional

import torch
import torchaudio
from dataclasses import dataclass

from vllm import RequestOutput

from auralis.common.definitions.dto.output import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.definitions.scheduler.phase_outputs import FirstPhaseOutput, SecondPhaseOutput

AudioTokenGenerator = AsyncGenerator[RequestOutput, None]
AudioOutputGenerator = AsyncGenerator[TTSOutput, None]



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
    async def first_phase(
            self,
            request: TTSRequest,
    ) -> List[FirstPhaseOutput]:
        """Get token generators and conditioning for audio generation.

        This method prepares the generation context by processing the input text and any
        conditioning signals (speaker embeddings, GPT conditioning) specified in the request.

        Args:
            request (TTSRequest): The TTS request containing input text and optional speaker files.

        Returns:
            List[TokenGeneratorsAndPossiblyConditioning]: A list of tuples containing token generators and optional
                conditioning tensors (speaker embeddings and/or GPT conditioning).

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    async def second_phase(
            self,
            *args, **kwargs
    ) -> SecondPhaseOutput:

        """
        Asynchronously generate speech tokens from input.

        This abstract method defines the interface for converting input data, such as text or token generators,
        into audio output tokens using optional conditioning signals like speaker embeddings or multimodal data.

        Args:
            *args: Variable arguments, typically including token generators and optional conditioning tensors.
            **kwargs: Keyword arguments, which may include additional context or configuration details.

        Returns:
            AudioOutputGenerator: A generator yielding audio tokens as part of the speech synthesis process.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def third_phase(self, *args, **kwargs) -> AudioOutputGenerator:

        """
        Convert tokens to speech waveforms.

        This method is the main entry point for asynchronous speech synthesis. It takes token
        generators and optional conditioning signals as input and yields TTSOutput objects
        containing audio chunks.

        Args:
            *args: Variable arguments, which should be token generators and optional conditioning tensors.
            **kwargs: Keyword arguments, which should contain the original TTS request for reference.

        Yields:
            AsyncGenerator[TTSOutput, None]: A generator yielding TTSOutput objects containing audio chunks.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


    @property
    @abstractmethod
    def first_phase_resource_limit(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def second_phase_resource_limit(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def third_phase_resource_limit(self) -> int:
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

    @asynccontextmanager
    async def cuda_memory_manager(self):
        """Context manager for CUDA memory management.

        Ensures proper allocation and deallocation of CUDA memory during processing.
        """
        try:
            yield
        finally:
            torch.cuda.synchronize()
            await asyncio.sleep(0.1)
            torch.cuda.empty_cache()