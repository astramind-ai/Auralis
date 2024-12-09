#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from typing import Union, Optional, AsyncGenerator, List, Callable

import torch
import torchaudio

from auralis.common.definitions.dto.output import TTSOutput
from auralis.common.definitions.dto.requests import TTSRequest
from auralis.common.definitions.scheduler.context import GenerationContext


class BaseAsyncTTSEngine(ABC, torch.nn.Module):
    """
    Base interface for TTS engines.
    It assumes a three-phase generation process:
    1. Conditioning phase: where the audio conditioning is generated.
    2. Phonetic phase: where the audio tokens are generated.
    3. Speech phase: where the audio is generated.
    """

    ### Phases ###
    @abstractmethod
    async def preprocess_inputs(self, request: TTSRequest) -> GenerationContext:
        """
        Preprocesses a TTS request, returning a GenerationContext object.
        This method should be implemented by subclasses.

        Args:
            request: TTS request object.

        Returns:
            A GenerationContext object.
        """
        raise NotImplementedError

    @abstractmethod
    async def conditioning_phase(
            self,
            request: TTSRequest,
    ) -> List[GenerationContext]:
        """
        This phase should be where the audio conditioning is generated.
        in XTTSv2 this is composed by a text embedding and a speaker embedding, as well as the voice cloning embedding

        Args:
            request: TTS request object.

        Returns:
            A list of generation context objects.
        """
        raise NotImplementedError

    @abstractmethod
    async def phonetic_phase(
            self,
            context: GenerationContext
    ) -> GenerationContext:
        """
        This phase should be where the audio tokens are generated.
        In XTTSv2 this is the part where the GPT model generates the phonetic tokens

        Args:
            context: A generation context object.

        Returns:
            A generation context object.
        """
        raise NotImplementedError

    @abstractmethod
    async def speech_phase(
            self,
            context: GenerationContext,
    ) -> TTSOutput:
        """
        This phase should be where the audio is generated.
        In XTTSv2 this is the part where the vocoder generates the audio

        Args:
            context: A generation context object.

        Returns:
            A TTSOutput object.
        """
        raise NotImplementedError

    ### Utilities ###

    @abstractmethod
    def get_memory_usage_curve(self):
        """Get memory usage curve by manually testing for vllm memory usage at different concurrency."""
        raise NotImplementedError

    @classmethod
    def from_pretrained(
            cls,
            *args,
            **kwargs
    ) -> 'BaseAsyncTTSEngine':
        """Load a pretrained model."""
        raise NotImplementedError


    @property
    def device(self):
        """Get the current device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the current dtype of the model."""
        return next(self.parameters()).dtype

    @staticmethod
    def get_memory_percentage(memory: int) -> Optional[float]:
        """
        Estimate the memory occupation of the model on the GPU.

        The function tries to estimate the memory occupation of the model by
        checking the free memory on each GPU and summing up the memory
        required by the model. If the estimated memory occupation is between
        0 and 1, it is returned, otherwise None is returned.

        Args:
            memory (int): The memory required by the model in bytes.

        Returns:
            Optional[float]: The estimated memory occupation of the model
                between 0 and 1, or None if the estimation fails.
        """
        for i in range(torch.cuda.device_count()):
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            used_memory = total_memory - free_memory
            estimated_mem_occupation = (memory + used_memory) / total_memory
            if estimated_mem_occupation > 0 and estimated_mem_occupation < 1:
                return estimated_mem_occupation
        return None

    @staticmethod
    def load_audio(audio_path: Union[str, Path], sampling_rate: int = 22050) -> torch.Tensor:
        """
        Loads an audio file into a tensor.

        Args:
            audio_path: path to the audio file.
            sampling_rate: target sampling rate.

        Returns:
            A tensor containing the audio data.
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
        """
        Context manager to manage CUDA memory.

        This context manager ensures that the CUDA memory is deallocated
        after the code block is finished. It also ensures that the memory
        is deallocated even if an exception is raised.

        Notes:
            - This context manager is designed to be used with the `async with`
              statement. It is not thread-safe and should not be used with
              the `with` statement.
            - The `torch.cuda.empty_cache()` call is necessary to deallocate
              the memory, but it can take some time. To avoid blocking the
              event loop, we use `asyncio.sleep(0.1)` to give other tasks a
              chance to run.
        """
        try:
            yield
        finally:
            torch.cuda.synchronize()
            await asyncio.sleep(0.1)
            torch.cuda.empty_cache()
