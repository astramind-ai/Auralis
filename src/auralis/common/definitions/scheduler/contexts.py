#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
import copy
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Union, List

import torch

from auralis import TTSRequest
from auralis.common.definitions.batch.batchable_item import BatchableItem
from auralis.common.definitions.types.generator import Tokens, SpeakerEmbeddings, DecodingEmbeddingsModifier, \
    Spectrogram





@dataclass
class BaseContext(BatchableItem):
    """
    Represents a context for generating text-to-speech output.

    Attributes:
        request_id (Optional[str]): Unique identifier for the request.
        start_time (Optional[float]): The start time of the request.
        text (Optional[str]): The input text for generation.
        language (Optional[str]): The language of the text.
        temperature (Optional[float]): Sampling temperature for generation.
        top_p (Optional[float]): Top-p sampling parameter for generation.
        top_k (Optional[int]): Top-k sampling parameter for generation.
        repetition_penalty (Optional[float]): Penalty for repetition during generation.
        tokens (Optional[Tokens]): Tokens generated from the input text.
        decoding_embeddings_modifier (Optional[DecodingEmbeddingsModifier]): Modifier for decoding embeddings.
        speaker_embeddings (Optional[SpeakerEmbeddings]): Embeddings for the speaker's voice.
        spectrogram (Optional[Spectrogram]): Spectrogram representation of the generated audio.
    """
    parent_request_id: str
    start_time: float

    def __post_init__(self):
        self.request_id = uuid.uuid4().hex

    @classmethod
    def from_request(cls, request: TTSRequest, **kwargs) -> 'BaseContext':
        """
        Crea un GenerationContext da un TTSRequest.
        """
        shared_fields = {}
        self_keys = cls.__dataclass_fields__.keys() # noqa
        for k, v in vars(request).items():
            if k in self_keys:
                shared_fields[k] = v

        # Add additional fields
        shared_fields.update(kwargs)

        return cls(**shared_fields)

    def update(self, **kwargs):
        """
        Updates the context with new values while maintaining dataclass integrity.

        Args:
            **kwargs: New values for fields
        """
        # Verify valid fields
        valid_fields = self.__dataclass_fields__.keys() # type: ignore
        for field_name in kwargs:
            if field_name not in valid_fields:
                raise ValueError(f"Invalid field: {field_name}")

        # Update fields
        for field_name, value in kwargs.items():
            setattr(self, field_name, value)

        # Recall __post_init__
        if hasattr(self, '__post_init__'):
            self.__post_init__()

        return self

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class ConditioningContext(BaseContext):
    """
    Represents a context for conditioning text-to-speech output (first phase).

    This context is used for the first phase of text-to-speech processing.  It
    contains the text to be processed and the speaker audio files.  The
    ConditioningContext is used to create the conditioning data required for
    the second phase of text-to-speech processing.

    Attributes:
        tokens (Tokens): Tokens generated from the input text.
        speaker_files (Union[Union[str, List[str]], Union[bytes, List[bytes]]]): Speaker audio files or data.
        max_ref_length (Optional[int]): Maximum reference length for voice conditioning.
        gpt_cond_len (Optional[int]): Length of GPT conditioning.
        gpt_cond_chunk_len (Optional[int]): Chunk length for GPT conditioning.
    """
    tokens: Tokens
    speaker_files: Union[
        Union[str, List[str]], Union[bytes, List[bytes]], Union[torch.Tensor, List[torch.Tensor]]]
    # Shared voice parameters
    max_ref_length: Optional[int] = 60
    gpt_cond_len: Optional[int] = 30
    gpt_cond_chunk_len: Optional[int] = 4

    @property
    def length(self):
        return sum(t.shape[-1] for t in self.tokens)



@dataclass
class PhoneticContext(BaseContext):

    tokens: Tokens
    decoding_embeddings_modifier: Optional[DecodingEmbeddingsModifier] = None
    speaker_embeddings: Optional[SpeakerEmbeddings] = None

    # Generation parameters for the gpt model
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 5.0
    length_penalty: Optional[float] = 1.0
    do_sample: Optional[bool] = True
    stream: Optional[bool] = False

    @property
    def length(self):
        return len(self.tokens)


@dataclass
class SpeechContext(BaseContext):
    spectrogram: Spectrogram
    tokens: Tokens
    speaker_embeddings: Optional[SpeakerEmbeddings] = None

    @property
    def length(self):
        return self.spectrogram.shape[0] * self.spectrogram.shape[1]