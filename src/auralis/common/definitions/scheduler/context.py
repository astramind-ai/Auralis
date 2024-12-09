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
class GenerationContext(BatchableItem):
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
    start_time: Optional[float] = None
    text: Optional[str] = None
    language: Optional[str] = None
    speaker_files: Union[Union[str,List[str]], Union[bytes,List[bytes]], Union[torch.Tensor, List[torch.Tensor]]] = None

    # Generation parameters shared with ttsrequest
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 5.0
    length_penalty: Optional[float] = 1.0
    do_sample: Optional[bool] = True
    stream: Optional[bool] = False

    # Shared voice parameters
    max_ref_length: Optional[int] = 60
    gpt_cond_len: Optional[int] = 30
    gpt_cond_chunk_len: Optional[int] = 4

    # Generated states
    tokens: Optional[Tokens] = None
    # this is a modifier which will condition the decoding process in a autoregressive decoder only model
    decoding_embeddings_modifier: Optional[DecodingEmbeddingsModifier] = None
    speaker_embeddings: Optional[SpeakerEmbeddings] = None
    spectrogram: Optional[Spectrogram] = None

    # original request ref
    parent_request_id: Optional[str] = None
    request: Optional[TTSRequest] = None

    def __post_init__(self):
        self.request_id = uuid.uuid4()
        if self.start_time is None:
            self.start_time = time.time()

    def length(self, key: str):
        if key == 'conditioning':
            return sum(t.shape[-1] for t in self.tokens)
        elif key == 'phonetic':
            return self.max_ref_length
        elif key == 'speech':
            return 0

    @classmethod
    def from_request(cls, request: TTSRequest, **kwargs) -> 'GenerationContext':
        """
        Crea un GenerationContext da un TTSRequest.
        """
        shared_fields = {}
        self_keys = vars(cls).keys()
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