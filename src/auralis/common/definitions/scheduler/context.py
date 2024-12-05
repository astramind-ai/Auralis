#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from dataclasses import dataclass
from typing import Optional

from auralis.common.definitions.types.generator import Tokens, SpeakerEmbeddings, DecodingEmbeddingsModifier, \
    Spectrogram


@dataclass
class GenerationContext:
    request_id: str
    start_time: float
    text: str
    language: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    tokens: Optional[Tokens] = None
    # this is a modifier which will condition the decoding process in a autoregressive decoder only model
    decoding_embeddings_modifier: Optional[DecodingEmbeddingsModifier] = None
    spectrogram: Optional[Spectrogram] = None
    speaker_embeddings: Optional[SpeakerEmbeddings] = None

    @classmethod
    def from_request(self, request, **kwargs):
        _requests = 0
        for k, v in (request.__dict__.items(), kwargs.items()):
            if k == 'request_id':
                setattr(self, k, v + '_' + str(_requests))
                _requests += 1
            if k in self.__dict__.keys():
                setattr(self, k, v)
        return self