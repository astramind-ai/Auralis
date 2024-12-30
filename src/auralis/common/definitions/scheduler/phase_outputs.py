from typing import List, Union, AsyncGenerator, Sequence, Literal, Any

import torch
from pydantic import BaseModel
from pydantic.main import IncEx
from vllm import RequestOutput

from auralis.common.definitions.dto.requests import TTSRequest


class BasePhaseOutput(BaseModel, arbitrary_types_allowed=True):
    """
    The output of the base phase, when new model are added this class must be updated
    """
    request: TTSRequest

    def model_dump(
        self,
        *args,
        **kwargs
    ):
        request = self.request
        model_dict =  super().model_dump(*args, **kwargs)
        model_dict['request'] = request
        return model_dict

class FirstPhaseOutput(BasePhaseOutput):
    """
    The output of the first phase, when new model are added this class must be updated
    """
    generator: AsyncGenerator[RequestOutput, None]
    speaker_embedding: torch.Tensor
    multimodal_data: torch.Tensor

class SecondPhaseOutput(BasePhaseOutput):
    """
    The output of the second phase, when new model are added this class must be updated
    """
    tokens: List[Union[int, torch.Tensor, Sequence[int]]]
    hidden_states: torch.Tensor
    speaker_embeddings: torch.Tensor

# We omit the third phase since we don't use it, the third phase is the TTSOutput generator