#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from typing import List, Union

import torch

Token = Union[int, torch.Tensor]
Tokens = Union[Token, List[Token]]

SpeakerEmbeddings = torch.Tensor
DecodingEmbeddingsModifier = torch.Tensor
Spectrogram = Union[torch.Tensor, List[torch.Tensor]]

