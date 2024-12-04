#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from vllm import ModelRegistry

from .XTTSv2 import XTTSv2Engine
from .components.vllm_mm_gpt import XttsGPT
from ..registry import register_model

register_model("xtts", XTTSv2Engine)
ModelRegistry.register_model("XttsGPT", XttsGPT)
