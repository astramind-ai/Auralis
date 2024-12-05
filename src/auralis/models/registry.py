#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.

from enum import Enum
from typing import Dict, Optional


class SupportedModelTypes(Enum):
    XTTSv2 = "xtts"


class ModelRegistry:
    _instance = None
    _models: Dict[SupportedModelTypes, dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_model(cls, model_type: SupportedModelTypes, config_converter=None, **model_info):
        def decorator(model_class):
            cls._models[model_type] = {
                'class': model_class,
                'config_converter': config_converter,
                **model_info
            }
            return model_class

        return decorator

    @classmethod
    def get_model_info(cls, model_type: SupportedModelTypes) -> Optional[dict]:
        return cls._models.get(model_type)

    @classmethod
    def get_model_class(self, model_type: SupportedModelTypes):
        return self._models[model_type]['class']

# Decorator to register a model
def register_tts_model(model_type: SupportedModelTypes, **kwargs):
    return ModelRegistry.register_model(model_type, **kwargs)