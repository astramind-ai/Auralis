#  Copyright (c) 2024 Astramind. Licensed under Apache License, Version 2.0.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from auralis.common.definitions.dto.requests import TTSRequest


class BatchableItem(ABC):

    @abstractmethod
    def length(self, key: str):
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def lenght(self):
    #     raise NotImplementedError
    #

@dataclass
class BatchedItems:
    def __init__(self, stage = None):
        self.stage = stage
        self.items: Dict[str, BatchableItem] = {}
        self._indexes = []

    def batch(self, item: BatchableItem):
        # Here the logic splits up, we can have a TTSRequest or a GenerationContext
        if isinstance(item, TTSRequest):
            # if we have a TTSRequest, we need to add it to the batch
            self.
        self.items.append(item)

    def unbatch(self):
        raise NotImplementedError

    @property
    def length(self):
        return sum([item.length(self.stage) for item in self.items.values()])