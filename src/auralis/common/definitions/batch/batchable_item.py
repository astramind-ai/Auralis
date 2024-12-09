from abc import ABC, abstractmethod


class BatchableItem(ABC):

    @abstractmethod
    def length(self, key: str):
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def lenght(self):
    #     raise NotImplementedError
    #
