from abc import ABC, abstractmethod


class AbstractMapper(ABC):
        
    @abstractmethod
    def map(self, detection):
        pass