from abc import ABC, abstractmethod

class BasePIIModel(ABC):
    @abstractmethod
    def analyze(self, text: str) -> dict:
        pass