from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass