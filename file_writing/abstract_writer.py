from abc import ABC, abstractmethod

class AbstractWriter(ABC):
    @abstractmethod
    def write(self, filename, data):
        """Save data to a file."""

        pass
        
    @abstractmethod
    def _make_serializable(self, obj):
        """Convert objects to a JSON-serializable format."""
        pass