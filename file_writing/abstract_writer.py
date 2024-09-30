from abc import ABC, abstractmethod
from typing import Any

class AbstractWriter(ABC):
    """An abstract base class for writing data to a file."""

    @abstractmethod
    def write(self, filename: str, data: Any) -> None:
        """Save data to a file.

        Args:
            filename (str): The name of the file to save the data.
            data (Any): The data to be saved.
        """
        pass
        
    @abstractmethod
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to a serializable format.

        Args:
            obj (Any): The object to convert.

        Returns:
            Any: A serializable representation of the object.
        """
        pass