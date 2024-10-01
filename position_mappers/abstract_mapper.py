from abc import ABC, abstractmethod

class AbstractMapper(ABC):
    """An abstract base class for mapping detections."""

    @abstractmethod
    def map(self, detection: dict) -> dict:
        """Maps detection data to a different representation.

        Args:
            detection (dict): The detection data containing keypoints and object information.

        Returns:
            dict: The mapped detection data.
        """
        pass