from abc import ABC, abstractmethod
from ultralytics import YOLO
import supervision as sv

class AbstractAnnotator(ABC):

    @abstractmethod
    def annotate(self, frame):
        """
        Abstract method for annotation
        
        Args:
            frame (array): The current frame for detection.
        
        Returns:
            Annotated frame
        """
        pass
        