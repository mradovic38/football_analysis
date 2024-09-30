from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class AbstractAnnotator(ABC):

    @abstractmethod
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Abstract method for annotation
        
        Args:
            frame (np.ndarray): The current frame for detection.
            tracks (dict): Tracks data
        
        Returns:
            np.ndarray: Annotated frame
        """
        pass
        