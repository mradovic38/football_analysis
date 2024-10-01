from abc import ABC, abstractmethod
from ultralytics import YOLO
import torch
from typing import Any, Dict, List
from ultralytics.engine.results import Results
import numpy as np

class AbstractTracker(ABC):

    def __init__(self, model_path: str, conf: float = 0.1) -> None:
        """
        Load the model from the given path and set the confidence threshold.

        Args:
            model_path (str): Path to the model.
            conf (float): Confidence threshold for detections.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf  # Set confidence threshold
        self.cur_frame = 0  # Initialize current frame counter

    @abstractmethod
    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Abstract method for YOLO detection.

        Args:
            frames (List[np.ndarray]): List of frames for detection.

        Returns:
            List[Results]: List of YOLO detection result objects.
        """
        pass
        
    @abstractmethod
    def track(self, detection: Results) -> dict:
        """
        Abstract method for tracking detections.

        Args:
            detection (Results): YOLO detection results for a single frame.

        Returns:
            dict: Tracking data.
        """
        pass
