from abc import ABC, abstractmethod
from ultralytics import YOLO

class Tracker(ABC):

    def __init__(self, model_path):
        """
        Load the model from the given path.
        """
        self.model = YOLO(model_path)

    @abstractmethod
    def detect(self, frame):
        """
        Apply the model on the provided frame. Returns the detections array.
        """
        pass
        
    @abstractmethod
    def track(self, frame, detections):
        """
        Apply the tracking alghoritm 
        """
        pass