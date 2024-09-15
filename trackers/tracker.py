from abc import ABC, abstractmethod
from ultralytics import YOLO

class Tracker(ABC):

    def __init__(self, model_path, conf=0.2):
        """
        Load the model from the given path. 
        Sets the confidece level of the model to the provided conf argument.
        """
        self.model = YOLO(model_path)
        self.conf = conf

    @abstractmethod
    def detect(self, frame):
        """
        Apply the model on the provided frame. Returns the detections array.
        """
        pass
        
    @abstractmethod
    def track(self, frame, detection):
        """
        Apply the tracking alghoritm 
        """
        pass