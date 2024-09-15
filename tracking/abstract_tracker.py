from abc import ABC, abstractmethod
from ultralytics import YOLO
import supervision as sv

class AbstractTracker(ABC):

    def __init__(self, model_path, conf=0.1):
        """
        Load the model from the given path and set the confidence threshold.
        
        Args:
            model_path (str): Path to the detection model.
            conf (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_path)  # Load the YOLO model
        self.conf = conf  # Set confidence threshold
        self.tracker = sv.ByteTrack()  # Initialize ByteTracker
        self.cur_frame = 0  # Initialize current frame counter

    @abstractmethod
    def detect(self, frame):
        """
        Abstract method for detecting objects or keypoints.
        
        Args:
            frame (array): The current frame for detection.
        
        Returns:
            dict or list: Detections or keypoints.
        """
        pass
        
    @abstractmethod
    def track(self, frame, detection):
        """
        Abstract method for tracking detected objects or keypoints.
        
        Args:
            detection (dict): The detected objects or keypoints.
        
        Returns:
            dict or list: Tracking data.
        """
        pass