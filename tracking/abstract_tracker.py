from config import ROBOFLOW_API_KEY

from abc import ABC, abstractmethod
from ultralytics import YOLO
import supervision as sv
from inference import get_model


class AbstractTracker(ABC):

    def __init__(self, model_id, conf=0.1):
        """
        Load the model from the given path and set the confidence threshold.
        
        Args:
            model_path (str): Path to the detection model.
            conf (float): Confidence threshold for detections.
        """
        self.model = get_model(model_id=model_id, api_key=ROBOFLOW_API_KEY)  # Load the YOLO model
        self.conf = conf  # Set confidence threshold
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