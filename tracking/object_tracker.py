from tracking.abstract_tracker import AbstractTracker

import supervision as sv
import cv2
from typing import List
import numpy as np
from ultralytics.engine.results import Results

class ObjectTracker(AbstractTracker):

    def __init__(self, model_path: str, conf: float = 0.5, ball_conf: float = 0.3) -> None:
        """
        Initialize ObjectTracker with detection and tracking.

        Args:
            model_path (str): Model Path.
            conf (float): Confidence threshold for detection.
        """
        super().__init__(model_path, conf)  # Call the Tracker base class constructor

        self.ball_conf = ball_conf
        self.classes = ['ball', 'goalkeeper', 'player', 'referee']
        self.tracker = sv.ByteTrack(lost_track_buffer=5)  # Initialize ByteTracker
        self.tracker.reset()
        self.all_tracks = {class_name: {} for class_name in self.classes}  # Initialize tracks
        self.cur_frame = 0  # Frame counter initialization
        self.original_size = (1920, 1080)  # Original frame size (1920x1080)
        self.scale_x = self.original_size[0] / 1280
        self.scale_y = self.original_size[1] / 1280

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Perform object detection on multiple frames.

        Args:
            frames (List[np.ndarray]): List of frames to perform object detection on.

        Returns:
            List[Results]: Detection results for each frame.
        """
        # Preprocess: Resize frames to 1280x1280
        resized_frames = [self._preprocess_frame(frame) for frame in frames]

        # Use YOLOv8's predict method to handle batch inference
        detections = self.model.predict(resized_frames, conf=self.conf)

        return detections  # Batch of detections

    def track(self, detection: Results) -> dict:
        """
        Perform object tracking on detection.

        Args:
            detection (Results): Detected objects for a single frame.

        Returns:
            dict: Dictionary containing tracks of the frame.
        """
        # Convert Ultralytics detections to supervision
        detection_sv = sv.Detections.from_ultralytics(detection)

        # Perform ByteTracker object tracking on the detections
        tracks = self.tracker.update_with_detections(detection_sv)

        self.current_frame_tracks = self._tracks_mapper(tracks, self.classes)
        
        # Store the current frame's tracking information in all_tracks
        self.all_tracks[self.cur_frame] = self.current_frame_tracks.copy()

        # Increment the current frame counter
        self.cur_frame += 1

        # Return only the last frame's data
        return self.current_frame_tracks
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame by resizing it to 1280x1280.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            np.ndarray: The resized frame.
        """
        # Resize the frame to 1280x1280
        resized_frame = cv2.resize(frame, (1280, 1280))
        return resized_frame
    
    def _tracks_mapper(self, tracks: sv.Detections, class_names: List[str]) -> dict:
        """
        Maps tracks to a dictionary by class and tracker ID. Also, adjusts bounding boxes to 1920x1080 resolution.

        Args:
            tracks (sv.Detections): Tracks from the frame.
            class_names (List[str]): List of class names.

        Returns:
            dict: Mapped detections for the frame.
        """
        # Initialize the dictionary
        result = {class_name: {} for class_name in class_names}

        # Extract relevant data from tracks
        xyxy = tracks.xyxy  # Bounding boxes
        class_ids = tracks.class_id  # Class IDs
        tracker_ids = tracks.tracker_id  # Tracker IDs
        confs = tracks.confidence

        # Iterate over all tracks
        for bbox, class_id, track_id, conf in zip(xyxy, class_ids, tracker_ids, confs):
            class_name = class_names[class_id]

            # Skip balls with confidence lower than ball_conf
            if class_name == "ball" and conf < self.ball_conf:
                continue  # Skip low-confidence ball detections

            # Create class_name entry if not already present
            if class_name not in result:
                result[class_name] = {}

            # Scale the bounding box back to the original resolution (1920x1080)
            scaled_bbox = [
                bbox[0] * self.scale_x,  # x1
                bbox[1] * self.scale_y,  # y1
                bbox[2] * self.scale_x,  # x2
                bbox[3] * self.scale_y   # y2
            ]

            # Add track_id entry if not already present
            if track_id not in result[class_name]:
                result[class_name][track_id] = {'bbox': scaled_bbox}

        return result
