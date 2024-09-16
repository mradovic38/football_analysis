from inference import get_model

import supervision as sv
from tracking.abstract_tracker import AbstractTracker

class ObjectTracker(AbstractTracker):

    def __init__(self, model_id, cls_tracks, cls_sv, conf=0.1):
        """
        Initialize ObjectTracker with class-specific tracking and detection.

        Args:
            model_path (str): Path to the YOLO model for object detection.
            cls_tracks (list): List of classes for tracking.
            cls_sv (list): List of classes for detection and annotation.
            conf (float): Confidence threshold for detection.
        """
        super().__init__(model_id, conf)  # Call the Tracker base class constructor
        
        self.cls_tracks = cls_tracks  # Classes to use tracker on
        self.cls_sv = cls_sv  # Classes to not use tracker on

        self.tracker = sv.ByteTrack()  # Initialize ByteTracker
        self.tracker.reset()
        self.all_tracks = {class_name: {} for class_name in self.cls_sv + self.cls_tracks}  # Initialize tracks
        

    def detect(self, frame):
        '''
        Perform object detection on a single frame.

        Args:
            frame (array): Current frame

        Returns:
            array: Detection results.
        '''
        return self.model.infer(frame, self.conf)[0]
        
    def track(self, detection):
        '''
        Perform object tracking on detection.

        Args:
            detection (array): Current frame detection

        Returns:
            dict: Dictionary containing tracks for the last frame.
        '''
        # Convert Ultralytics detections to supervision
        detection_sv = sv.Detections.from_inference(detection)


        # Perform ByteTracker object tracking on the detections
        detection_tracks = self.tracker.update_with_detections(detection_sv)

        self.current_frame_tracks = self._detections_mapper(detection_tracks, self.cls_sv + self.cls_tracks)
        
        # Store the current frame's tracking information in all_tracks
        self.all_tracks[self.cur_frame] = self.current_frame_tracks.copy()

        # Increment the current frame counter
        self.cur_frame += 1

        # Return only the last frame's data
        return self.current_frame_tracks
    

    def _detections_mapper(self, detections, class_names):
        # Initialize the dictionary
        result = {class_name: {} for class_name in class_names}
        
        # Extract relevant data from detections
        xyxy = detections.xyxy  # Bounding boxes
        class_ids = detections.class_id  # Class IDs
        tracker_ids = detections.tracker_id  # Tracker IDs

        # Iterate over all detections
        for bbox, class_id, track_id in zip(xyxy, class_ids, tracker_ids):
            class_name = class_names[class_id]

            # Create class_name entry if not already present
            if class_name not in result:
                result[class_name] = {}

            # Add track_id entry if not already present
            if track_id not in result[class_name]:
                result[class_name][track_id] = {'bbox': bbox.tolist()}

        return result