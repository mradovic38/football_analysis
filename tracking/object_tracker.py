import supervision as sv
from tracking.abstract_tracker import AbstractTracker

class ObjectTracker(AbstractTracker):

    def __init__(self, model_path, cls_tracks, cls_sv, conf=0.1):
        """
        Initialize ObjectTracker with class-specific tracking and detection.

        Args:
            model_path (str): Path to the YOLO model for object detection.
            cls_tracks (list): List of classes for tracking.
            cls_sv (list): List of classes for detection and annotation.
            conf (float): Confidence threshold for detection.
        """
        super().__init__(model_path, conf)  # Call the Tracker base class constructor
        self.cls_tracks = cls_tracks  # Classes to use tracker on
        self.cls_sv = cls_sv  # Classes to not use tracker on
        self.all_tracks = {class_name: {} for class_name in self.cls_sv + self.cls_tracks}  # Initialize tracks

    def detect(self, frame):
        '''
        Perform object detection on a single frame.

        Args:
            frame (array): Current frame

        Returns:
            array: Detection results.
        '''
        return self.model.predict(frame, self.conf)
        
    def track(self, detection):
        '''
        Perform object tracking on detection.

        Args:
            detection (array): Current frame detection

        Returns:
            dict: Dictionary containing tracks for the last frame.
        '''
        # Convert Ultralytics detections to supervision
        detection_sv = sv.Detections.from_ultralytics(detection)

        # Perform ByteTracker object tracking on the detections
        detection_tracks = self.tracker.update_with_detections(detection_sv)

        # Clear current frame tracking information
        self.current_frame_tracks = {class_name: {} for class_name in self.cls_sv + self.cls_tracks}

        for frame_detection in detection_tracks:
            bbox = frame_detection[0].tolist()
            class_id = frame_detection[3]
            track_id = frame_detection[4]

            if detection.names[class_id] in self.cls_tracks:
                if track_id not in self.current_frame_tracks[detection.names[class_id]]:
                    self.current_frame_tracks[detection.names[class_id]][track_id] = {'bbox': bbox}

        # Update tracks with tracked objects
        for frame_detection in detection_sv:
            bbox = frame_detection[0].tolist()
            class_id = frame_detection[3]
            
            if detection.names[class_id] in self.cls_sv:
                if track_id not in self.current_frame_tracks[detection.names[class_id]]:
                    self.current_frame_tracks[detection.names[class_id]][track_id] = {'bbox': bbox}
        
        # Store the current frame's tracking information in all_tracks
        self.all_tracks[self.cur_frame] = self.current_frame_tracks.copy()

        # Increment the current frame counter
        self.cur_frame += 1

        # Return only the last frame's data
        return self.current_frame_tracks