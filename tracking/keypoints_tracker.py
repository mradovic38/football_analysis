from tracking.abstract_tracker import AbstractTracker
import supervision as sv

class KeypointsTracker(AbstractTracker):


    def __init__(self, model_path, conf=0.1, kp_conf=0.35):
        """
        Initialize KeypointsTracker for tracking keypoints.
        
        Args:
            model_path (str): Path to the YOLO model for keypoints.
            conf (float): Confidence threshold for keypoints.
        """
        super().__init__(model_path, conf)  # Call the Tracker base class constructor
        self.kp_conf = kp_conf # Keypoint Confidence Threshold
        self.tracks = []  # Initialize tracks list


    def detect(self, frame):
        """
        Perform keypoint detection on the given frame.
        
        Args:
            frame (array): The current frame for detection.
        
        Returns:
            list: Detected keypoints.
        """

        preds = list(self.model.predict([frame], self.conf))[0]

        keypoints = [(kp['x'], kp['y'], kp['confidence']) for kp in preds['keypoints'] if kp['confidence'] > self.kp_conf]
        return keypoints
        
    def track(self, detection):
        """
        Perform keypoint tracking based on detections.
        
        Args:
            detection (list): List of detected keypoints.
        
        Returns:
            dict: Tracking data for the last frame.
        """
        
        detections = self.convert_keypoints_to_detections(detection)
        detection_sv = sv.Detections.from_ultralytics(detections)
        detection_tracks = self.tracker.update_with_detections(detection_sv)

        frame_keypoints = {}
        for track in detection_tracks:
            bbox, class_id, track_id = track[:3]
            x_center, y_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            frame_keypoints[track_id] = (x_center, y_center)

        # Save for the current frame and return only the last frame data
        self.tracks.append(frame_keypoints)
        self.cur_frame += 1
        return frame_keypoints

    def _convert_keypoints_to_detections(self, keypoints_data):
        """
        Convert keypoints data to detections for tracking.
        
        Args:
            keypoints_data (list): List of keypoints.
        
        Returns:
            list: Converted detections.
        """
        detections = [(x - 4, y - 4, x + 4, y + 4, confidence) for x, y, confidence in keypoints_data]
        return detections