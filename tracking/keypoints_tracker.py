from tracking.abstract_tracker import AbstractTracker

import supervision as sv
import cv2

class KeypointsTracker(AbstractTracker):


    def __init__(self, model_id, conf=0.1, kp_conf=0.35):
        """
        Initialize KeypointsTracker for tracking keypoints.
        
        Args:
            model_path (str): Path to the YOLO model for keypoints.
            conf (float): Confidence threshold for keypoints.
        """
        super().__init__(model_id, conf)  # Call the Tracker base class constructor
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

        frame = frame.copy()
        frame = self._adjust_contrast(frame)
        self.frame = frame
        return self.model.infer(frame, self.conf)[0]
    
    def track(self, detection):
        """
        Perform keypoint tracking based on detections.
        
        Args:
            detection (list): List of detected keypoints.
        
        Returns:
            dict: Tracking data for the last frame.
        """

        detection = sv.KeyPoints.from_inference(detection)


        # Extract xy coordinates, confidence, and the number of keypoints
        xy = detection.xy[0]  # Shape: (32, 2), assuming there are 32 keypoints
        confidence = detection.confidence[0]  # Shape: (32,), confidence values


        # Create the map of keypoints with confidence greater than the threshold
        filtered_keypoints = {
            i: (coords[0], coords[1])  # i is the key (index), (x, y) are the values
            for i, (coords, conf) in enumerate(zip(xy, confidence))
            if conf > self.kp_conf
        }

        self.tracks.append(detection)
        self.cur_frame += 1

        return filtered_keypoints
    

    def _adjust_contrast(self, frame):
        """
        Adjust the contrast of the frame using Histogram Equalization.
        
        Args:
            frame (array): The input image frame.
        
        Returns:
            array: The frame with adjusted contrast.
        """
        # Check if the frame is colored (3 channels). If so, convert to grayscale for histogram equalization.
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to the Y channel (luminance)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # Convert back to BGR format
            frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # If the frame is already grayscale, apply histogram equalization directly
            frame_equalized = cv2.equalizeHist(frame)

        return frame_equalized