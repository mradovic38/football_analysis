from .abstract_annotator import AbstractAnnotator

import cv2
import numpy as np
from typing import Dict

class KeypointsAnnotator(AbstractAnnotator):
    """Annotates frames with keypoints, drawing points at the keypoints' locations."""

    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates the frame with keypoints.

        Args:
            frame (np.ndarray): The current frame to be annotated.
            tracks (Dict): A dictionary containing keypoints, where the key is 
                           the keypoint ID and the value is a tuple (x, y) of coordinates.
        
        Returns:
            np.ndarray: The frame with keypoints annotated on it.
        """
         
        frame = frame.copy()

        for kp_id, (x, y) in tracks.items():
            # Draw a circle for each keypoint (dot) with a radius of 5 and color (0, 255, 0) (green)
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            # Annotate the keypoint ID next to the dot
            cv2.putText(frame, str(kp_id), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame
