from .abstract_mapper import AbstractMapper

import cv2
import numpy as np

class KeypointsMapper(AbstractMapper):
    def __init__(self, top_down_keypoints) -> None:
        self.top_down_keypoints = top_down_keypoints

    def map(detections):
        return detections

    def compute_homography(src_points, dst_points):
        """
        Compute a single homography matrix between source and destination points.

        Arguments:
            src_points: numpy array of shape (n, 2) - source points coordinates
            dst_points: numpy array of shape (n, 2) - destination points coordinates

        Returns:
            homography: numpy array of shape (3, 3) - the homography matrix
        """
        # Compute the homography matrix using RANSAC
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        h, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        
        return h.astype(np.float32)