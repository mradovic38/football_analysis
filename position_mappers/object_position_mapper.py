from .abstract_mapper import AbstractMapper
from .homography import get_homography, apply_homography, HomographySmoother
from utils.bbox_utils import get_feet_pos

import numpy as np

class ObjectPositionMapper(AbstractMapper):
    """
    A class to map object positions from detected keypoints to a top-down view.

    This class implements the mapping of detected objects to their corresponding
    positions in a top-down representation based on the homography obtained from 
    detected keypoints.
    """

    def __init__(self, top_down_keypoints: np.ndarray, alpha: float = 0.9) -> None:
        """
        Initializes the ObjectPositionMapper.

        Args:
            top_down_keypoints (np.ndarray): An array of shape (n, 2) containing the top-down keypoints.
            alpha (float): Smoothing factor for homography smoothing.
        """
        super().__init__()
        self.top_down_keypoints = top_down_keypoints
        self.homography_smoother = HomographySmoother(alpha=alpha)

    def map(self, detection: dict) -> dict:
        """Maps the detection data to their positions in the top-down view.

        This method retrieves keypoints and object information from the detection data,
        computes the homography matrix, smooths it over frames, and projects the foot positions
        of detected objects.

        Args:
            detection (dict): The detection data containing keypoints and object information.

        Returns:
            dict: The detection data with projected positions added.
        """
        detection = detection.copy()
        
        keypoints = detection['keypoints']
        object_data = detection['object']

        if not keypoints or not object_data:
            return detection

        H = get_homography(keypoints, self.top_down_keypoints)
        smoothed_H = self.homography_smoother.smooth(H)  # Apply smoothing to the homography matrix

        for _, object_info in object_data.items():
            for _, track_info in object_info.items():
                bbox = track_info['bbox']
                feet_pos = get_feet_pos(bbox)  # Get the foot position
                projected_pos = apply_homography(feet_pos, smoothed_H)  # Apply the smoothed homography function
                track_info['projection'] = projected_pos  # Add the projection to the track info

        return detection