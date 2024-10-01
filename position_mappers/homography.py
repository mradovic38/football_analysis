import cv2
import numpy as np
from typing import Tuple, List

def get_homography(keypoints: dict, top_down_keypoints: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix between detected keypoints and top-down keypoints.

    Args:
        keypoints (dict): A dictionary of detected keypoints, where keys are identifiers 
        and values are (x, y) coordinates.
        top_down_keypoints (np.ndarray): An array of shape (n, 2) containing the top-down keypoints.

    Returns:
        np.ndarray: A 3x3 homography matrix that maps the keypoints to the top-down view.
    """
    kps: List[Tuple[float, float]] = []
    proj_kps: List[Tuple[float, float]] = []

    for key in keypoints.keys():
        kps.append(keypoints[key])
        proj_kps.append(top_down_keypoints[key])

    def _compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Compute a single homography matrix between source and destination points.

        Args:
            src_points (array): Source points coordinates of shape (n, 2).
            dst_points (array): Destination points coordinates of shape (n, 2).

        Returns:
            np.ndarray: The computed homography matrix of shape (3, 3).
        """
        # Compute the homography matrix using RANSAC
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        h, _ = cv2.findHomography(src_points, dst_points)

        return h.astype(np.float32)

    H = _compute_homography(np.array(kps), np.array(proj_kps))

    return H


def apply_homography(pos: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Apply a homography transformation to a 2D point.

    Args:
        pos (Tuple[float, float]): The (x, y) coordinates of the point to be projected.
        H (np.ndarray): The homography matrix of shape (3, 3).

    Returns:
        Tuple[float, float]: The projected (x, y) coordinates in the destination space.
    """
    x, y = pos
    pos_homogeneous = np.array([x, y, 1])
    projected_pos = np.dot(H, pos_homogeneous)
    projected_pos /= projected_pos[2]  # Normalize homogeneous coordinates

    return projected_pos[0], projected_pos[1]
