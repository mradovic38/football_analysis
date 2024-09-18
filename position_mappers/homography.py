
import cv2
import numpy as np


def get_homography(keypoints, top_down_keypoints):
    kps = []
    proj_kps = []

    for key in keypoints.keys():
        kps.append(keypoints[key])
        proj_kps.append(top_down_keypoints[key])


    def _compute_homography(src_points, dst_points):
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
    

    H = _compute_homography(kps, proj_kps)

    
    return H


def apply_homography(pos, H):
    x, y = pos
    pos_homogeneous = np.array([x, y, 1])
    projected_pos = np.dot(H, pos_homogeneous)
    projected_pos /= projected_pos[2]  # Normalize homogeneous coordinates
    return projected_pos[0], projected_pos[1]