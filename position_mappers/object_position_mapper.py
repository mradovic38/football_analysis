from .abstract_mapper import AbstractMapper
from .homography import get_homography, apply_homography
from utils.bbox_utils import get_feet_pos

class ObjectPositionMapper(AbstractMapper):

    def __init__(self, top_down_keypoints) -> None:
        super().__init__()
        self.top_down_keypoints = top_down_keypoints

    def map(self, detection):
        detection = detection.copy()
        
        keypoints = detection['keypoints']
        object = detection['object']

        H = get_homography(keypoints, self.top_down_keypoints)


        for _, object_data in object.items():
            for _, track_info in object_data.items():
                bbox = track_info['bbox']
                feet_pos = get_feet_pos(bbox)  # Get the foot position
                projected_pos = apply_homography(feet_pos, H)  # Apply the general homography function
                track_info['projection'] = projected_pos  # Add the projection to the track info

        
        return detection

        