from .abstract_annotator import AbstractAnnotator
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper

import cv2
import numpy as np

class Annotator(AbstractAnnotator):

    def __init__(self, obj_tracker, kp_tracker, club_assigner, ball_to_player_assigner, top_down_keypoints, field_img_path):
        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.field_image = cv2.imread(field_img_path)

    def __call__(self, frame):
        obj_detections = self.obj_tracker.detect(frame)
        kp_detections = self.kp_tracker.detect(frame)

        obj_tracks = self.obj_tracker.track(obj_detections)
        kp_tracks = self.kp_tracker.track(kp_detections)

        self.club_assigner.assign_clubs(frame, obj_tracks)
        
        self.ball_to_player_assigner.assign(obj_tracks)

        all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

        all_tracks = self.obj_mapper.map(all_tracks)

        print(all_tracks)

        return self.annotate(frame, all_tracks)

    
    def annotate(self, frame, tracks):
        # Annotate the main frame with object and keypoint annotations
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])

        # Project the player and object positions on the football field image
        projection_frame = self.projection_annotator.annotate(self.field_image.copy(), tracks['object'])

        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the frames (keep original size of the main frame)
        scale_proj = .6  # Scale the projection to 60% of its original size
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (keep it full-size)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 50px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame