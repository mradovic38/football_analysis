from .abstract_annotator import AbstractAnnotator
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
import cv2

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

        # Combine the projection frame and the original frame into one
        #combined_frame = cv2.addWeighted(frame, 0.7, projection_frame, 0.3, 0)

        return projection_frame