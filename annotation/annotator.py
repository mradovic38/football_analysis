from .abstract_annotator import AbstractAnnotator
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
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
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        
        # TODO: read image from path

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
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])


        return frame
