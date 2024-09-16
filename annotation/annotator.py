from .abstract_annotator import AbstractAnnotator
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
import cv2

class Annotator(AbstractAnnotator):

    def __init__(self, obj_tracker, kp_tracker, club_assigner, ball_to_player_assigner):
        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner

    def __call__(self, frame):
        obj_detections = self.obj_tracker.detect(frame)
        kp_detections = self.kp_tracker.detect(frame)

        obj_tracks = self.obj_tracker.track(obj_detections)
        kp_tracks = self.kp_tracker.track(kp_detections)

        print(obj_tracks)

        self.club_assigner.assign_clubs(frame, obj_tracks)
        
        self.ball_to_player_assigner.assign(obj_tracks)

        all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

        return self.annotate(frame, all_tracks)

    
    def annotate(self, frame, tracks):
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])


        return frame
