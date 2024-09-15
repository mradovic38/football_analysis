from abstract_annotator import AbstractAnnotator

class Annotator(AbstractAnnotator):

    def __init__(self, obj_tracker, kp_tracker):
        self.obj_tracker = obj_tracker
        self.kp_tracker = kp_tracker

    def __call__(self, frame):
        obj_detections = self.object_tracker.detect(frame)
        kp_detections = self.kp_tracker.detect(frame)

        obj_tracks = self.object_tracker.track(obj_detections)
        kp_tracks = self.kp_tracker.track(kp_detections)
