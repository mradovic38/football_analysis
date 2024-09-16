from .abstract_annotator import AbstractAnnotator

import cv2

class KeypointsAnnotator(AbstractAnnotator):

    def annotate(self, frame, tracks):
        frame = frame.copy()

        for kp_id, (x, y) in tracks.items():
            # Draw a circle for each keypoint (dot) with a radius of 5 and color (0, 255, 0) (green)
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            # Optionally, you can annotate the keypoint ID next to the dot
            cv2.putText(frame, str(kp_id), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame
