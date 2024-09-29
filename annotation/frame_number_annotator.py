from .abstract_annotator import AbstractAnnotator

import cv2

class FrameNumberAnnotator(AbstractAnnotator):

    def annotate(self, frame, tracks):
        frame = frame.copy()

        frame_num = tracks['frame_num']

        cv2.putText(frame, f'{frame_num}', (frame.shape[1] - 100, 40), 
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return frame



