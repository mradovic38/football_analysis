from .abstract_annotator import AbstractAnnotator

import cv2

class FPSAnnotator(AbstractAnnotator):

    def annotate(self, frame, tracks):
        frame = frame.copy()

        fps = tracks['fps']

        cv2.putText(frame, f'{fps:.2f}', (frame.shape[1] - 100, 40), 
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return frame



