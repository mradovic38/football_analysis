from .abstract_annotator import AbstractAnnotator
from utils import get_bbox_width, get_bbox_center, get_feet_pos
import cv2
import numpy as np

class ObjectAnnotator(AbstractAnnotator):

    
    def annotate(self, frame, tracks):

        frame = frame.copy()

        for track in tracks:
            for track_id, item in tracks[track].items():
                color = item.get('club_color', (255, 255, 0))

                if track == 'ball':
                    frame = self.draw_triangle(frame, item['bbox'], color)
                else:
                    speed = item.get('speed', 0)
                    frame = self.draw_ellipse(frame, item['bbox'], color, track_id, speed, track=='goalkeeper')

                if 'has_ball' in item and item['has_ball']:
                    frame = self.draw_triangle(frame, item['bbox'], color)
                    

        return frame
    

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)
        x = int(x)

        points = np.array([
            [x, y],
            [x-8, y-18],
            [x+8, y-18],
        ])


        cv2.drawContours(frame,
                         [points],
                         0,
                         color,
                         cv2.FILLED)
        cv2.drawContours(frame,
                         [points],
                         0,
                         (255-color[0], 255-color[1], 255-color[2]),
                         1)
        
        return frame
    

    def draw_ellipse(self, frame, bbox, color, track_id, speed, is_keeper=False):
        
        y = int(bbox[3])
        x, _ = get_bbox_center(bbox)
        x = int(x)
        w = int(get_bbox_width(bbox))

        if is_keeper:
            
            for thickness in range(1, 3):
                cv2.ellipse(frame,
                            center=(x, y), 
                            axes=(w+5*(thickness-1), 20),
                            angle=0,
                            startAngle=-30,
                            endAngle=240,
                            color=color,
                            thickness=thickness,
                            lineType=cv2.LINE_AA)
        else:
            cv2.ellipse(frame,
                        center=(x, y), 
                        axes=(w, 20),
                        angle=0,
                        startAngle=-30,
                        endAngle=240,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA
            )

        y = int(bbox[3]) + 10

        h = 10
        w = 20
        cv2.rectangle(frame,
                      (x - w, y - h),
                      (x + w, y + h),
                      color,
                      cv2.FILLED
        )
        
        x1 = x -  len(str(track_id)) * 5
        

        cv2.putText(frame,
                    text=f"{track_id}",
                    org=(x1, y + h//2),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255-color[0], 255-color[1], 255-color[2]),
                    thickness=2
                    )
        
        
        speed_str = f"{speed: .2f} km/h"
        x2 = x - len(speed_str) * 5
        cv2.putText(frame,
                    text=speed_str,
                    org=(x2, y + 20),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255-color[0], 255-color[1], 255-color[2]),
                    thickness=2
                    )


        return frame
