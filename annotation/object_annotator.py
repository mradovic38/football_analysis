from .abstract_annotator import AbstractAnnotator
from utils import get_bbox_width, get_bbox_center, get_feet_pos, is_color_dark

import cv2
import numpy as np

class ObjectAnnotator(AbstractAnnotator):

    def __init__(self) -> None:
        super().__init__()
        self.ball_annotation_color = (48, 48, 190)
        self.referee_annotation_color = (40, 40, 40)
        
    
    def annotate(self, frame, tracks):

        frame = frame.copy()

        for track in tracks:
            for track_id, item in tracks[track].items():
                color = item.get('club_color', (255, 255, 0))

                color = (color[2], color[1], color[0]) # Convert from RGB to BGR

                if track == 'ball':
                    frame = self.draw_triangle(frame, item['bbox'], self.ball_annotation_color)

                elif track == 'referee':
                    frame = self.draw_ellipse(frame, item['bbox'], self.referee_annotation_color, track_id, -1, track)

                else:
                    speed = item.get('speed', 0)
                    frame = self.draw_ellipse(frame, item['bbox'], color, track_id, speed, track)

                    if 'has_ball' in item and item['has_ball']:
                        frame = self.draw_triangle(frame, item['bbox'], color)
                    

        return frame
    

    def draw_triangle(self, frame, bbox, color):
        color2 = (255, 255, 255) if is_color_dark(color) else (0, 0, 0)

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
                         color2,
                         1)
        
        return frame
    
    def _draw_double_ellipse(self, frame, x, y, w, color):
        size_decrement = 5
        
        # Draw Double Line
        for i in range(2):
            cv2.ellipse(frame,
            center=(x, y), 
            axes=(w- i * size_decrement, 20- i * size_decrement),
            angle=0,
            startAngle=-30,
            endAngle=240,
            color=color,
            thickness=2, 
            lineType=cv2.LINE_AA
        )
            
    
    def _draw_dashed_ellipse(self, frame, x, y, w, color):
        # Parameters for the dashed ellipse
        dash_length = 15  # Length of each dash
        total_angle = 270  # Total angle to cover (for this example, from -30 to 240 degrees)

        # Loop through the ellipse in steps of dash_length * 2 to alternate dashes and gaps
        for angle in range(-30, total_angle, dash_length * 2):
            # Draw the dash segment
            cv2.ellipse(frame,
                        center=(x, y),
                        axes=(w, 20),
                        angle=0,
                        startAngle=angle,
                        endAngle=angle + dash_length,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA
            )

       

    def draw_ellipse(self, frame, bbox, color, track_id, speed, obj_cls='player'):
        color2 = (255, 255, 255) if is_color_dark(color) else (0, 0, 0)
        
        y = int(bbox[3])
        x, _ = get_bbox_center(bbox)
        x = int(x)
        w = int(get_bbox_width(bbox))
        

        if obj_cls == 'referee':
            self._draw_dashed_ellipse(frame, x, y, w, color)

        elif obj_cls == 'goalkeeper':
            self._draw_double_ellipse(frame, x, y, w, color)
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
                    color=color2,
                    thickness=2
                    )
        
        if speed >= 0:
            speed_str = f"{speed: .2f} km/h"
            x2 = x - len(speed_str) * 5
            cv2.putText(frame,
                        text=speed_str,
                        org=(x2, y + 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=color2,
                        thickness=2
                        )


        return frame
