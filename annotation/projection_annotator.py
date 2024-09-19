from .abstract_annotator import AbstractAnnotator

import cv2
import numpy as np

class ProjectionAnnotator(AbstractAnnotator):

    def _is_color_dark(self, color):
        """Check if the color is dark or light using luminance."""
        luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])  # Luminance formula
        return luminance < 128

    def _draw_outline(self, frame, pos, shape='circle', size=5, is_dark=True):
        """Draw a white or black outline around the player, goalkeeper, or referee."""
        outline_color = (255, 255, 255) if is_dark else (0, 0, 0)
        
        if shape == 'circle':
            # Draw a larger circle outline
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=size + 2, color=outline_color, thickness=2)
        elif shape == 'square':
            # Draw a larger square outline
            top_left = (int(pos[0]) - (size + 2), int(pos[1]) - (size + 2))
            bottom_right = (int(pos[0]) + (size + 2), int(pos[1]) + (size + 2))
            cv2.rectangle(frame, top_left, bottom_right, color=outline_color, thickness=2)
        elif shape == 'dotted_circle':
            # Draw a dotted circle outline (for referee)
            for i in range(0, 360, 15):  # Dots every 15 degrees
                angle_rad = np.radians(i)
                dot_x = int(pos[0]) + int((size + 2) * np.cos(angle_rad))
                dot_y = int(pos[1]) + int((size + 2) * np.sin(angle_rad))
                cv2.circle(frame, (dot_x, dot_y), radius=1, color=(0, 0, 0), thickness=-1)

    def annotate(self, frame, tracks):
        # Iterate through the tracks and annotate the players, goalkeepers, referees, and ball
        for class_name, track_data in tracks.items():
            for track_id, track_info in track_data.items():
                # Get the projected position on the field
                proj_pos = track_info['projection']  # (x, y) tuple
                color = track_info.get('club_color', (255, 255, 255))  # Default color if not provided
                is_dark_color = self._is_color_dark(color)

                if class_name == 'player' or class_name == 'goalkeeper':
                    # Determine if the player or goalkeeper has possession of the ball
                    if track_info.get('has_ball', False):
                        # Draw a larger, bolder circle for the player/goalkeeper with ball possession
                        self._draw_outline(frame, proj_pos, shape='circle', size=7, is_dark=is_dark_color)
                        cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=7, color=color, thickness=-1)
                    else:
                        # Draw a normal player/goalkeeper circle or square
                        shape = 'square' if class_name == 'goalkeeper' else 'circle'
                        self._draw_outline(frame, proj_pos, shape=shape, size=5, is_dark=is_dark_color)
                        if shape == 'circle':
                            cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=5, color=color, thickness=-1)
                        else:
                            top_left = (int(proj_pos[0]) - 5, int(proj_pos[1]) - 5)
                            bottom_right = (int(proj_pos[0]) + 5, int(proj_pos[1]) + 5)
                            cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=-1)
                
                elif class_name == 'referee':
                    # Draw a circle with a black dotted outline for the referee
                    self._draw_outline(frame, proj_pos, shape='dotted_circle', size=5, is_dark=is_dark_color)
                    cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=5, color=(0, 0, 0), thickness=1)

                elif class_name == 'ball':
                    # Draw a yellow plus sign for the ball
                    color = (0, 255, 255)  # Yellow
                    self._draw_outline(frame, proj_pos, shape='x', size=5, is_dark=False)
                    cv2.line(frame, (int(proj_pos[0]) - 5, int(proj_pos[1])), 
                             (int(proj_pos[0]) + 5, int(proj_pos[1])), color=color, thickness=2)
                    cv2.line(frame, (int(proj_pos[0]), int(proj_pos[1]) - 5), 
                             (int(proj_pos[0]), int(proj_pos[1]) + 5), color=color, thickness=2)

        return frame
    