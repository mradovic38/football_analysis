from .abstract_annotator import AbstractAnnotator
from utils import get_bbox_width, get_bbox_center, is_color_dark, rgb_bgr_converter

import cv2
import numpy as np
from typing import Dict, Tuple

class ObjectAnnotator(AbstractAnnotator):
    """Annotates objects in a frame, such as the ball, players, referees, and goalkeepers."""

    def __init__(self, ball_annotation_color: Tuple[int, int, int] = (48, 48, 190), 
                 referee_annotation_color: Tuple[int, int, int] = (40, 40, 40)) -> None:
        """
        Initializes the ObjectAnnotator with predefined ball and referee annotation colors.

        Args:
            ball_annotation_color (Tuple[int, int, int]): RGB color to annotate the ball with.
            referee_annotation_color (Tuple[int, int, int]): RGB color to annotate the referees with.
        """
        self.ball_annotation_color = ball_annotation_color
        self.referee_annotation_color = referee_annotation_color
        super().__init__()
        
    
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates the frame with objects like players, referees, and the ball.
        
        Args:
            frame (np.ndarray): The current frame to be annotated.
            tracks (Dict): A dictionary containing object tracking data, categorized by object types.

        Returns:
            np.ndarray: The annotated frame.
        """

        frame = frame.copy()

        # Iterate over the tracked objects
        for track in tracks:
            for track_id, item in tracks[track].items():
                # Get the club color, or default to yellow
                color = item.get('club_color', (255, 255, 0))
                # Convert RGB to BGR format for OpenCV
                color = rgb_bgr_converter(color)

                # Annotate based on object type
                if track == 'ball':
                    frame = self.draw_triangle(frame, item['bbox'], self.ball_annotation_color)
                elif track == 'referee':
                    frame = self.draw_ellipse(frame, item['bbox'], self.referee_annotation_color, track_id, -1, track)
                else:
                    speed = item.get('speed', 0)
                    frame = self.draw_ellipse(frame, item['bbox'], color, track_id, speed, track)

                    # If the player has the ball, draw a triangle to indicate it
                    if 'has_ball' in item and item['has_ball']:
                        frame = self.draw_triangle(frame, item['bbox'], color)

        return frame
    

    def draw_triangle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draws a triangle to indicate the ball's position.

        Args:
            frame (np.ndarray): The frame where the triangle will be drawn.
            bbox (Tuple[int, int, int, int]): The bounding box of the object.
            color (Tuple[int, int, int]): The color of the triangle.

        Returns:
            np.ndarray: The frame with the triangle drawn on it.
        """

        # Adjust the color for the triangle outline based on the ball color's darkness
        color2 = (255, 255, 255) if is_color_dark(color) else (0, 0, 0)

        # Get the x and y position of the triangle
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)
        x = int(x)

        # Define the triangle points
        points = np.array([
            [x, y],
            [x - 8, y - 18],
            [x + 8, y - 18]
        ])

        # Draw the filled triangle
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        # Draw the outline of the triangle
        cv2.drawContours(frame, [points], 0, color2, 1)

        return frame
    
    def _draw_double_ellipse(self, frame: np.ndarray, x: int, y: int, w: int, color: Tuple[int, int, int]) -> None:
        """
        Draws two concentric ellipses for special objects like goalkeepers.

        Args:
            frame (np.ndarray): The frame where the ellipses will be drawn.
            x (int): X-coordinate of the ellipse center.
            y (int): Y-coordinate of the ellipse center.
            w (int): The width (major axis) of the ellipse.
            color (Tuple[int, int, int]): The color of the ellipse.
        """
        size_decrement = 5  # Reduce the size of the second ellipse
        # Draw two concentric ellipses
        for i in range(2):
            cv2.ellipse(frame, center=(x, y), axes=(w - i * size_decrement, 20 - i * size_decrement),
                        angle=0, startAngle=-30, endAngle=240, color=color, thickness=2, lineType=cv2.LINE_AA)
            
    
    def _draw_dashed_ellipse(self, frame: np.ndarray, x: int, y: int, w: int, color: Tuple[int, int, int]) -> None:
        """
        Draws a dashed ellipse, used for annotating referees.

        Args:
            frame (np.ndarray): The frame where the ellipse will be drawn.
            x (int): X-coordinate of the ellipse center.
            y (int): Y-coordinate of the ellipse center.
            w (int): The width (major axis) of the ellipse.
            color (Tuple[int, int, int]): The color of the ellipse.
        """
        dash_length = 15  # Length of each dash
        total_angle = 270  # Total angle to cover

        # Draw dashed lines by alternating between dashes and gaps
        for angle in range(-30, total_angle, dash_length * 2):
            cv2.ellipse(frame, center=(x, y), axes=(w, 20), angle=0,
                        startAngle=angle, endAngle=angle + dash_length, color=color, thickness=2, lineType=cv2.LINE_AA)

       

    def draw_ellipse(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], 
                     track_id: int, speed: float, obj_cls: str = 'player') -> np.ndarray:
        """
        Draws an ellipse around an object and annotates it with its ID and speed.

        Args:
            frame (np.ndarray): The frame where the ellipse will be drawn.
            bbox (Tuple[int, int, int, int]): The bounding box of the object.
            color (Tuple[int, int, int]): The color of the ellipse.
            track_id (int): The unique identifier of the object.
            speed (float): The speed of the object (in km/h).
            obj_cls (str): The object class, either 'player', 'goalkeeper', or 'referee'.

        Returns:
            np.ndarray: The frame with the ellipse and annotations drawn.
        """
        # Adjust the color for the text and ID based on the darkness of the primary color
        color2 = (255, 255, 255) if is_color_dark(color) else (0, 0, 0)

        # Get the position and size for the ellipse
        y = int(bbox[3])
        x, _ = get_bbox_center(bbox)
        x = int(x)
        w = int(get_bbox_width(bbox))

        # Determine the ellipse style based on the object class
        if obj_cls == 'referee':
            self._draw_dashed_ellipse(frame, x, y, w, color)
        elif obj_cls == 'goalkeeper':
            self._draw_double_ellipse(frame, x, y, w, color)
        else:
            # Standard ellipse for players
            cv2.ellipse(frame, center=(x, y), axes=(w, 20), angle=0, startAngle=-30, endAngle=240, color=color,
                        thickness=2, lineType=cv2.LINE_AA)

        # Draw a small rectangle under the ellipse to hold the object's ID
        y = int(bbox[3]) + 10
        h, w = 10, 20
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, cv2.FILLED)

        # Display the track ID
        x1 = x - len(str(track_id)) * 5
        cv2.putText(frame, text=f"{track_id}", org=(x1, y + h // 2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=color2, thickness=2)

        # If the object's speed is available, annotate it as well
        if speed >= 0:
            speed_str = f"{speed:.2f} km/h"
            x2 = x - len(speed_str) * 5
            cv2.putText(frame, text=speed_str, org=(x2, y + 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                        color=color2, thickness=2)

        return frame
