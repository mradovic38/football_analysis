from .abstract_annotator import AbstractAnnotator
from utils import is_color_dark, rgb_bgr_converter

import cv2
import numpy as np
from scipy.spatial import Voronoi
from typing import Dict


class ProjectionAnnotator(AbstractAnnotator):
    """
    Class to annotate projections on a projection image, including Voronoi regions for players (and goalkeepers), 
    and different markers for ball, players, referees, and goalkeepers.
    """

    def _draw_outline(self, frame: np.ndarray, pos: tuple, shape: str = 'circle', size: int = 10, is_dark: bool = True) -> None:
        """
        Draws a white or black outline around the object based on its color and shape.
        
        Parameters:
            frame (np.ndarray): The image on which to draw the outline.
            pos (tuple): The (x, y) position of the object.
            shape (str): The shape of the outline ('circle', 'square', 'dashed_circle', 'plus').
            size (int): The size of the outline.
            is_dark (bool): Flag indicating whether the color is dark (determines outline color).
        """
        outline_color = (255, 255, 255) if is_dark else (0, 0, 0)

        if shape == 'circle':
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=size + 2, color=outline_color, thickness=2)
        elif shape == 'square':
            top_left = (int(pos[0]) - (size + 2), int(pos[1]) - (size + 2))
            bottom_right = (int(pos[0]) + (size + 2), int(pos[1]) + (size + 2))
            cv2.rectangle(frame, top_left, bottom_right, color=outline_color, thickness=2)
        elif shape == 'dashed_circle':
            dash_length, gap_length = 30, 30
            for i in range(0, 360, dash_length + gap_length):
                start_angle_rad, end_angle_rad = np.radians(i), np.radians(i + dash_length)
                start_x = int(pos[0]) + int((size + 2) * np.cos(start_angle_rad))
                start_y = int(pos[1]) + int((size + 2) * np.sin(start_angle_rad))
                end_x = int(pos[0]) + int((size + 2) * np.cos(end_angle_rad))
                end_y = int(pos[1]) + int((size + 2) * np.sin(end_angle_rad))
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color=(0, 0, 0), thickness=2)
        elif shape == 'plus':
            cv2.line(frame, (int(pos[0]) - size, int(pos[1])), (int(pos[0]) + size, int(pos[1])), color=outline_color, thickness=10)
            cv2.line(frame, (int(pos[0]), int(pos[1]) - size), (int(pos[0]), int(pos[1]) + size), color=outline_color, thickness=10)


    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates an image with projected player, goalkeeper, referee, and ball positions, along with Voronoi regions.
        
        Parameters:
            frame (np.ndarray): The image on which to draw the annotations.
            tracks (Dict): A dictionary containing tracking information for 'player', 'goalkeeper', 'referee', and 'ball'.

        Returns:
            np.ndarray: The annotated frame.
        """
        frame = frame.copy()
        frame = self._draw_voronoi(frame, tracks)

        for class_name, track_data in tracks.items():
            if class_name != 'ball':  # Ball is drawn later
                for track_id, track_info in track_data.items():
                    proj_pos = track_info['projection']
                    color = track_info.get('club_color', (255, 255, 255))
                    color = rgb_bgr_converter(color)
                    is_dark_color = is_color_dark(color)

                    if class_name in ['player', 'goalkeeper']:
                        shape = 'square' if class_name == 'goalkeeper' else 'circle'
                        self._draw_outline(frame, proj_pos, shape=shape, is_dark=is_dark_color)

                        if track_info.get('has_ball', False):
                            cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=15, color=(0, 255, 0), thickness=2)
                        if shape == 'circle':
                            cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=10, color=color, thickness=-1)
                        else:
                            top_left = (int(proj_pos[0]) - 10, int(proj_pos[1]) - 10)
                            bottom_right = (int(proj_pos[0]) + 10, int(proj_pos[1]) + 10)
                            cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=-1)

                    elif class_name == 'referee':
                        self._draw_outline(frame, proj_pos, shape='dashed_circle', is_dark=is_dark_color)

        if 'ball' in tracks:
            for track_id, track_info in tracks['ball'].items():
                proj_pos = track_info['projection']
                self._draw_outline(frame, proj_pos, shape='plus', is_dark=is_color_dark((0, 255, 255)))
                color = (0, 255, 255)
                cv2.line(frame, (int(proj_pos[0]) - 10, int(proj_pos[1])), (int(proj_pos[0]) + 10, int(proj_pos[1])), color=color, thickness=6)
                cv2.line(frame, (int(proj_pos[0]), int(proj_pos[1]) - 10), (int(proj_pos[0]), int(proj_pos[1]) + 10), color=color, thickness=6)

        return frame

    def _draw_voronoi(self, image: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Draws Voronoi regions for players and goalkeepers on the frame.
        
        Parameters:
            image (np.ndarray): The image on which to draw the Voronoi regions.
            tracks (Dict): A dictionary containing tracking information for 'player' and 'goalkeeper'.

        Returns:
            np.ndarray: The frame with Voronoi regions drawn.
        """
        height, width = image.shape[:2]
        overlay = image.copy()
        points, player_colors = [], []

        for class_name in ['player', 'goalkeeper']:
            track_data = tracks.get(class_name, {})
            for track_id, track_info in track_data.items():
                x, y = track_info['projection'][:2]
                points.append([x, y])
                player_colors.append(rgb_bgr_converter(track_info['club_color']))

        boundary_margin = 1000
        boundary_points = [
            [-boundary_margin, -boundary_margin], [width // 2, -boundary_margin],
            [width + boundary_margin, -boundary_margin], [-boundary_margin, height // 2],
            [width + boundary_margin, height // 2], [-boundary_margin, height + boundary_margin],
            [width // 2, height + boundary_margin], [width + boundary_margin, height + boundary_margin]
        ]
        boundary_color = (128, 128, 128)
        points.extend(boundary_points)
        player_colors.extend([boundary_color] * len(boundary_points))

        if len(points) > 2:
            points = np.array(points)
            vor = Voronoi(points)
            for region_index, region in enumerate(vor.point_region):
                if -1 not in vor.regions[region] and len(vor.regions[region]) > 0:
                    polygon = [vor.vertices[i] for i in vor.regions[region]]
                    polygon = np.array(polygon, np.int32).reshape((-1, 1, 2))
                    color = player_colors[region_index] if region_index < len(player_colors) else boundary_color
                    cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
                    cv2.fillPoly(overlay, [polygon], color=color)

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image



    