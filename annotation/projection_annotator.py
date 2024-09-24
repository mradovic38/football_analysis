from .abstract_annotator import AbstractAnnotator

import cv2
import numpy as np
from scipy.spatial import Voronoi

class ProjectionAnnotator(AbstractAnnotator):

    def _is_color_dark(self, color):
        """
        Check if the color is dark or light using luminance.
        """
        luminance = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])  # Luminance formula
        return luminance < 128

    def _draw_outline(self, frame, pos, shape='circle', size=10, is_dark=True):
        """
        Draw a white or black outline around the object based on color and shape.
        """
        outline_color = (255, 255, 255) if is_dark else (0, 0, 0)

        if shape == 'circle':
            # Draw a larger circle outline
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=size + 2, color=outline_color, thickness=2)
        elif shape == 'square':
            # Draw a larger square outline
            top_left = (int(pos[0]) - (size + 2), int(pos[1]) - (size + 2))
            bottom_right = (int(pos[0]) + (size + 2), int(pos[1]) + (size + 2))
            cv2.rectangle(frame, top_left, bottom_right, color=outline_color, thickness=2)
        elif shape == 'dashed_circle':
            # Draw a dashed circle outline for the referee
            dash_length = 30  # Length of each dash in degrees
            gap_length = 30   # Gap between dashes in degrees

            # We will create a series of dashes by calculating the start and end points of each dash
            for i in range(0, 360, dash_length + gap_length):  # 45 degrees steps for each dash
                start_angle_rad = np.radians(i)
                end_angle_rad = np.radians(i + dash_length)

                # Calculate the start and end points of each dash
                start_x = int(pos[0]) + int((size + 2) * np.cos(start_angle_rad))
                start_y = int(pos[1]) + int((size + 2) * np.sin(start_angle_rad))
                end_x = int(pos[0]) + int((size + 2) * np.cos(end_angle_rad))
                end_y = int(pos[1]) + int((size + 2) * np.sin(end_angle_rad))

                # Draw each dash as a line segment
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color=(0, 0, 0), thickness=2)

        elif shape == 'plus':
            # Draw an outline for the plus sign (ball)
            cv2.line(frame, (int(pos[0]) - (size), int(pos[1])), 
                    (int(pos[0]) + (size), int(pos[1])), color=outline_color, thickness=10)
            cv2.line(frame, (int(pos[0]), int(pos[1]) - (size)), 
                    (int(pos[0]), int(pos[1]) + (size)), color=outline_color, thickness=10)

    def annotate(self, frame, tracks):
        frame = frame.copy()

        frame = self._draw_voronoi(frame, tracks)
        

        # Draw all other objects (players, goalkeepers, referees) first
        for class_name, track_data in tracks.items():
            if class_name != 'ball':  # Skip the ball for now, we'll draw it later
                for track_id, track_info in track_data.items():
                    # Get the projected position on the field
                    proj_pos = track_info['projection']  # (x, y) tuple
                    color = track_info.get('club_color', (255, 255, 255))  # Default color if not provided
                    is_dark_color = self._is_color_dark(color)

                    if class_name == 'player' or class_name == 'goalkeeper':
                        # Determine if the player or goalkeeper has possession of the ball
                        if track_info.get('has_ball', False):
                            # Draw a player/goalkeeper with possession accent (green ring)
                            shape = 'square' if class_name == 'goalkeeper' else 'circle'
                            self._draw_outline(frame, proj_pos, shape=shape, is_dark=is_dark_color)
                            cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=15, color=(0, 255, 0), thickness=2)  # Green ring for possession

                            # Draw the player or goalkeeper marker (circle or square)
                            if shape == 'circle':
                                cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=10, color=color, thickness=-1)
                            else:
                                top_left = (int(proj_pos[0]) - 10, int(proj_pos[1]) - 10)
                                bottom_right = (int(proj_pos[0]) + 10, int(proj_pos[1]) + 10)
                                cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=-1)

                        else:
                            # Draw a player/goalkeeper without possession
                            shape = 'square' if class_name == 'goalkeeper' else 'circle'
                            self._draw_outline(frame, proj_pos, shape=shape, is_dark=is_dark_color)
                            if shape == 'circle':
                                cv2.circle(frame, (int(proj_pos[0]), int(proj_pos[1])), radius=10, color=color, thickness=-1)
                            else:
                                top_left = (int(proj_pos[0]) - 10, int(proj_pos[1]) - 10)
                                bottom_right = (int(proj_pos[0]) + 10, int(proj_pos[1]) + 10)
                                cv2.rectangle(frame, top_left, bottom_right, color=color, thickness=-1)

                    elif class_name == 'referee':
                        # Draw a circle with black dotted outline for the referee
                        self._draw_outline(frame, proj_pos, shape='dashed_circle', is_dark=is_dark_color)

        # Now draw the ball last so it appears on top
        if 'ball' in tracks:
            for track_id, track_info in tracks['ball'].items():
                proj_pos = track_info['projection']  # (x, y) tuple

                # First, draw the outline for the ball 
                is_dark_color = self._is_color_dark((0, 255, 255))  # Yellow color
                self._draw_outline(frame, proj_pos, shape='plus', is_dark=is_dark_color)

                # Then, draw the ball as a yellow plus sign
                color = (0, 255, 255)  # Yellow
                cv2.line(frame, (int(proj_pos[0]) - 10, int(proj_pos[1])), 
                        (int(proj_pos[0]) + 10, int(proj_pos[1])), color=color, thickness=6)
                cv2.line(frame, (int(proj_pos[0]), int(proj_pos[1]) - 10), 
                        (int(proj_pos[0]), int(proj_pos[1]) + 10), color=color, thickness=6)

        return frame

                    
    def _draw_voronoi(self, image, tracks):
        # Get the image dimensions
        height, width = image.shape[:2]

        # Create an overlay to draw the Voronoi diagram
        overlay = image.copy()

        # Extract player and goalkeeper positions from object tracks
        points = []
        player_colors = []

        # Loop through 'player' and 'goalkeeper' classes only
        for class_name in ['player', 'goalkeeper']:
            track_data = tracks.get(class_name, {})
            for track_id, track_info in track_data.items():
                x, y = track_info['projection'][:2]  # Get the projected positions
                points.append([x, y])
                player_colors.append(track_info['club_color'])  # Color based on club/team

        # Add extra points far outside the pitch boundaries to ensure full coverage
        boundary_margin = 1000  # Push boundary points far outside the image dimensions

        boundary_points = [
            [-boundary_margin, -boundary_margin],               # Top-left corner (far outside)
            [width // 2, -boundary_margin],                     # Top-center (far outside)
            [width + boundary_margin, -boundary_margin],        # Top-right corner (far outside)
            [-boundary_margin, height // 2],                    # Mid-left (far outside)
            [width + boundary_margin, height // 2],             # Mid-right (far outside)
            [-boundary_margin, height + boundary_margin],       # Bottom-left corner (far outside)
            [width // 2, height + boundary_margin],             # Bottom-center (far outside)
            [width + boundary_margin, height + boundary_margin] # Bottom-right corner (far outside)
        ]

        # Add boundary points to the list of Voronoi points
        points.extend(boundary_points)
        
        # Dummy color (gray) for boundary points
        boundary_color = (128, 128, 128)
        player_colors.extend([boundary_color] * len(boundary_points))

        # Ensure there are enough points to create Voronoi diagram
        if len(points) > 2:
            points = np.array(points)
            vor = Voronoi(points)

            # Iterate through each Voronoi region and plot it
            for region_index, region in enumerate(vor.point_region):
                if not -1 in vor.regions[region] and len(vor.regions[region]) > 0:
                    polygon = [vor.vertices[i] for i in vor.regions[region]]
                    polygon = np.array(polygon, np.int32)
                    polygon = polygon.reshape((-1, 1, 2))

                    # Draw the polygon on the overlay with 60% transparency
                    color = player_colors[region_index] if region_index < len(player_colors) else boundary_color
                    cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
                    cv2.fillPoly(overlay, [polygon], color=color)

        # Blend the overlay with the original image using 60% alpha for the Voronoi regions
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        return image



    