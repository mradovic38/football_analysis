from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from speed_estimation import SpeedEstimator
from .frame_number_annotator import FrameNumberAnnotator
from file_writing import TracksJsonWriter
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner
from ball_to_player_assignment import BallToPlayerAssigner
from utils import rgb_bgr_converter

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):
    """
    A video processor for football footage that tracks objects and keypoints,
    estimates speed, assigns the ball to player, calculates the ball possession 
    and adds various annotations.
    """

    def __init__(self, obj_tracker: ObjectTracker, kp_tracker: KeypointsTracker, 
                 club_assigner: ClubAssigner, ball_to_player_assigner: BallToPlayerAssigner, 
                 top_down_keypoints: np.ndarray, field_img_path: str, 
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True) -> None:
        """
        Initializes the video processor with necessary components for tracking, annotations, and saving tracks.

        Args:
            obj_tracker (ObjectTracker): The object tracker for tracking players and balls.
            kp_tracker (KeypointsTracker): The keypoints tracker for detecting and tracking keypoints.
            club_assigner (ClubAssigner): Assigner to determine clubs for the tracked players.
            ball_to_player_assigner (BallToPlayerAssigner): Assigns the ball to a specific player based on tracking.
            top_down_keypoints (np.ndarray): Keypoints to map objects to top-down positions.
            field_img_path (str): Path to the image of the football field used for projection.
            save_tracks_dir (Optional[str]): Directory to save tracking information. If None, no tracks will be saved.
            draw_frame_num (bool): Whether or not to draw current frame number on the output video.
        """

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.draw_frame_num = draw_frame_num
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator() 

        if save_tracks_dir:
            self.save_tracks_dir = save_tracks_dir
            self.writer = TracksJsonWriter(save_tracks_dir)
        
        field_image = cv2.imread(field_img_path)
        # Convert the field image to grayscale (black and white)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to 3 channels (since the main frame is 3-channel)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        # Initialize the speed estimator with the field image's dimensions
        self.speed_estimator = SpeedEstimator(field_image.shape[1], field_image.shape[0])
        
        self.frame_num = 0

        self.field_image = field_image

    def process(self, frames: List[np.ndarray], fps: float = 1e-6) -> List[np.ndarray]:
        """
        Processes a batch of video frames, detects and tracks objects, assigns ball possession, and annotates the frames.

        Args:
            frames (List[np.ndarray]): List of video frames.
            fps (float): Frames per second of the video, used for speed estimation.

        Returns:
            List[np.ndarray]: A list of annotated video frames.
        """
        
        self.cur_fps = max(fps, 1e-6)

        # Detect objects and keypoints in all frames
        batch_obj_detections = self.obj_tracker.detect(frames)
        batch_kp_detections = self.kp_tracker.detect(frames)

        processed_frames = []

        # Process each frame in the batch
        for idx, (frame, object_detection, kp_detection) in enumerate(zip(frames, batch_obj_detections, batch_kp_detections)):
            
            # Track detected objects and keypoints
            obj_tracks = self.obj_tracker.track(object_detection)
            kp_tracks = self.kp_tracker.track(kp_detection)

            # Assign clubs to players based on their tracked position
            obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)

            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

            # Map objects to a top-down view of the field
            all_tracks = self.obj_mapper.map(all_tracks)

            # Assign the ball to the closest player and calculate speed
            all_tracks['object'], _ = self.ball_to_player_assigner.assign(
                all_tracks['object'], self.frame_num, 
                all_tracks['keypoints'].get(8, None),  # keypoint for player 1
                all_tracks['keypoints'].get(24, None)  # keypoint for player 2
            )

            # Estimate the speed of the tracked objects
            all_tracks['object'] = self.speed_estimator.calculate_speed(
                all_tracks['object'], self.frame_num, self.cur_fps
            )
            
            # Save tracking information if saving is enabled
            if self.save_tracks_dir:
                self._save_tracks(all_tracks)

            self.frame_num += 1

            # Annotate the current frame with the tracking information
            annotated_frame = self.annotate(frame, all_tracks)

            # Append the annotated frame to the processed frames list
            processed_frames.append(annotated_frame)

        return processed_frames

    
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates the given frame with analised data

        Args:
            frame (np.ndarray): The current video frame to be annotated.
            tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.

        Returns:
            np.ndarray: The annotated video frame.
        """
         
        # Draw the frame number if required
        if self.draw_frame_num:
            frame = self.frame_num_annotator.annotate(frame, {'frame_num': self.frame_num})
        
        # Annotate the frame with keypoint and object tracking information
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        
        # Project the object positions onto the football field image
        projection_frame = self.projection_annotator.annotate(self.field_image, tracks['object'])

        # Combine the frame and projection into a single canvas
        combined_frame = self._combine_frame_projection(frame, projection_frame)

        # Annotate possession on the combined frame
        combined_frame = self._annotate_possession(combined_frame)

        return combined_frame
    

    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        """
        Combines the original video frame with the projection of player positions on the field image.

        Args:
            frame (np.ndarray): The original video frame.
            projection_frame (np.ndarray): The projected field image with annotations.

        Returns:
            np.ndarray: The combined frame.
        """
        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the projection to 70% of its original size
        scale_proj = 0.7
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (top-left corner)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 25px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        return combined_frame
    

    def _annotate_possession(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotates the possession progress bar on the top-left of the frame.

        Args:
            frame (np.ndarray): The frame to be annotated.

        Returns:
            np.ndarray: The annotated frame with possession information.
        """
        frame = frame.copy()
        overlay = frame.copy()

        # Position and size for the possession overlay (top-left with 20px margin)
        overlay_width = 500
        overlay_height = 100
        gap_x = 20  # 20px from the left
        gap_y = 20  # 20px from the top

        # Draw background rectangle (black with transparency)
        cv2.rectangle(overlay, (gap_x, gap_y), (gap_x + overlay_width, gap_y + overlay_height), (0, 0, 0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Position for possession text
        text_x = gap_x + 15
        text_y = gap_y + 30

        # Display "Possession" above the progress bar
        cv2.putText(frame, 'Possession:', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)

        # Position and size for the possession bar (20px margin)
        bar_x = text_x
        bar_y = text_y + 25
        bar_width = overlay_width - bar_x
        bar_height = 15

        # Get possession data from the ball-to-player assigner
        possession = self.ball_to_player_assigner.get_ball_possessions()[-1]
        possession_club1 = possession[0]
        possession_club2 = possession[1]

        # Calculate sizes for each possession segment in pixels
        club1_width = int(bar_width * possession_club1)
        club2_width = int(bar_width * possession_club2)
        neutral_width = bar_width - club1_width - club2_width

        club1_color = self.club_assigner.club1.player_jersey_color
        club2_color = self.club_assigner.club2.player_jersey_color
        neutral_color = (128, 128, 128)

        # Convert Club Colors from RGB to BGR
        club1_color = rgb_bgr_converter(club1_color)
        club2_color = rgb_bgr_converter(club2_color)

        # Draw club 1's possession (left)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + club1_width, bar_y + bar_height), club1_color, -1)

        # Draw neutral possession (middle)
        cv2.rectangle(frame, (bar_x + club1_width, bar_y), (bar_x + club1_width + neutral_width, bar_y + bar_height), neutral_color, -1)

        # Draw club 2's possession (right)
        cv2.rectangle(frame, (bar_x + club1_width + neutral_width, bar_y), (bar_x + bar_width, bar_y + bar_height), club2_color, -1)

        # Draw outline for the entire progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)

        # Calculate the position for the possession text under the bars
        possession_club1_text = f'{int(possession_club1 * 100)}%'
        possession_club2_text = f'{int(possession_club2 * 100)}%'

        # Display possession percentages for each club
        self._display_possession_text(frame, club1_width, club2_width, neutral_width, bar_x, bar_y, possession_club1_text, possession_club2_text, club1_color, club2_color)

        return frame
    

    def _display_possession_text(self, frame: np.ndarray, club1_width: int, club2_width: int,
                                  neutral_width: int, bar_x: int, bar_y: int, 
                                 possession_club1_text: str, possession_club2_text: str, 
                                 club1_color: Tuple[int, int, int], club2_color: Tuple[int, int, int]) -> None:
        """
        Helper function to display possession percentages for each club below the progress bar.

        Args:
            frame (np.ndarray): The frame where the text will be displayed.
            club1_width (int): Width of club 1's possession bar.
            club2_width (int): Width of club 2's possession bar.
            neutral_width (int): Width of the neutral possession area.
            bar_x (int): X-coordinate of the progress bar.
            bar_y (int): Y-coordinate of the progress bar.
            possession_club1_text (str): Text for club 1's possession percentage.
            possession_club2_text (str): Text for club 2's possession percentage.
            club1_color (tuple): BGR color of club 1.
            club2_color (tuple): BGR color of club 2.
        """
        # Text for club 1
        club1_text_x = bar_x + club1_width // 2 - 10  # Center of club 1's possession bar
        club1_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club1_color, 1)  # Club 1's color

        # Text for club 2
        club2_text_x = bar_x + club1_width + neutral_width + club2_width // 2 - 10  # Center of club 2's possession bar
        club2_text_y = bar_y + 35  # 20 pixels below the bar
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club2_color, 1)  # Club 2's color



    def _save_tracks(self, all_tracks: Dict[str, Dict[int, np.ndarray]]) -> None:
        """
        Saves the tracking information for objects and keypoints to the specified directory.

        Args:
            all_tracks (Dict[str, Dict[int, np.ndarray]]): A dictionary containing tracking data for objects and keypoints.
        """
        self.writer.write(self.writer.get_object_tracks_path(), all_tracks['object'])
        self.writer.write(self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])

    

    