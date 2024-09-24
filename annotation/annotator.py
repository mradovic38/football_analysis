from .abstract_annotator import AbstractAnnotator
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from speed_estimation import SpeedEstimator

import cv2
import numpy as np

class Annotator(AbstractAnnotator):

    def __init__(self, obj_tracker, kp_tracker, club_assigner, ball_to_player_assigner, top_down_keypoints, field_img_path):
        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.possession_tracker = ball_to_player_assigner.possession_tracker
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)

        
        
        field_image = cv2.imread(field_img_path)
        # Convert the field image to grayscale (black and white)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale back to 3 channels (since the main frame is 3-channel)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        self.speed_estimator = SpeedEstimator(field_image.shape[1], field_image.shape[0])
        
        self.frame_num = 0

        self.field_image = field_image

    def __call__(self, frame):
        obj_detections = self.obj_tracker.detect(frame)
        kp_detections = self.kp_tracker.detect(frame)

        obj_tracks = self.obj_tracker.track(obj_detections)
        kp_tracks = self.kp_tracker.track(kp_detections)

        obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)
        
        obj_tracks, _ = self.ball_to_player_assigner.assign(obj_tracks, self.frame_num)


        all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}

        all_tracks = self.obj_mapper.map(all_tracks)

        all_tracks['object'] = self.speed_estimator.calculate_speed(all_tracks['object'], self.frame_num)

        print(all_tracks)

        self.frame_num += 1

        return self.annotate(frame, all_tracks)

    
    def annotate(self, frame, tracks):
        # Annotate the main frame with object and keypoint annotations
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        
        # Project the player and object positions on the football field image
        projection_frame = self.projection_annotator.annotate(self.field_image.copy(), tracks['object'])

        # Target canvas size
        canvas_width, canvas_height = 1920, 1080
        
        # Get dimensions of the original frame and projection frame
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape

        # Scale the frames (keep original size of the main frame)
        scale_proj = .7  # Scale the projection to 70% of its original size
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))

        # Create a blank canvas of 1920x1080
        combined_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the main frame onto the canvas (keep it full-size)
        combined_frame[:h_frame, :w_frame] = frame

        # Set the position for the projection frame at the bottom-middle
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25  # 50px margin from bottom

        # Blend the projection with 75% visibility (alpha transparency)
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset + new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha, overlay, 1 - alpha, 0, overlay)

        # Annotate possession
        combined_frame = self._annotate_possession(combined_frame)

        return combined_frame
    

    def _annotate_possession(self, frame):
        overlay = frame.copy()

        # Position and size for the overlay (top-left with 20px margin)
        overlay_width = 500
        overlay_height = 100
        gap_x = 20  # 20px from the left
        gap_y = 20  # 20px from the top

        # Draw background rectangle (black with transparency)
        cv2.rectangle(overlay, (gap_x, gap_y), (gap_x + overlay_width, gap_y + overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text_x = gap_x + 10
        text_y = gap_y + 10 + 15
        # Write "Possession" above the progress bar
        cv2.putText(frame, 'Possession:', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)

        # Position and size for the possession bar (20px margin)
        bar_x = text_x
        bar_y = text_y + 25
        bar_width = overlay_width - bar_x
        bar_height = 15

        # Get possession data from possession_dict
        possession = self.possession_tracker.possession[-1]
        possession_club1 = possession[0]
        possession_club2 = possession[1]

        # Calculate sizes for each possession segment in pixels
        club1_width = int(bar_width * possession_club1)
        club2_width = int(bar_width * possession_club2)
        neutral_width = bar_width - club1_width - club2_width

        club1_color = self.club_assigner.club1.player_jersey_color
        club2_color = self.club_assigner.club2.player_jersey_color
        neutral_color = (128, 128, 128)

        # Draw club 1's possession (on the left)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + club1_width, bar_y + bar_height), club1_color, -1)

        # Draw neutral possession (in the middle, gray)
        cv2.rectangle(frame, (bar_x + club1_width, bar_y), (bar_x + club1_width + neutral_width, bar_y + bar_height), neutral_color, -1)

        # Draw club 2's possession (on the right)
        cv2.rectangle(frame, (bar_x + club1_width + neutral_width, bar_y), (bar_x + bar_width, bar_y + bar_height), club2_color, -1)

        # Draw outline for the entire progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)

        # Calculate the position for the possession text under the bars
        possession_club1_text = f'{int(possession_club1 * 100)}%'
        possession_club2_text = f'{int(possession_club2 * 100)}%'

        # Text for club 1
        club1_text_x = bar_x + club1_width // 2 - 20  # Center of club 1's possession bar
        club1_text_y = bar_y + bar_height + 20  # 20 pixels below the bar
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club1_text, (club1_text_x, club1_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club1_color, 1)  # Club 1's color

        # Text for club 2
        club2_text_x = bar_x + club1_width + neutral_width + club2_width // 2 - 20  # Center of club 2's possession bar
        club2_text_y = bar_y + bar_height + 20  # 20 pixels below the bar
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
        cv2.putText(frame, possession_club2_text, (club2_text_x, club2_text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, club2_color, 1)  # Club 2's color

        return frame