from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor

import numpy as np

def main():


    obj_tracker = ObjectTracker(
        model_path = 'models/weights/object-detection.pt'
    )

    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt',
        conf=.3
    )

    # Assign Clubs
    club1 = Club('Club1', (232, 247, 248), (6, 25, 21))
    club2 = Club('Club2', (172, 251, 145), (239, 156, 132))


    club_assigner = ClubAssigner(club1, club2, images_to_save=25, images_save_path='output_videos')

    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351], # 0-5
        [32, 122], [32, 229], # 6-7
        [64, 176], # 8
        [96, 57], [96, 122], [96, 229], [96, 293], # 9-12
        [263, 0], [263, 122], [263, 229], [263, 351], # 13-16
        [431, 57], [431, 122], [431, 229], [431, 293], # 17-20
        [463, 176], # 21
        [495, 122], [495, 229], # 22-23
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351], # 24-29
        [210, 176], [317, 176] # 30-31
    ])

    processor = FootballVideoProcessor(obj_tracker, 
                                       kp_tracker, 
                                       club_assigner, 
                                       ball_player_assigner, 
                                       top_down_keypoints,
                                       'input_videos/field_2d_v2.png', 
                                       save_tracks_dir='output_videos',
                                       draw_frame_num=True)
    
    process_video(processor, video_source='input_videos/08fd33_6.mp4', output_video='output_videos/test.mp4', batch_size=10, skip_seconds=18)


if __name__ == '__main__':
    main()
