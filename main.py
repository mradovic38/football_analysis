# from utils import read_video, save_video

# from club_assignment import ClubAssigner, Club
# from ball_to_player_assignment import BallToPlayerAssigner
# from cam_movement import CamMovementEstimator
# from perspective_transformer import PerspectiveTransformer


from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import Annotator

import numpy as np

def main():


    obj_tracker = ObjectTracker(
        model_id = 'smart-football-object-detection-icwha/3'
    )

    kp_tracker = KeypointsTracker(
        model_id='football-field-detection-f07vi-apxzb/1',
        conf=.3
    )

    # Assign Clubs
    club1 = Club('Club1', (229, 244, 248), (0, 0, 0))
    club2 = Club('Club2', (172, 251, 145), (239, 156, 132))


    club_assigner = ClubAssigner(club1, club2)

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

    annotator = Annotator(obj_tracker, kp_tracker, club_assigner, ball_player_assigner, top_down_keypoints,'input_videos/field_2d_v2.png')
    
    process_video(annotator, 'input_videos/08fd33_6.mp4', 'output_videos/test.avi')




    # # Video read
    # frames = read_video('input_videos/08fd33_6.mp4')

    # # cv2.imwrite('output_videos/frame.jpg', frames[0])
    # # return

    # # Init tracker
    # tracker = ObjectTracker('models/object-detection.pt', 
    #                         classes_with_tracks=['goalkeeper', 'player', 'referee'],
    #                         classes_sv=['ball'])

    # tracks = tracker.track_objects(frames,
    #                                from_stub=True,
    #                                stub_path='stubs/tracks.pkl') 
    
    # # Ball position interpolation
    # tracks['ball'] = tracker.interpolate_positions(tracks['ball'])

    # # Get object positions
    # tracker.add_track_positions(tracks)

    # # Track camera movement
    # cam_movement_estimator = CamMovementEstimator(frames[0], 0)
    # cam_movement = cam_movement_estimator.get_cam_movement(frames,
    #                                                      from_stub=True,
    #                                                      stub_path='stubs/cam_movement.pkl')
    
    # cam_movement_estimator.adjust_positions_to_tracks(tracks, cam_movement)

    # perspective_transformer = PerspectiveTransformer(110, 68)
    # perspective_transformer.transform_positions(tracks)
    # #print(cam_movement)
    
    
    # # for _, item in tracks['Player'][0].items():
    # #     bbox = item['bbox']

    # #     frame = frames[0]

    # #     img_cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    # #     cv2.imwrite('output_videos/player_cropped.jpg', img_cropped)
        
    # #     break
    # # return

    

    # # Assign Clubs
    # club1 = Club('Club1', (255, 255, 255), (0, 0, 0))
    # club2 = Club('Club2', (179, 255, 147), (239, 156, 132))


    # club_assigner = ClubAssigner(club1, club2)
    # #club_assigner.assign_club(frames[0], tracks['player'][0])
    # club_assigner.assign_clubs(frames, tracks)

    # # Assign Ball to player
    # ball_player_assigner = BallToPlayerAssigner(club1.name, club2.name)
    # ball_possessors = ['player', 'goalkeeper']

    # for ball_possessor in ball_possessors:
    #     for cur_frame, player in enumerate(tracks[ball_possessor]):
    #         ball_bbox = tracks['ball'][cur_frame][1]['bbox']
    #         player_w_ball = ball_player_assigner.assign(player, ball_bbox)

    #         if player_w_ball != -1:
    #             tracks[ball_possessor][cur_frame][player_w_ball]['has_ball'] = True

    # # Draw annotations
    # detection_shapes = {'player': 'ellipse', 'goalkeeper': 'ellipse', 'referee': 'ellipse', 'ball':'triangle'}
    # detection_colors = {'player': (255, 0, 0), 'goalkeeper': (0, 255, 0), 'referee': (0, 0, 255), 'ball': (125, 0, 125)}
    # goalkeeper_class = 'goalkeeper'
    # possessions = ball_player_assigner.get_ball_possessions()
    # #print(possessions)
    # output_frames = tracker.draw_annotations(frames, tracks, goalkeeper_class, detection_shapes, detection_colors, possessions, [club1.player_jersey_color, club2.player_jersey_color])

    # # Draw camera movement
    # output_frames = cam_movement_estimator.draw_movement(output_frames, cam_movement)

    # # Video save
    # save_video(output_frames, 'output_videos/video_out.avi')

if __name__ == '__main__':
    main()
