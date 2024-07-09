import pickle
import os
import cv2
import numpy as np
from utils.bbox_utils import point_distance, point_coord_diff

class CamMovementEstimator():
    def __init__(self, 
                 frame,
                 frame_id,
                 max_corners=100, 
                 quality_level=0.3, 
                 min_distance=3, 
                 block_size=7, 
                 window_size=(15,15),
                 max_level = 2,
                 crit = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 sensitivity=5  # minimum distance 
                 ): 
        
        self.frame_id = frame_id
        self.sensitivity = sensitivity 

        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _mask = np.zeros_like(frame_grayscale)
        _mask[:, 0:40] = 1

        self.features = dict(
            maxCorners = max_corners,
            qualityLevel = quality_level, # the higher the quality level the better the features, but the lesser the amount of features we can get
            minDistance = min_distance, # minimum distance between the features
            blockSize = block_size, # search size of the features
            mask = _mask
        )

        self.lk_params = dict(
            winSize = window_size,
            maxLevel = max_level, # downscale the image up to max_level times
            criteria = crit # stopping criteria
        )

    def adjust_positions_to_tracks(self, tracks, cam_movement):
        for obj, obj_tracks in tracks.items():
            for i, track in enumerate(obj_tracks):
                for track_id, t in track.items():
                    pos = t['position']
                    cm = cam_movement[i]
                    pos = (pos[0] - cm[0], pos[1] - cm[1])
                    tracks[obj][i][track_id]['position_adj'] = pos

    def draw_movement(self, frames, cam_movement, color=(255,255,255)):
        output_frames = []

        for i, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()

            cv2.rectangle(overlay, (0,0), (500,100), color, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            x_mov, y_mov = cam_movement[i]
            frame = cv2.putText(frame, f"Camera Movement X: {x_mov : .2f}",
                                (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
            
            frame = cv2.putText(frame, f"Camera Movement Y: {y_mov : .2f}",
                                (10,60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
            
            output_frames.append(frame)

        return output_frames

    def get_cam_movement(self, frames, from_stub=False, stub_path=None):
        # Load cam movement from a pickle file if requested
        if from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as file:
                cam_movement = pickle.load(file)
                return cam_movement
            
        cam_movement = [[0,0]] * len(frames)

        old_greyscale = cv2.cvtColor(frames[self.frame_id], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_greyscale, **self.features)

        for i in range(0, len(frames)):
            if i==self.frame_id:
                continue

            frame_grayscale = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_greyscale,
                                                          frame_grayscale,
                                                          old_features,
                                                          None,
                                                          **self.lk_params)
            
            max_dist = 0
            mov_x, mov_y = 0, 0

            for j, (old, new) in enumerate(zip(old_features, new_features)):
                old_features_point = old.ravel()
                new_features_point = new.ravel()
                
                
                dist = point_distance(old_features_point, new_features_point)

                if dist > max_dist:
                    max_dist = dist
                    mov_x, mov_y = point_coord_diff(old_features_point, new_features_point)

    
       
            if max_dist > self.sensitivity:
                cam_movement[i] = [mov_x, mov_y]
                old_features = cv2.goodFeaturesToTrack(frame_grayscale, **self.features)

            old_greyscale = frame_grayscale.copy()


        if stub_path:
            with open(stub_path, 'wb') as file:
                pickle.dump(cam_movement, file)

        return cam_movement
    


                    
