from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_bbox_width, get_bbox_center, get_feet_pos
import cv2
import numpy as np
import pandas as pd

class ObjectTracker:
    def __init__(self, model_path, classes_with_tracks, classes_sv):
        '''
        Initialize the ObjectTracker.

        Args:
            model_path (str): Path to the model.
            classes_with_tracks (list): List of classes for which tracks will be maintained.
            classes_sv (list): List of classes for which supervision is applied.
        '''
         
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.classes_tracks = classes_with_tracks 
        self.classes_sv = classes_sv

    def interpolate_positions(self, positions):
        # Turn into array
        positions = [x.get(list(x.keys())[0] if x.keys() else 0, {}).get('bbox', []) for x in positions]
        # Convert to pandas dataframe
        df_pos = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing positions
        df_pos.interpolate(inplace=True)
        #df_pos.fillna(method='ffill', inplace=True)

        # If missing in the first frames
        df_pos.bfill(inplace=True)

        positions = [{1 : {'bbox': x}} for x in df_pos.to_numpy().tolist()]
    
        return positions

    def detect(self, frames, batch_sz=30, conf=0.1):
        '''
        Perform object detection on frames.

        Args:
            frames (list): List of frames to perform detection on.
            batch_sz (int): Batch size for detection (default is 30).
            conf (float): Confidence threshold for detection (default is 0.1).

        Returns:
            list: List of detection results.
        '''

        res = []

        for i in range(0, len(frames), batch_sz):
            res += self.model.predict(frames[i:i+batch_sz], conf)
        
        return res
    

    def track_objects(self, frames, from_stub=False, stub_path=None):
        '''
        Perform object tracking on frames.

        Args:
            frames (list): List of frames to perform tracking on.
            from_stub (bool): Whether to load tracks from a pickle file (default is False).
            stub_path (str): Path to the pickle file containing tracks (default is None).

        Returns:
            dict: Dictionary containing tracks for each class.
        '''

        # Load tracks from a pickle file if requested
        if from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as file:
                tracks = pickle.load(file)
                return tracks
            
        # Initialize tracks dictionary
        tracks = {class_name: [] for class_name in self.classes_sv + self.classes_tracks}


        # Perform object detection on frames
        detections = self.detect(frames)

        

        # Iterate over each frame and its corresponding detections
        for cur_frame, detection in enumerate(detections):
            # Map class names to class IDs (inverse of detection.names)
            #clss = {v:k for k,v in detection.names.items()}
            
            # Convert Ultralytics detections to supervision
            detection_sv = sv.Detections.from_ultralytics(detection)

            # Perform ByteTracker object tracking on the detections
            detection_tracks = self.tracker.update_with_detections(detection_sv)


            # append a dictionary to store track_id and bbox of a frame
            for track in tracks:
                tracks[track].append({})

            
            # Update tracks with detected objects
            for frame_detection in detection_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                # Add object to tracks if its class ID is recognized
                if detection.names[class_id] in self.classes_tracks:
                    tracks[detection.names[class_id]][cur_frame][track_id] = {'bbox':bbox}

            # Update tracks with tracked objects
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                
                if detection.names[class_id] in self.classes_sv:
                    print(frame_detection)
                    print('new_frame')
                    #all_sv_classes_detections = [d for d in frame_detection if d['class'] == class_id]
                    #confidence_scores = [d['confidence'] for d in all_sv_classes_detections]
                    #print(confidence_scores)
                    tracks[detection.names[class_id]][cur_frame][track_id] = {'bbox':bbox}
                    #had_sv[had_sv_id] = frame_detection.probs[class_id]
    

        # Save tracks to a pickle file if requested
        if stub_path:
            with open(stub_path, 'wb') as file:
                pickle.dump(tracks, file)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id, position, is_keeper=False):
        
        y = int(bbox[3])
        x, _ = get_bbox_center(bbox)
        x = int(x)
        w = int(get_bbox_width(bbox))

        if is_keeper:
            
            for thickness in range(1, 3):
                cv2.ellipse(frame,
                            center=(x, y), 
                            axes=(w+5*(thickness-1), 20),
                            angle=0,
                            startAngle=-30,
                            endAngle=240,
                            color=color,
                            thickness=thickness,
                            lineType=cv2.LINE_AA)
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
                    color=(255-color[0], 255-color[1], 255-color[2]),
                    thickness=2
                    )
        
        x2 = x - len(f"X: {position[0]: .2f} Y: {position[1]: .2f}") * 5
        cv2.putText(frame,
                    text=f"X: {position[0]: .2f} Y: {position[1]: .2f}",
                    org=(x2, y + 20),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=(255-color[0], 255-color[1], 255-color[2]),
                    thickness=2
                    )


        return frame
    
    def draw_triangle(self, frame, bbox, color, is_keeper=False):
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)
        x = int(x)

        points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])


        cv2.drawContours(frame,
                         [points],
                         0,
                         color,
                         cv2.FILLED)
        cv2.drawContours(frame,
                         [points],
                         0,
                         (255-color[0], 255-color[1], 255-color[2]),
                         2)
        
        return frame

    def draw_possessions(self, frame, frame_idx, team_colors, possessions):
        overlay = frame.copy()

        cv2.rectangle(overlay, (1300, 850), (1920, 1000), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        w = 25
        h = 25
        cv2.rectangle(frame,
                      (1350 - w, 900 - h),
                      (1350 + w, 900 + h),
                      team_colors[0],
                      cv2.FILLED)
        cv2.putText(frame,
                    f"Ball control: {possessions[frame_idx][0]*100: .2f}%",
                    (1380, 915),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
                    color=(0,0,0),
                    thickness=3)
        
        cv2.rectangle(frame,
                      (1350 - w, 950 - h),
                      (1350 + w, 950 + h),
                      team_colors[1],
                      cv2.FILLED)
        
        cv2.putText(frame,
                    f"Ball control: {possessions[frame_idx][1]*100: .2f}%",
                    (1380, 965),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
                    color=(0,0,0),
                    thickness=3)
        
        return frame
    
    def draw_annotations(self, frames, tracks, goalkeeper_class, detection_shapes, detection_colors, possessions, club_colors):
        output_frames = []

        for i, frame in enumerate(frames):
            frame = frame.copy()

            frame = self.draw_possessions(frame, i, club_colors, possessions)

            for track in tracks:
                for track_id, item in tracks[track][i].items():
                    color = detection_colors.get(track, (255, 0, 0))
                    color = item.get('club_color', color)

                    if 'has_ball' in item and item['has_ball']:
                        frame = self.draw_triangle(frame, item['bbox'], color)

                    
                    if track in detection_shapes:
                        match detection_shapes[track]:
                            case 'ellipse':
                                #color = detection_colors.get(track, (255, 0, 0))
                                #color = item.get('club_color', (255, 0, 0))
                                pos = item.get('position_transformed', (0, 0))
                                frame = self.draw_ellipse(frame, item['bbox'], color, track_id, pos, track==goalkeeper_class)
                            case 'triangle':
                                frame = self.draw_triangle(frame, item['bbox'], color, track==goalkeeper_class)
                        
            output_frames.append(frame)
        return output_frames
    

    def add_track_positions(self, tracks):
        for obj, obj_tracks in tracks.items():
            for i, track in enumerate(obj_tracks):
                for track_id, t in track.items():
                    bbox = t['bbox']
                    if obj == "ball":
                        pos = get_bbox_center(bbox)
                    else:
                        pos = get_feet_pos(bbox)
                    tracks[obj][i][track_id]['position'] = pos

                


            
           
