from tracker import Tracker
import supervision as sv

class ObjectTracker(Tracker):

    def __init__(self, tracker, cls_tracks, cls_sv):
        Tracker.__init__(self)

        self.tracker = tracker
        self.cls_tracks = cls_tracks
        self.cls_sv = cls_sv

        # Initialize tracks dictionary
        self.tracks = {class_name: [] for class_name in self.classes_sv + self.classes_tracks}
        self.cur_frame = 0

    def detect(self, frame):
        '''
        Perform object detection on a single frame.

        Args:
            frame (array): Current frame
            conf (float): Confidence threshold for detection (default is 0.1).

        Returns:
            array: detection results.
        '''

        return self.model.predict(frame, self.conf)
        
    def track(self, detection):
        '''
        Perform object tracking on frames.

        Args:
            frames (list): List of frames to perform tracking on.

        Returns:
            dict: Dictionary containing tracks for each class.
        '''

        # Convert Ultralytics detections to supervision
        detection_sv = sv.Detections.from_ultralytics(detection)

        # Perform ByteTracker object tracking on the detections
        detection_tracks = self.tracker.update_with_detections(detection_sv)

        # Append a dictionary to store track_id and bbox of a frame
        for track in self.tracks:
            self.tracks[track].append({})

        for frame_detection in detection_tracks:
            bbox = frame_detection[0].tolist()
            class_id = frame_detection[3]
            track_id = frame_detection[4]

            # Add object to tracks if its class ID is recognized
            if detection.names[class_id] in self.cls_tracks:
                self.tracks[detection.names[class_id]][self.cur_frame][track_id] = {'bbox':bbox}

        # Update tracks with tracked objects
        for frame_detection in detection_sv:
            bbox = frame_detection[0].tolist()
            class_id = frame_detection[3]
            
            if detection.names[class_id] in self.cls_sv:
                print(frame_detection)
                self.tracks[detection.names[class_id]][self.cur_frame][track_id] = {'bbox':bbox}
        
        self.cur_frame+=1

        return self.tracks