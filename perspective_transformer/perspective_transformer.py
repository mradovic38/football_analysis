import numpy as np
import cv2

class PerspectiveTransformer():
    def __init__(self, field_length, field_width):
        field_length = field_length / 18

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ]).astype(np.float32)

        self.target_vertices = np.array([
            [0, field_width],
            [0, 0],
            [field_length, 0],
            [field_length, field_width]
        ]).astype(np.float32)

        self.transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)


    def transform_point(self, point):

        if not cv2.pointPolygonTest(self.pixel_vertices, (int(point[0]), int(point[1])), False) >= 0:
            return None
        
        point = point.reshape(-1, 1, 2).astype(np.float32)
        point = cv2.perspectiveTransform(point, self.transformer)

        return point.reshape(-1, 2)

        
    def transform_positions(self, tracks):
        for obj, obj_tracks in tracks.items():
            for i, track in enumerate(obj_tracks):
                for track_id, t in track.items():
                    pos = np.array(t['position_adj'])
                    pos = self.transform_point(np.array(pos))
                    if pos is not None:
                        tracks[obj][i][track_id]['position_transformed'] = pos.squeeze().tolist()

    