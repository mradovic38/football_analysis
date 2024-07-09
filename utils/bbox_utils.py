def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return x2 - x1

def point_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def point_coord_diff(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_feet_pos(bbox):
    x1, _, x2, y2 = bbox
    return (x1+x2)/2, int(y2)