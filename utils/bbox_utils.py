from typing import Tuple

def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculate the center coordinates of a bounding box.

    Args:
        bbox (Tuple[float, float, float, float]): The bounding box defined by (x1, y1, x2, y2).

    Returns:
        Tuple[float, float]: The center coordinates (center_x, center_y) of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_bbox_width(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate the width of a bounding box.

    Args:
        bbox (Tuple[float, float, float, float]): The bounding box defined by (x1, y1, x2, y2).

    Returns:
        float: The width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return x2 - x1

def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (Tuple[float, float]): The first point (x1, y1).
        p2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        float: The distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def point_coord_diff(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the coordinate differences between two points.

    Args:
        p1 (Tuple[float, float]): The first point (x1, y1).
        p2 (Tuple[float, float]): The second point (x2, y2).

    Returns:
        Tuple[float, float]: The differences (dx, dy) between the two points.
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def get_feet_pos(bbox: Tuple[float, float, float, float]) -> Tuple[float, int]:
    """
    Calculate the feet position from a bounding box.

    Args:
        bbox (Tuple[float, float, float, float]): The bounding box defined by (x1, y1, x2, y2).

    Returns:
        Tuple[float, int]: The feet position as (feet_x, feet_y), where feet_y is rounded to an integer.
    """
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, int(y2)
