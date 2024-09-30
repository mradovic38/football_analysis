from typing import Tuple

def is_color_dark(color: Tuple[int, int, int]) -> bool:
    """
    Check if the color is dark or light using luminance.
    """
    luminance = (0.2126 * color[2] + 0.7152 * color[1] + 0.0722 * color[1])  # Luminance formula
    return luminance < 128


def rgb_bgr_converter(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Convert color from BGR to RGB color code and vice versa
    """
    return (color[2], color[1], color[0])