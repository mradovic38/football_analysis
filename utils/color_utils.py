from typing import Tuple

def is_color_dark(color: Tuple[int, int, int]) -> bool:
    """
    Check if the given RGB color is dark or light using the luminance formula.

    Args:
        color (Tuple[int, int, int]): The RGB color represented as a tuple (R, G, B).

    Returns:
        bool: True if the color is dark, False if it is light.
    """
    luminance = (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])  # Corrected to use the right color channels
    return luminance < 128


def rgb_bgr_converter(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Convert a color from RGB to BGR format and vice versa.

    Args:
        color (Tuple[int, int, int]): The color represented as a tuple (R, G, B).

    Returns:
        Tuple[int, int, int]: The color converted to BGR format as a tuple (B, G, R).
    """
    return (color[2], color[1], color[0])
