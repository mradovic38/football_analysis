def is_color_dark(color):
    """
    Check if the color is dark or light using luminance.
    """
    luminance = (0.2126 * color[2] + 0.7152 * color[1] + 0.0722 * color[1])  # Luminance formula
    return luminance < 128