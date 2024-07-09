import cv2

def read_video(path):
    """
    Load a video file and return its frames.

    Args:
        path (str): Path to the video file.

    Returns:
        list: List of frames.
    """

    # Open the video file
    cap = cv2.VideoCapture(path)
    frames = []

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return []

    # Read frames until the video ends
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Append the frame to the list of frames
        frames.append(frame)

    # Release the video capture object
    cap.release()
    
    return frames 


def save_video(out_frames, out_vpath, fps=30.0):
    '''
    Save frames as a video.
    
    Args:
        out_frames (list): List of frames to be saved as video.
        out_vpath (str): Output video file path.
        fps (float): Frames per second for the output video (default is 30.0).
    '''
    
    # Check if any frames to save
    if len(out_frames) == 0:
        print("Error: No frames to save.")
        return

    # Get the shape of the first frame
    height, width, _ = out_frames[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_vpath, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in out_frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
