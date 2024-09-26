import cv2
import os
import glob
import tempfile
import shutil
import time
import numpy as np

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

def _convert_frames_to_video(frame_dir, output_video, fps, frame_size, original_frame_count):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    processed_frame_count = len(frame_files)

    if processed_frame_count <= 0:
        out.release()
        print("There are no frames to save")
        return

    
    if processed_frame_count < original_frame_count:
        duplication_ratio = original_frame_count / processed_frame_count
    else:
        duplication_ratio = 1
    
    for filename in frame_files:
        img = cv2.imread(filename)
        repeat_count = int(np.ceil(duplication_ratio))
        for _ in range(repeat_count):
            out.write(img)
    
    out.release()
    print(f"Video saved as {output_video}")
    print(f"Original frame count: {original_frame_count}")
    print(f"Processed frame count: {processed_frame_count}")
    print(f"Frame duplication ratio: {duplication_ratio:.2f}")

def process_video(processor, video_source=0, output_video="output.mp4"):
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the time per frame
        frame_time = 1 / max(fps,1)

        # Variables for FPS calculation
        start_time = time.time()
        fps_calc_interval = 1  # Calculate FPS every 1 second
        frames_in_interval = 0
        current_fps = 1e-6

        
        try:
            prev_time = time.time()
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                elapsed_time =  max(current_time - prev_time, 1e-6)

                
                
                # If enough time has passed to process this frame
                if elapsed_time >= frame_time:
                    # Process the frame
                    processed_frame = processor.process(frame, current_fps)
                    
                    # Save the processed frame as an image
                    frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_filename, processed_frame)
                    
                    frame_count += 1
                    frames_in_interval += 1
                    
                    # Display the processed frame (optional)
                    cv2.imshow('Processed Video', processed_frame)
                    
                    # Reset the timer
                    prev_time = current_time
                    
                    # Calculate frames to skip
                    frames_to_skip = int(elapsed_time / frame_time) - 1
                    for _ in range(frames_to_skip):
                        cap.grab()  # Skip frames if processing is slower than frame rate

                # Calculate and update FPS
                if current_time - start_time >= fps_calc_interval:
                    interval = max(current_time - start_time, 1e-6)  # Ensure we don't divide by zero
                    current_fps = frames_in_interval / interval
                    print(f"Current FPS: {current_fps}")  # Debug print
                    frames_in_interval = 0
                    start_time = current_time

                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Convert saved frames to video
            _convert_frames_to_video(temp_dir, output_video, fps, (width, height), total_frames)



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
