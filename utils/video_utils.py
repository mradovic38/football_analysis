import cv2
import os
import glob
import queue
import threading
import tempfile
import time
import signal
from typing import List, Tuple, Optional, Type
import numpy as np

def read_video(path: str) -> List[np.ndarray]:
    """
    Load a video file and return its frames.

    Args:
        path (str): Path to the video file.

    Returns:
        List[np.ndarray]: List of frames, where each frame is an image array.
    """
    # Open the video file
    cap = cv2.VideoCapture(path)
    frames = []

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return frames

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


def _convert_frames_to_video(frame_dir: str, output_video: str, fps: float, frame_size: Tuple[int, int]) -> None:
    """
    Convert frames in a directory to a video file.

    Args:
        frame_dir (str): Directory containing frame images.
        output_video (str): Path to save the output video.
        fps (float): Frames per second for the output video.
        frame_size (Tuple[int, int]): Size of the frames as (width, height).
    """
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    frame_count = len(frame_files)

    if frame_count <= 0:
        out.release()
        print("There are no frames to save")
        return
    
    for filename in frame_files:
        img = cv2.imread(filename)
        out.write(img)
    
    out.release()
    print(f"Video saved as {output_video}")


def process_video(processor = None, video_source: str = "0", output_video: Optional[str] = "output.mp4", 
                  batch_size: int = 30, skip_seconds: int = 0) -> None:
    """
    Process a video file or stream, capturing, processing, and displaying frames.

    Args:
        processor (AbstractVideoProcessor): Object responsible for processing frames.
        video_source (str, optional): Video source (default is "0" for webcam).
        output_video (Optional[str], optional): Path to save the output video or None to skip saving.
        batch_size (int, optional): Number of frames to process at once.
        skip_seconds (int, optional): Seconds to skip at the beginning of the video.
    """
    from annotation import AbstractVideoProcessor  # Lazy import

    if processor is not None and not isinstance(processor, AbstractVideoProcessor):
        raise ValueError("The processor must be an instance of AbstractVideoProcessor.")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_skip = int(skip_seconds * fps)

    # Skip the first 'frames_to_skip' frames
    for _ in range(frames_to_skip):
        cap.read()  # Simply read and discard the frames

    frame_queue = queue.Queue(maxsize=100)
    processed_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    
    def signal_handler(signum, frame):
        """Signal handler to initiate shutdown on interrupt."""

        print("Interrupt received, initiating shutdown...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    
    def frame_capture_thread() -> None:
        """Thread to capture frames from the video source."""

        print("Starting frame capture")
        frame_count = frames_to_skip  # Start counting frames from here
        try:
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("No more frames to capture or end of video")
                    break
                frame_queue.put((frame_count, frame))
                frame_count += 1
        except Exception as e:
            print(f"Error in frame capture: {e}")
        finally:
            cap.release()
            frame_queue.put(None)  # Signal end of capture
        print("Frame capture complete")

    def frame_processing_thread() -> None:
        """Thread to process frames from the frame queue."""
        
        print("Starting frame processing")
        frame_batch = []
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=1)
                if item is None:
                    print("No more frames to process")
                    if frame_batch:
                        process_batch(frame_batch)
                    break
                frame_count, frame = item
                frame_batch.append((frame_count, frame))

                if len(frame_batch) == batch_size:
                    process_batch(frame_batch)
                    frame_batch = []
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {e}")

        processed_queue.put(None)  # Signal end of processing
        print("Frame processing complete")

    def process_batch(batch: List[Tuple[int, np.ndarray]]) -> None:
        """
        Process a batch of frames and put results in the processed queue.

        Args:
            batch (List[Tuple[int, np.ndarray]]): List of tuples containing frame count and frame data.
        """
        frames = [frame for _, frame in batch]
        try:
            processed_batch = processor.process(frames, fps)
            for (frame_count, _), processed_frame in zip(batch, processed_batch):
                processed_queue.put((frame_count, processed_frame))
        except Exception as e:
            print(f"Error processing batch: {e}")

    def frame_display_thread(temp_dir: str) -> None:
        """Thread to display processed frames."""

        print("Starting frame display")
        while not stop_event.is_set():
            try:
                item = processed_queue.get(timeout=1)
                if item is None:
                    print("No more frames to display")
                    break
                frame_count, processed_frame = item

                frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, processed_frame)
                
                cv2.imshow('Processed Video', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed, initiating shutdown")
                    stop_event.set()
                    break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error displaying frame: {e}")

        cv2.destroyAllWindows()
        print("Frame display complete")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            threads = [
                threading.Thread(target=frame_capture_thread, name="Capture"),
                threading.Thread(target=frame_processing_thread, name="Processing"),
                threading.Thread(target=frame_display_thread, args=(temp_dir,), name="Display")
            ]

            for thread in threads:
                thread.start()

            # Wait for user to press 'q'
            while any(thread.is_alive() for thread in threads):
                if stop_event.is_set():
                    print("Stopping threads...")
                    break
                time.sleep(0.1)

            stop_event.set()  # Ensure all threads know to stop

            for thread in threads:
                thread.join(timeout=10)  # Give each thread 10 seconds to join
                if thread.is_alive():
                    print(f"Thread {thread.name} did not terminate gracefully")

            # Ensure all queues are empty
            while not frame_queue.empty():
                frame_queue.get()
            while not processed_queue.empty():
                processed_queue.get()

            print("All threads have completed.")
            # Only convert to video if output_video is not None
            if output_video is not None:
                print("Converting frames to video...")
                _convert_frames_to_video(temp_dir, output_video, fps, (width, height))

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()

        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Video processing completed. Program will now exit.")
    os._exit(0)  # Force exit the program


def save_video(out_frames: List[np.ndarray], out_vpath: str, fps: float = 30.0) -> None:
    """
    Save frames as a video file.

    Args:
        out_frames (List[np.ndarray]): List of frames to be saved as video.
        out_vpath (str): Output video file path.
        fps (float, optional): Frames per second for the output video (default is 30.0).
    """

    # Check if any frames to save
    if len(out_frames) == 0:
        print("Error: No frames to save.")
        return

    # Get the shape of the first frame
    height, width, _ = out_frames[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(out_vpath, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in out_frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
