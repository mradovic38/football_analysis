from .abstract_writer import AbstractWriter

import os
import json
import numpy as np
from typing import Any, List


class TracksJsonWriter(AbstractWriter):
    """
    A class to write tracking data to JSON files.

    This class handles writing both object tracks and keypoint tracks to separate JSON files.
    It ensures that existing data can be appended without losing previous entries.
    """

    def __init__(self, save_dir: str = '', object_fname: str = 'object_tracks', 
                 keypoints_fname: str = 'keypoint_tracks') -> None:
        """
        Initializes the TracksJsonWriter.

        Args:
            save_dir (str): Directory to save JSON files.
            object_fname (str): Filename for object tracks (without extension).
            keypoints_fname (str): Filename for keypoint tracks (without extension).
        """
        super().__init__()
        self.save_dir = save_dir
        self.obj_path = os.path.join(self.save_dir, f'{object_fname}.json')
        self.kp_path = os.path.join(self.save_dir, f'{keypoints_fname}.json')

        if os.path.exists(save_dir):
            self._remove_existing_files(files=[self.kp_path, self.obj_path]) 
        else:
            os.makedirs(save_dir)
    
    def get_object_tracks_path(self) -> str:
        """Returns the path for the object tracks JSON file."""
        return self.obj_path
    
    def get_keypoints_tracks_path(self) -> str:
        """Returns the path for the keypoint tracks JSON file."""
        return self.kp_path

    def write(self, filename: str, tracks: Any) -> None:
        """Write tracks to a JSON file.

        If the file already exists, new tracks are appended.

        Args:
            filename (str): The name of the file to save tracks.
            tracks (Any): The tracking data to write to the file.
        """
        # Convert all tracks to a serializable format
        serializable_tracks = self._make_serializable(tracks)

        if os.path.exists(filename):
            # If file exists, load existing data and append new tracks
            with open(filename, 'r') as f:
                existing_data = json.load(f)
            existing_data.append(serializable_tracks)
            data_to_save = existing_data
        else:
            # If file doesn't exist, create a new list with current tracks
            data_to_save = [serializable_tracks]

        # Write the serializable data to the file
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)  # Added indent for better readability

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to a JSON-serializable format.

        Args:
            obj (Any): The object to convert.

        Returns:
            Any: A JSON-serializable representation of the object.
        """
        if isinstance(obj, dict):
            # Ensure both keys and values are serializable
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Convert lists recursively
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            # Convert tuples recursively
            return tuple(self._make_serializable(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            # Convert numpy int to Python int
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            # Convert numpy float to Python float
            return float(obj)
        elif isinstance(obj, (int, float)):
            # No conversion needed for Python-native types
            return obj
        else:
            # Return the object as is if it's not a type we need to convert
            return obj
        
    def _remove_existing_files(self, files: List[str]) -> None:
        """
        Remove files from the filesystem if they exist.

        Args:
            files (list): List of file paths to check and remove.
        """
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
