import math
from collections import deque

class SpeedEstimator:
    def __init__(self, field_width=528, field_height=352, real_field_length=100, real_field_width=50, smoothing_window=5):
        self.field_width = field_width
        self.field_height = field_height
        self.real_field_length = real_field_length  # 100 meters
        self.real_field_width = real_field_width    # 50 meters
        self.previous_positions = {}
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        
        # Calculate scaling factors
        self.scale_x = real_field_length / field_width
        self.scale_y = real_field_width / field_height
        
        # Maximum realistic speed (km/h)
        self.max_speed = 40

    def calculate_speed(self, tracks, frame_number, fps):
        for track_type in tracks:
            for player_id, track in tracks[track_type].items():
                if 'projection' in track:
                    current_position = track['projection']
                    
                    if player_id in self.previous_positions:
                        prev_position, prev_frame = self.previous_positions[player_id]
                        
                        # Calculate distance in meters
                        distance = self._calculate_distance(prev_position, current_position)
                        
                        # Calculate time difference in seconds
                        time_diff = (frame_number - prev_frame) / fps
                        
                        # Calculate speed in km/h
                        speed = (distance / time_diff) * 3.6 if time_diff > 0 else 0

                        # print(f"Player {player_id}: Distance: {distance:.2f}m, Time: {time_diff:.2f}s, Raw Speed: {speed:.2f}km/h")
                        
                        # Apply maximum speed check
                        speed = min(speed, self.max_speed)
                        
                        # Apply smoothing
                        smoothed_speed = self._smooth_speed(player_id, speed)
                        
                        # Add speed to track
                        tracks[track_type][player_id]['speed'] = smoothed_speed
                    else:
                        # If it's the first time we're seeing this player, set speed to 0
                        tracks[track_type][player_id]['speed'] = 0.0
                        self.speed_history[player_id] = deque([0.0] * self.smoothing_window, maxlen=self.smoothing_window)
                    
                    # Update previous position
                    self.previous_positions[player_id] = (current_position, frame_number)
                else:
                    # If there's no projection, set speed to 0
                    tracks[track_type][player_id]['speed'] = 0.0
        
        return tracks

    def _calculate_distance(self, pos1, pos2):
        # Calculate Euclidean distance in meters
        dx = (pos2[0] - pos1[0]) * self.scale_x
        dy = (pos2[1] - pos1[1]) * self.scale_y
        return math.sqrt(dx**2 + dy**2)

    def _smooth_speed(self, player_id, speed):
        if player_id not in self.speed_history:
            self.speed_history[player_id] = deque([0.0] * self.smoothing_window, maxlen=self.smoothing_window)
        
        self.speed_history[player_id].append(speed)
        return sum(self.speed_history[player_id]) / len(self.speed_history[player_id])

    def reset(self):
        # Reset previous positions and speed history (call this at the start of a new video or when needed)
        self.previous_positions = {}
        self.speed_history = {}