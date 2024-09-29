from utils import point_distance, get_bbox_center
from .possession_tracking import PossessionTracker

class BallToPlayerAssigner:
    def __init__(self, club1, club2, max_ball_distance=70, grace_period=4, ball_grace_period=2, fps=30, max_ball_speed=200, speed_check_frames=5, penalty_point_distance=15):
        self.max_ball_distance = max_ball_distance
        self.grace_period_frames = int(grace_period * fps)  # Convert player possession grace period to frames
        self.ball_grace_period_frames = int(ball_grace_period * fps)  # Convert ball detection grace period to frames
        self.max_ball_speed = max_ball_speed  # Maximum allowed ball movement in pixels
        self.speed_check_frames = speed_check_frames  # Number of frames over which ball movement is checked
        self.possession_tracker = PossessionTracker(club1, club2)
        self.last_possession_frame = None  # Track the frame when possession was lost
        self.last_player_w_ball = None     # Track the last player who had the ball
        self.last_possessing_team = -1     # Track the team of the last player with possession
        self.ball_exists = False           # Track if the ball is currently in the scene
        self.ball_lost_frame = None        # Track the frame when the ball was last detected
        self.ball_history = []             # Store the ball's position and frame number history
        self.penalty_point_distance = penalty_point_distance

    def is_ball_movement_valid(self, ball_pos, current_frame):
        """
        Checks if the ball's movement between the last tracked frame and the current frame is valid.
        The movement is considered valid if the ball hasn't moved more than the allowed distance within the
        specified number of frames.
        """
        if not self.ball_history:
            return True  # No history, so movement is valid
        
        last_ball_pos, last_frame = self.ball_history[-1]

        # Only check movement if it's within the last `speed_check_frames`
        if current_frame - last_frame <= self.speed_check_frames:
            # Calculate the distance the ball has moved since the last frame
            distance_moved = point_distance(ball_pos, last_ball_pos)

            # Check if the ball movement is within the allowed speed
            if distance_moved > self.max_ball_speed:
                return False  # Movement is too large, likely invalid

        return True

    def assign(self, tracks, current_frame, penalty_point_1_pos, penalty_point_2_pos):
        # Copy the tracks to avoid mutating the original data
        tracks = tracks.copy()

        # Initialize player with ball to -1 (no possession)
        player_w_ball = -1

        valid_ball_tracks = []  # Store valid ball tracks
        best_ball_key = None

        # Check if the ball exists in the tracks
        if 'ball' in tracks and tracks['ball']:
            self.ball_exists = False  # Assume ball doesn't exist until a valid one is found
            self.ball_lost_frame = current_frame  # Reset the frame when the ball was last seen
            
            

            # Loop through all ball tracks
            for ball_key, ball_data in tracks['ball'].items():
                ball_pos = ball_data['projection']  # Access the 'projection' of the ball instance
                ball_bbox_center = get_bbox_center(ball_data['bbox'])  # Get the center of the bounding box

                

                # Filter out the ball if it's too close to any penalty point
                is_near_penalty_point = False
                if penalty_point_1_pos is not None:
                    if point_distance(ball_bbox_center, penalty_point_1_pos) < self.penalty_point_distance:
                        is_near_penalty_point = True
                if penalty_point_2_pos is not None:
                    if point_distance(ball_bbox_center, penalty_point_2_pos) < self.penalty_point_distance:
                        is_near_penalty_point = True

                # Validate the ball's movement and check its distance from the penalty points
                if not is_near_penalty_point and self.is_ball_movement_valid(ball_pos, current_frame):
                    # Add this ball track as valid
                    valid_ball_tracks.append((ball_key, ball_pos))

        # Choose the most suitable ball based on some criteria, like closest to a player
        if valid_ball_tracks:
            self.ball_exists = True
            # If there are multiple valid ball tracks, choose the one with the shortest distance to a player
            min_dis = self.max_ball_distance
            best_ball_key, best_ball_pos = None, None

            players = {**tracks.get('player', {}), **tracks.get('goalkeeper', {})}

            # Loop through each valid ball track
            for ball_key, ball_pos in valid_ball_tracks:
                # Loop through players to find the closest one to the ball
                for player_id, player in players.items():
                    player_pos = player['projection']
                    dis = point_distance(ball_pos, player_pos)  # Calculate distance between ball and player

                    # If player is within max distance, update the best ball
                    if dis <= min_dis:
                        min_dis = dis
                        player_w_ball = player_id
                        best_ball_key, best_ball_pos = ball_key, ball_pos

            # Update ball history with the current valid ball position and frame number
            if best_ball_key is not None:
                self.ball_history.append((best_ball_pos, current_frame))
                # Limit the history size to avoid excessive memory usage
                if len(self.ball_history) > self.speed_check_frames:
                    self.ball_history.pop(0)

                # If a player is within the defined distance and possession changes
                if player_w_ball != -1 and 'club' in players[player_w_ball]:
                    # Assign ball possession to the closest player
                    self.possession_tracker.add_possession(players[player_w_ball]['club'])
                    self.last_player_w_ball = player_w_ball
                    self.last_possession_frame = current_frame  # Reset the possession frame count
                    self.last_possessing_team = players[player_w_ball]['club']

                    # Mark possession on the tracks
                    if player_w_ball in tracks['player']:
                        tracks['player'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['player'][player_w_ball]['club'])
                    elif player_w_ball in tracks['goalkeeper']:
                        tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['goalkeeper'][player_w_ball]['club'])
                    else:
                        self.possession_tracker.add_possession(self.last_possessing_team)

                else:
                    # Check if the last player is still in possession based on the grace period in frames
                    if self.last_player_w_ball is not None:
                        elapsed_frames = current_frame - self.last_possession_frame
                        if elapsed_frames <= self.grace_period_frames:
                            # Retain possession for the last player
                            player_w_ball = self.last_player_w_ball
                            self.possession_tracker.add_possession(self.last_possessing_team)

                            # Mark possession on the tracks for the last player
                            if player_w_ball in tracks['player']:
                                tracks['player'][player_w_ball]['has_ball'] = True
                                self.possession_tracker.add_possession(tracks['player'][player_w_ball]['club'])
                            elif player_w_ball in tracks['goalkeeper']:
                                tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                                self.possession_tracker.add_possession(tracks['goalkeeper'][player_w_ball]['club'])
                            else:
                                self.possession_tracker.add_possession(self.last_possessing_team)
                        else:
                            # If grace period is over, remove possession
                            self.possession_tracker.add_possession(-1)
                            self.last_player_w_ball = None  # Reset last player possession
                    else:
                        self.possession_tracker.add_possession(-1)

        else:
            # Handle the case when the ball is not present
            if self.last_player_w_ball is not None:
                elapsed_frames_since_ball_seen = current_frame - self.ball_lost_frame if self.ball_lost_frame else float('inf')

                # If the ball hasn't been seen for less than the ball grace period, retain possession
                if elapsed_frames_since_ball_seen <= self.ball_grace_period_frames:
                    player_w_ball = self.last_player_w_ball
                    self.possession_tracker.add_possession(self.last_possessing_team)

                    # Mark possession on the tracks for the last player
                    if player_w_ball in tracks['player']:
                        tracks['player'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['player'][player_w_ball]['club'])
                    elif player_w_ball in tracks['goalkeeper']:
                        tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['goalkeeper'][player_w_ball]['club'])
                    else:
                        self.possession_tracker.add_possession(self.last_possessing_team)
                else:
                    # If the ball is not detected for more than the grace period, reset possession
                    self.possession_tracker.add_possession(-1)
                    self.last_player_w_ball = None
            else:
                # If no previous possession and ball isn't present, mark no possession
                self.possession_tracker.add_possession(-1)
        
        ball_tracks_cpy = tracks['ball'].copy()

        for bid in ball_tracks_cpy.keys():
            if best_ball_key == None or bid != best_ball_key:
                del tracks['ball'][bid]

        return tracks, player_w_ball

    def get_ball_possessions(self):
        return self.possession_tracker.possession
