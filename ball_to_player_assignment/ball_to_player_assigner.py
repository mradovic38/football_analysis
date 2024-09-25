from utils import get_bbox_center, point_distance, get_feet_pos
from .ball_possession_tracker import PossessionTracker

class BallToPlayerAssigner:
    def __init__(self, club1, club2, max_ball_distance=50, grace_period=4, ball_grace_period=3, fps=30, max_ball_speed=300, speed_check_frames=5):
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

    def assign(self, tracks, current_frame):
        # Copy the tracks to avoid mutating the original data
        tracks = tracks.copy()

        # Initialize player with ball to -1 (no possession)
        player_w_ball = -1

        # Check if the ball exists in the tracks
        if 'ball' in tracks and tracks['ball']:
            self.ball_exists = True  # Ball is present
            self.ball_lost_frame = current_frame  # Reset the frame when the ball was last seen
            first_key = next(iter(tracks['ball']))  # Get the first key in tracks['ball']
            ball_bbox = tracks['ball'][first_key]['bbox']  # Access the 'bbox' of the first ball instance
            ball_pos = get_bbox_center(ball_bbox)  # Get the center position of the ball

            # Validate the ball's movement before proceeding
            if not self.is_ball_movement_valid(ball_pos, current_frame):
                # Invalid movement: disregard this ball detection and proceed as if the ball was not found
                self.ball_exists = False
            else:
                # Update ball history with the current valid ball position and frame number
                self.ball_history.append((ball_pos, current_frame))
                # Limit the history size to avoid excessive memory usage
                if len(self.ball_history) > self.speed_check_frames:
                    self.ball_history.pop(0)

            min_dis = self.max_ball_distance  # Initialize the minimum distance to the max distance
            
            # Extract players and goalkeepers from the tracks
            players = {**tracks.get('player', {}), **tracks.get('goalkeeper', {})}

            # Loop through players and check their proximity to the ball
            if self.ball_exists:  # Proceed only if ball movement was valid
                for player_id, player in players.items():
                    player_bbox = player['bbox']
                    player_pos = get_feet_pos(player_bbox)  # Get the player's feet position
                    dis = point_distance(ball_pos, player_pos)  # Calculate distance between ball and player

                    # If player is within max distance, update the player with possession
                    if dis <= min_dis:
                        min_dis = dis
                        player_w_ball = player_id

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
                    elif player_w_ball in tracks['goalkeeper']:
                        tracks['goalkeeper'][player_w_ball]['has_ball'] = True
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
                            elif player_w_ball in tracks['goalkeeper']:
                                tracks['goalkeeper'][player_w_ball]['has_ball'] = True
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
                    elif player_w_ball in tracks['goalkeeper']:
                        tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                    else:
                        self.possession_tracker.add_possession(self.last_possessing_team)
                else:
                    # If the ball is not detected for more than the grace period, reset possession
                    self.possession_tracker.add_possession(-1)
                    self.last_player_w_ball = None
            else:
                # If no previous possession and ball isn't present, mark no possession
                self.possession_tracker.add_possession(-1)

        return tracks, player_w_ball

    def get_ball_possessions(self):
        return self.possession_tracker.possession
