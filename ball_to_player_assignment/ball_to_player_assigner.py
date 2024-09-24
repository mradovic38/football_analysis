from utils import get_bbox_center, point_distance, get_feet_pos
from .ball_possession_tracker import PossessionTracker

class BallToPlayerAssigner:
    def __init__(self, club1, club2, max_ball_distance=70, grace_period=5, fps=30):
        self.max_ball_distance = max_ball_distance
        self.grace_period_frames = int(grace_period * fps)  # Convert grace period to frames
        self.possession_tracker = PossessionTracker(club1, club2)
        self.last_possession_frame = None  # Track the frame when possession was lost
        self.last_player_w_ball = None     # Track the last player who had the ball
        self.last_possessing_team = -1     # Track the team of the last player with possession
        self.ball_exists = False            # Track if the ball is currently in the scene

    def assign(self, tracks, current_frame):
        # Copy the tracks to avoid mutating the original data
        tracks = tracks.copy()

        # Initialize player with ball to -1 (no possession)
        player_w_ball = -1

        # Check if the ball exists in the tracks
        if 'ball' in tracks and tracks['ball']:
            self.ball_exists = True  # Ball is present
            first_key = next(iter(tracks['ball']))  # Get the first key in tracks['ball']
            ball_bbox = tracks['ball'][first_key]['bbox']  # Access the 'bbox' of the first ball instance
            ball_pos = get_bbox_center(ball_bbox)  # Get the center position of the ball

            min_dis = self.max_ball_distance  # Initialize the minimum distance to the max distance
            
            # Extract players and goalkeepers from the tracks
            players = {**tracks.get('player', {}), **tracks.get('goalkeeper', {})}

            # Loop through players and check their proximity to the ball
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
                else:
                    tracks['goalkeeper'][player_w_ball]['has_ball'] = True
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
                        else:
                            tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                    else:
                        # If grace period is over, remove possession
                        self.possession_tracker.add_possession(-1)
                        self.last_player_w_ball = None  # Reset last player possession
                else:
                    self.possession_tracker.add_possession(-1)

        else:
            # If the ball is not present, check if we had a previous possession
            if self.last_player_w_ball is not None:
                # Maintain possession for the last player until the ball is found again
                player_w_ball = self.last_player_w_ball
                self.possession_tracker.add_possession(self.last_possessing_team)

                # Mark possession on the tracks for the last player
                if player_w_ball in tracks['player']:
                    tracks['player'][player_w_ball]['has_ball'] = True
                else:
                    tracks['goalkeeper'][player_w_ball]['has_ball'] = True
            else:
                self.possession_tracker.add_possession(-1)

        return tracks, player_w_ball

    def get_ball_possessions(self):
        return self.possession_tracker.possession
