from utils import point_distance, get_bbox_center
from .possession_tracking import PossessionTracker
from club_assignment import Club

from typing import Dict, Tuple, Any

class BallToPlayerAssigner:
    """Assigns the ball to a player if it fits the criteria"""

    def __init__(self, 
                 club1: Club, 
                 club2: Club, 
                 max_ball_distance: float = 10.0, 
                 grace_period: float = 4.0, 
                 ball_grace_period: float = 2.0, 
                 fps: int = 30, 
                 max_ball_speed: float = 250.0, 
                 speed_check_frames: int = 5, 
                 penalty_point_distance: float = 15.0) -> None:
        """
        Initializes the BallToPlayerAssigner with necessary parameters.

        Args:
            club1 (Club): The Club object of the first club.
            club2 (Club): The Club object of the second club.
            max_ball_distance (float): The maximum distance to consider a player as being able to possess the ball.
            grace_period (float): The time in seconds a player retains possession after losing the ball.
            ball_grace_period (float): The time in seconds to allow a player to retain possession after the ball detection is lost.
            fps (int): Frames per second for the video feed.
            max_ball_speed (float): The maximum allowed ball movement in pixels between frames.
            speed_check_frames (int): The number of frames to check for ball movement.
            penalty_point_distance (float): The distance from the penalty point within which the ball is considered invalid.
        """
        self.max_ball_distance = max_ball_distance
        self.grace_period_frames = int(grace_period * fps)
        self.ball_grace_period_frames = int(ball_grace_period * fps)
        self.max_ball_speed = max_ball_speed
        self.speed_check_frames = speed_check_frames
        self.possession_tracker = PossessionTracker(club1, club2)
        self.last_possession_frame = None
        self.last_player_w_ball = None
        self.last_possessing_team = -1
        self.ball_exists = False
        self.ball_lost_frame = None
        self.ball_history = []
        self.penalty_point_distance = penalty_point_distance

    def is_ball_movement_valid(self, ball_pos: Tuple[float, float], current_frame: int) -> bool:
        """
        Checks if the ball's movement is valid based on its previous position.

        Args:
            ball_pos (Tuple[float, float]): The current position of the ball (x, y).
            current_frame (int): The current frame number.

        Returns:
            bool: True if the ball movement is valid, False otherwise.
        """
        if not self.ball_history:
            return True  # No history, so movement is valid
        
        last_ball_pos, last_frame = self.ball_history[-1]

        if current_frame - last_frame <= self.speed_check_frames:
            distance_moved = point_distance(ball_pos, last_ball_pos)

            if distance_moved > self.max_ball_speed:
                return False  # Movement is too large, likely invalid

        return True

    def assign(self, tracks: Dict[str, Any], current_frame: int, penalty_point_1_pos: Tuple[float, float], penalty_point_2_pos: Tuple[float, float]) -> Tuple[Dict[str, Any], int]:
        """
        Assigns the ball to the nearest player based on various criteria.

        Args:
            tracks (Dict[str, Any]): A dictionary containing tracked objects.
            current_frame (int): The current frame number.
            penalty_point_1_pos (Tuple[float, float]): The position of the first penalty point (x, y).
            penalty_point_2_pos (Tuple[float, float]): The position of the second penalty point (x, y).

        Returns:
            Tuple[Dict[str, Any], int]: Updated tracks and the ID of the player with the ball.
        """
        # Copy the tracks to avoid mutating the original data
        tracks = tracks.copy()
        player_w_ball = -1
        valid_ball_tracks = []  
        best_ball_key = None
        to_delete = []

        if 'ball' in tracks and tracks['ball']:
            self.ball_exists = False
            self.ball_lost_frame = current_frame
            
            for ball_key, ball_data in tracks['ball'].items():
                ball_pos = ball_data['projection']  
                ball_bbox_center = get_bbox_center(ball_data['bbox'])  

                is_near_penalty_point = False
                if penalty_point_1_pos is not None:
                    if point_distance(ball_bbox_center, penalty_point_1_pos) < self.penalty_point_distance:
                        is_near_penalty_point = True
                if penalty_point_2_pos is not None:
                    if point_distance(ball_bbox_center, penalty_point_2_pos) < self.penalty_point_distance:
                        is_near_penalty_point = True

                if not is_near_penalty_point and self.is_ball_movement_valid(ball_pos, current_frame):
                    valid_ball_tracks.append((ball_key, ball_pos))
                else:
                    to_delete.append(ball_key)

        if valid_ball_tracks:
            self.ball_exists = True
            min_dis = self.max_ball_distance
            best_ball_key, best_ball_pos = None, None
            players = {**tracks.get('player', {}), **tracks.get('goalkeeper', {})}

            for ball_key, ball_pos in valid_ball_tracks:
                for player_id, player in players.items():
                    player_pos = player['projection']
                    dis = point_distance(ball_pos, player_pos)

                    if dis <= min_dis:
                        min_dis = dis
                        player_w_ball = player_id
                        best_ball_key, best_ball_pos = ball_key, ball_pos

            if best_ball_key is not None:
                self.ball_history.append((best_ball_pos, current_frame))
                if len(self.ball_history) > self.speed_check_frames:
                    self.ball_history.pop(0)

                if player_w_ball != -1 and 'club' in players[player_w_ball]:
                    self.possession_tracker.add_possession(players[player_w_ball]['club'])
                    self.last_player_w_ball = player_w_ball
                    self.last_possession_frame = current_frame  
                    self.last_possessing_team = players[player_w_ball]['club']

                    if player_w_ball in tracks['player']:
                        tracks['player'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['player'][player_w_ball]['club'])
                    elif player_w_ball in tracks['goalkeeper']:
                        tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['goalkeeper'][player_w_ball]['club'])
                    else:
                        self.possession_tracker.add_possession(self.last_possessing_team)

                else:
                    if self.last_player_w_ball is not None:
                        elapsed_frames = current_frame - self.last_possession_frame
                        if elapsed_frames <= self.grace_period_frames:
                            player_w_ball = self.last_player_w_ball
                            self.possession_tracker.add_possession(self.last_possessing_team)

                            if player_w_ball in tracks['player']:
                                tracks['player'][player_w_ball]['has_ball'] = True
                                self.possession_tracker.add_possession(tracks['player'][player_w_ball]['club'])
                            elif player_w_ball in tracks['goalkeeper']:
                                tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                                self.possession_tracker.add_possession(tracks['goalkeeper'][player_w_ball]['club'])
                            else:
                                self.possession_tracker.add_possession(self.last_possessing_team)
                        else:
                            self.possession_tracker.add_possession(-1)
                            self.last_player_w_ball = None  
                    else:
                        self.possession_tracker.add_possession(-1)

        else:
            if self.last_player_w_ball is not None:
                elapsed_frames_since_ball_seen = current_frame - self.ball_lost_frame if self.ball_lost_frame else float('inf')

                if elapsed_frames_since_ball_seen <= self.ball_grace_period_frames:
                    player_w_ball = self.last_player_w_ball
                    self.possession_tracker.add_possession(self.last_possessing_team)

                    if player_w_ball in tracks['player']:
                        tracks['player'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['player'][player_w_ball]['club'])
                    elif player_w_ball in tracks['goalkeeper']:
                        tracks['goalkeeper'][player_w_ball]['has_ball'] = True
                        self.possession_tracker.add_possession(tracks['goalkeeper'][player_w_ball]['club'])
                    else:
                        self.possession_tracker.add_possession(self.last_possessing_team)
                else:
                    self.possession_tracker.add_possession(-1)
                    self.last_player_w_ball = None
            else:
                self.possession_tracker.add_possession(-1)
        
        
        for bid in to_delete:
            del tracks['ball'][bid]

        ball_tracks_cpy = tracks['ball'].copy()

        if best_ball_key:
            for bid in ball_tracks_cpy.keys():
                if bid != best_ball_key:
                    del tracks['ball'][bid]


        return tracks, player_w_ball

    def get_ball_possessions(self) -> Any:
        """
        Returns the current ball possessions tracked by the possession tracker.

        Returns:
            Any: The current ball possessions.
        """
        return self.possession_tracker.possession
