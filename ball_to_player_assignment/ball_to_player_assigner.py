from utils import get_bbox_center, point_distance
from .ball_possession_tracker import PossessionTracker

class BallToPlayerAssigner:
    def __init__(self, club1_name, club2_name, max_ball_distance=70):
        self.max_ball_distance = max_ball_distance
        self.possession_tracker = PossessionTracker(club1_name, club2_name)
    
    def assign(self, tracks):

        player_w_ball = -1

        if 'ball' not in tracks or not tracks['ball']:
            self.possession_tracker.add_possession(-1)
            return player_w_ball

        players = {**tracks['player'], **tracks['goalkeeper']}
        first_key = next(iter(tracks['ball']))  # Get the first key in tracks['ball']
        ball_bbox = tracks['ball'][first_key]['bbox']  # Access the 'bbox' of the first key

        ball_pos = get_bbox_center(ball_bbox)

        min_dis = self.max_ball_distance

       

        for player_id, player in players.items():
            player_bbox = player['bbox']

            l_dis = point_distance((player_bbox[0], player_bbox[3]), ball_pos)
            r_dis = point_distance((player_bbox[2], player_bbox[3]), ball_pos)

            dis = min(l_dis, r_dis)

            if dis <= min_dis:
                min_dis = dis
                player_w_ball = player_id

        if player_w_ball!=-1 and 'club' in players[player_w_ball]:
            self.possession_tracker.add_possession(players[player_w_ball]['club'], players[player_w_ball]['club_color'])
        else:
            self.possession_tracker.add_possession(-1)

        return player_w_ball

    def get_ball_possessions(self):
        return self.possession_tracker.possession