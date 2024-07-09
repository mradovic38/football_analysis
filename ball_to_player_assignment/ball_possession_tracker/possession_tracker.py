class PossessionTracker():
    def __init__(self, club1_name, club2_name):
        self.possession_dict =  {-1:0, club1_name:0, club2_name:0}
        self.club1_name = club1_name
        self.club2_name = club2_name
        self.possession = []
        self.sum = 0
        self.club_colors = {0:(255,255,255), 1:(255,255,255)}


    def add_possession(self, club_id, club_color=(255,255,255)):
        self.possession_dict[club_id]+=1
        self.sum+=1
        self.possession.append({-1: self.possession_dict[-1]/self.sum, 
                                0: self.possession_dict[self.club1_name]/self.sum, 
                                1: self.possession_dict[self.club2_name]/self.sum})
        if club_id != -1:
            self.club_colors[club_id] = club_color