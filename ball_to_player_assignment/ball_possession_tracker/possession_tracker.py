class PossessionTracker():
    def __init__(self, club1, club2):
        self.possession_dict =  {-1:0, club1.name:0, club2.name:0}
        self.club1_name = club1.name
        self.club2_name = club2.name
        self.possession = []
        self.sum = 0


    def add_possession(self, club_name):
        self.possession_dict[club_name]+=1
        self.sum+=1
        self.possession.append({-1: self.possession_dict[-1]/self.sum, 
                                0: self.possession_dict[self.club1_name]/self.sum, 
                                1: self.possession_dict[self.club2_name]/self.sum})