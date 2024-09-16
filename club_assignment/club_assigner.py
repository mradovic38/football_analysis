from sklearn.cluster import KMeans
from .club import Club
import numpy as np

class ClubAssigner:
    def __init__(self, club1: Club, club2: Club):
        self.club1 = club1
        self.club2 = club2
        self.model = ClubAssignerModel(self.club1, self.club2)
        self.club_colors = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color
        }


    
    def clustering(self, img):
        # Reshape image to 2D array
        img_reshape = img.reshape(-1, 3)

        # K-Means clustering
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=3, random_state=42)
        kmeans.fit(img_reshape)

        # Get Cluster Labels
        labels = kmeans.labels_

        # Reshape the labels into the image shape
        cluster_img = labels.reshape(img.shape[0], img.shape[1])

        # Most frequent cluster of the corners should be the background cluster
        corners = [cluster_img[0, 0], cluster_img[0, -1], cluster_img[-1, 0], cluster_img[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)

        # The other cluster is a player cluster
        player_cluster = 1-bg_cluster

        # Get the color of the cluster
        jersey_color = kmeans.cluster_centers_[player_cluster]
        return jersey_color
    
    def get_jersey_color(self, frame, bbox):
        img = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        img_top = img[0 : img.shape[0]//2, :]

        jersey_color = self.clustering(img_top)

        return jersey_color

    # def assign_club(self, frame, detections):
    #     colors = []

    #     for _, detection in detections.items():
    #         bbox = detection['bbox']
    #         color = self.get_jersey_color(frame, bbox)
    #         colors.append(color)

    #     model = ClubAssignerModel(self.club1, self.club2)
    #     model.fit(np.array(colors))

    #     self.model = model

    #     self.club_colors = [self.club1.player_jersey_color, self.club2.player_jersey_color]

    def get_player_club(self, frame, bbox, player_id):
        # if player_id in self.players_club:
        #     return self.players_club[player_id]
        
        color = self.get_jersey_color(frame, bbox)

        pred = self.model.predict(color.reshape(1, -1)) # wrongfully labeled goalkeepers will have pred > 1

        team_id = pred % 2 # % 2 to assign goalkeepers to players team

        #print(team_id)
        return list(self.club_colors.keys())[team_id], pred
    

    def assign_clubs(self, frame, tracks):
        
        to_delete = []
        for player_id, track in tracks['goalkeeper'].items():
            bbox = track['bbox']
            club, pred = self.get_player_club(frame, bbox, player_id)
            
            tracks['goalkeeper'][player_id]['club'] = club
            tracks['goalkeeper'][player_id]['club_color'] = self.club_colors[club]

            #if pred <= 1:
                #tracks['player'][cur_frame][player_id] = tracks['goalkeeper'][cur_frame][player_id]
                #to_delete.append(player_id) 
        # for i in to_delete:            
        #     tracks['goalkeeper'].pop(i)

        to_delete = []
        for player_id, track in tracks['player'].items():
            bbox = track['bbox']
            club, pred = self.get_player_club(frame, bbox, player_id)
            
            tracks['player'][player_id]['club'] = club
            tracks['player'][player_id]['club_color'] = self.club_colors[club]

            if pred > 1:
                tracks['goalkeeper'][player_id] = tracks['player'][player_id]
                to_delete.append(player_id) 

        for i in to_delete:            
            tracks['player'].pop(i)

        return tracks

                

class ClubAssignerModel:


    def __init__(self, club1: Club, club2: Club):
        self.centroids = np.array([club1.player_jersey_color, club2.player_jersey_color, 
                                      club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])
    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)[0]
