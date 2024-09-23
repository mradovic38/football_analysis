from sklearn.cluster import KMeans
from .club import Club
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2

class ClubAssigner:
    def __init__(self, club1: Club, club2: Club):
        self.club1 = club1
        self.club2 = club2
        self.model = ClubAssignerModel(self.club1, self.club2)
        self.club_colors = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color
        }
        self.kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
        self.scaler = StandardScaler()

    def clustering(self, img):
        # Convert to LAB color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Reshape image to 2D array
        img_reshape = lab_img.reshape(-1, 3)
        
        # Scale the data
        img_scaled = self.scaler.fit_transform(img_reshape)
        
        # K-Means clustering
        self.kmeans.fit(img_scaled)
        
        # Get Cluster Labels
        labels = self.kmeans.labels_
        
        # Reshape the labels into the image shape
        cluster_img = labels.reshape(img.shape[0], img.shape[1])
        
        # Most frequent cluster of the corners should be the background cluster
        corners = [cluster_img[0, 0], cluster_img[0, -1], cluster_img[-1, 0], cluster_img[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)
        
        # The other clusters are player clusters
        player_clusters = [i for i in range(3) if i != bg_cluster]
        
        # Get the colors of the player clusters
        jersey_colors = [self.kmeans.cluster_centers_[i] for i in player_clusters]
        
        # Convert back to RGB
        jersey_colors_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2RGB)[0][0] for color in jersey_colors]
        
        return jersey_colors_rgb

    def get_jersey_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_top = img[0:img.shape[0]//2, :]
        jersey_colors = self.clustering(img_top)
        return jersey_colors

    def get_player_club(self, frame, bbox, player_id):
        colors = self.get_jersey_color(frame, bbox)
        predictions = [self.model.predict(color) for color in colors]
        team_ids = [pred % 2 for pred in predictions]  # % 2 to assign goalkeepers to players team
        
        # Choose the most common team prediction
        team_id = max(set(team_ids), key=team_ids.count)
        
        return list(self.club_colors.keys())[team_id], predictions[team_ids.index(team_id)]

    def assign_clubs(self, frame, tracks):
        for track_type in ['goalkeeper', 'player']:
            for player_id, track in tracks[track_type].items():
                bbox = track['bbox']
                club, pred = self.get_player_club(frame, bbox, player_id)
                
                tracks[track_type][player_id]['club'] = club
                tracks[track_type][player_id]['club_color'] = self.club_colors[club]
        
        return tracks

class ClubAssignerModel:
    def __init__(self, club1: Club, club2: Club):
        self.centroids = np.array([club1.player_jersey_color, club2.player_jersey_color,
                                   club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])
        self.centroids_lab = np.array([cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2LAB)[0][0] for color in self.centroids])

    def predict(self, data):
        # Ensure data is in the correct shape (3,) for RGB values
        if data.shape != (3,):
            raise ValueError(f"Expected color data shape (3,), got {data.shape}")
        
        # Reshape data to (1, 1, 3) for cv2.cvtColor
        data_reshaped = data.reshape(1, 1, 3)
        
        # Convert to LAB color space
        data_lab = cv2.cvtColor(np.uint8(data_reshaped), cv2.COLOR_RGB2LAB)[0][0]
        
        # Calculate distances
        distances = np.linalg.norm(data_lab - self.centroids_lab, axis=1)
        
        return np.argmin(distances)