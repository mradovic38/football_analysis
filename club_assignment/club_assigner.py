import os
from sklearn.cluster import KMeans
from .club import Club
import numpy as np
from sklearn.preprocessing import StandardScaler
import cv2

class ClubAssigner:
    def __init__(self, club1: Club, club2: Club, images_to_save=0, images_save_path=None):
        self.club1 = club1
        self.club2 = club2
        self.model = ClubAssignerModel(self.club1, self.club2)
        self.club_colors = {
            club1.name: club1.player_jersey_color,
            club2.name: club2.player_jersey_color
        }
        self.goalkeeper_colors = {
            club1.name: club1.goalkeeper_jersey_color,
            club2.name: club2.goalkeeper_jersey_color
        }
        self.player_kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
        self.goalkeeper_kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)

        # Saving images for analysis
        self.images_to_save =  images_to_save
        self.output_dir = images_save_path

        if not images_save_path:
            images_to_save = 0

        elif not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.saved_images = len([name for name in os.listdir(self.output_dir) if name.startswith('player')])

    
    def apply_mask(self, image, green_threshold=.05):
        """
        Apply a mask to an image based on green color in HSV space. 
        If the mask covers more than green_treshold of the image, apply the inverse of the mask.

        Parameters:
        image (array): An image to apply mask to.
        green_treshold (float): If non-green color is covering more than green_treshold of image, apply mask

        Returns:
        Masked images.
        """

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])

        # Create the mask
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Count the number of masked pixels
        total_pixels = image.shape[0] * image.shape[1]
        masked_pixels = cv2.countNonZero(cv2.bitwise_not(mask))
        mask_percentage = masked_pixels / total_pixels
        
        if mask_percentage > green_threshold:
            # Apply inverse mask
            return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        else:
            # Apply normal mask
            return image

    def clustering(self, img, is_goalkeeper=False):
        # Reshape image to 2D array
        img_reshape = img.reshape(-1, 3)
        
        # K-Means clustering
        kmeans = self.goalkeeper_kmeans if is_goalkeeper else self.player_kmeans
        kmeans.fit(img_reshape)
        
        # Get Cluster Labels
        labels = kmeans.labels_
        
        # Reshape the labels into the image shape
        cluster_img = labels.reshape(img.shape[0], img.shape[1])

        # Get Jersey Color
        corners = [cluster_img[0, 0], cluster_img[0, -1], cluster_img[-1, 0], cluster_img[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)

        # The other cluster is a player cluster
        player_cluster = 1-bg_cluster

        jersey_color_bgr = kmeans.cluster_centers_[player_cluster]
        
        return (jersey_color_bgr[2], jersey_color_bgr[1], jersey_color_bgr[0])

    def save_player_image(self, img, player_id, is_goalkeeper=False):
        # Use 'goalkeeper' or 'player' prefix based on is_goalkeeper flag
        prefix = 'goalkeeper' if is_goalkeeper else 'player'
        filename = os.path.join(self.output_dir, f"{prefix}_{player_id}.png")
        if os.path.exists(filename):
            return
        cv2.imwrite(filename, img)
        print(f"Saved {prefix} image: {filename}")
        # If 10 images have been saved, set the flag to True
        self.saved_images += 1

    def get_jersey_color(self, frame, bbox, player_id, is_goalkeeper=False):
        # Save player images only if needed
        if self.saved_images <= self.images_to_save:
            img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            img_top = img[0:img.shape[0]//2, :] 
            self.save_player_image(img_top, player_id, is_goalkeeper)  # Pass is_goalkeeper here
    
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_top = img[0:img.shape[0]//2, :]  # Use upper half for jersey detection
        masked_img_top = self.apply_mask(img_top, green_threshold=.08)
        jersey_color = self.clustering(masked_img_top, is_goalkeeper)
        
        return jersey_color

    def get_player_club(self, frame, bbox, player_id, is_goalkeeper=False):
        color = self.get_jersey_color(frame, bbox, player_id, is_goalkeeper)
        pred = self.model.predict(color, is_goalkeeper)
        
        color_dict = self.goalkeeper_colors if is_goalkeeper else self.club_colors
        return list(color_dict.keys())[pred], pred



    def assign_clubs(self, frame, tracks):
        tracks = tracks.copy()

        for track_type in ['goalkeeper', 'player']:
            for player_id, track in tracks[track_type].items():
                bbox = track['bbox']
                is_goalkeeper = (track_type == 'goalkeeper')
                club, _ = self.get_player_club(frame, bbox, player_id, is_goalkeeper)
                
                tracks[track_type][player_id]['club'] = club
                color_dict = self.goalkeeper_colors if is_goalkeeper else self.club_colors
                tracks[track_type][player_id]['club_color'] = color_dict[club]
        
        return tracks

class ClubAssignerModel:
    def __init__(self, club1: Club, club2: Club):
        self.player_centroids = np.array([club1.player_jersey_color, club2.player_jersey_color])
        self.goalkeeper_centroids = np.array([club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])

    def predict(self, extracted_color, is_goalkeeper=False):
        
        if is_goalkeeper:
            centroids = self.goalkeeper_centroids
        else:
            centroids = self.player_centroids

        # Calculate distances
        distances = np.linalg.norm(extracted_color - centroids, axis=1)
        
        return np.argmin(distances)
