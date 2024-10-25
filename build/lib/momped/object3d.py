import numpy as np
import cv2
from utils import detect_sift_features, find_matching_points
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Object3D:
    def __init__(self, npz_path):
        """
        Initialize Object3D with stored feature points from npz file.
        Args:
            npz_path: Path to the .npz file containing feature points
        """
        # Load the feature data
        self.load_feature_points(npz_path)
        # Store mesh vertices if available in npz file
        self.mesh_vertices = None

    def load_feature_points(self, filepath):
        """
        Load feature points from a .npz file
        Args:
            filepath: Path to the saved feature points (without extension)
        """
        # Load the compressed file
        data = np.load(filepath, allow_pickle=True)
        
        # Load feature points
        loaded_data = data['feature_points']
        self.feature_points = []
        
        for feat_dict in loaded_data:
            # Reconstruct keypoint
            kp_data = feat_dict['keypoint']
            keypoint = cv2.KeyPoint(
                x=float(kp_data['pt'][0]),
                y=float(kp_data['pt'][1]),
                size=float(kp_data['size']),
                angle=float(kp_data['angle']),
                response=float(kp_data['response']),
                octave=int(kp_data['octave']),
                class_id=int(kp_data['class_id'])
            )
            
            # Reconstruct feature dictionary
            feature = {
                'id': feat_dict['id'],
                'descriptor': feat_dict['descriptor'],
                'point3d': feat_dict['point3d'],
                'keypoint': keypoint,
                'response': feat_dict['response'],
                'size': feat_dict['size'],
                'angle': feat_dict['angle'],
                'octave': feat_dict['octave']
            }
            self.feature_points.append(feature)
        
        # Load mesh vertices if available
        if 'mesh_vertices' in data:
            self.mesh_vertices = data['mesh_vertices']

        print(f"Loaded {len(self.feature_points)} features from {filepath}")

    def match_image_points(self, image):
        """
        Match image features with stored 3D points.
        Args:
            image: Input image
        Returns:
            image_points: 2D points in the image
            object_points: Corresponding 3D points in object space
        """
        # Use utility function to detect features
        keypoints, descriptors = detect_sift_features(image)
        
        if keypoints is None:
            return None, None
        
        # Use utility function to find matches
        matches = find_matching_points(self.feature_points, descriptors)
        
        if not matches:
            return None, None
        
        # Extract corresponding points
        image_points = []  # 2D points in image
        object_points = []  # 3D points in object space
        
        for query_idx, matched_feature in matches:
            # Get 2D point from keypoint
            image_points.append(keypoints[query_idx].pt)
            # Get corresponding 3D point
            object_points.append(matched_feature['point3d'])
        
        return np.array(image_points), np.array(object_points)

    def visualize_3d_points(self, current_points=None):
        """
        Visualize stored and current 3D points.
        Args:
            current_points: Optional numpy array of current 3D points to visualize
        """
        # Create new figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot stored points
        stored_points = np.array([f['point3d'] for f in self.feature_points])
        ax.scatter(stored_points[:, 0], 
                  stored_points[:, 1], 
                  stored_points[:, 2],
                  c='green', s=20, 
                  alpha=0.5,
                  label=f'Stored ({len(stored_points)})')

        # Plot current points if provided
        if current_points is not None and len(current_points) > 0:
            ax.scatter(current_points[:, 0], 
                      current_points[:, 1], 
                      current_points[:, 2],
                      c='red', s=50, 
                      label=f'Current ({len(current_points)})')

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

        # Set limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Visualization')
        
        # Add legend
        ax.legend()
        
        # Show the plot
        plt.show()

    def visualize_matches(self, image, matched_points=None):
        """
        Visualize feature matches on the image.
        Args:
            image: Input image
            matched_points: Optional array of 2D points to highlight
        Returns:
            image with visualized features
        """
        img_with_keypoints = image.copy()
        
        # Detect features in current image
        keypoints, _ = detect_sift_features(image)
        
        if keypoints is None:
            return img_with_keypoints
        
        if matched_points is None:
            # Draw all keypoints in green
            cv2.drawKeypoints(image, keypoints, img_with_keypoints, 
                            color=(0,255,0), 
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            # Convert matched points to keypoints format
            matched_kps = [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) 
                          for pt in matched_points]
            
            # Draw matched keypoints in red
            cv2.drawKeypoints(image, matched_kps, img_with_keypoints,
                            color=(0,0,255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Add text information
        info_text = [
            f'Total Features: {len(keypoints)}',
            f'Matched: {len(matched_points) if matched_points is not None else 0}'
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(img_with_keypoints, text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0,0,255), 2)
            y_offset += 25
        
        return img_with_keypoints

# Example usage:
if __name__ == "__main__":
    # Initialize object with stored features
    obj = Object3D("examples/ketchup.npz")
    
    # Load a test image
    image = cv2.imread("examples/ketchup_bottle.jpg")
    
    # Get matching points
    img_pts, obj_pts = obj.match_image_points(image)
    
    if img_pts is not None:
        print(f"Found {len(img_pts)} matching points")
        
        # Visualize 2D matches
        vis_img = obj.visualize_matches(image, img_pts)
        cv2.imshow("Matches", vis_img)
        cv2.waitKey(1)
        
        # Visualize 3D points
        obj.visualize_3d_points(obj_pts)