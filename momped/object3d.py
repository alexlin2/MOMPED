import numpy as np
import cv2
from utils import detect_sift_features, find_matching_points
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import open3d as o3d

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
                'point3d': feat_dict['point3d'] / 10.0,
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

    def match_image_points(self, image, mask=None):
        """
        Match image features with stored 3D points.
        Args:
            image: Input image
        Returns:
            image_points: 2D points in the image
            object_points: Corresponding 3D points in object space
        """

        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)
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
    
    def estimate3d(self, image_points, depth_image, camera_matrix, dist_coeffs=None):
        """
        Vectorized estimation of 3D points in camera frame from 2D image points using depth image.
        
        Args:
            image_points: Nx2 array of image points (u,v)
            depth_image: HxW depth image as float32 in meters
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (optional)
            
        Returns:
            points3d: Nx3 array of 3D points in camera frame
            valid_indices: Indices of valid points
        """
        # Undistort points if needed
        if dist_coeffs is not None:
            image_points = cv2.undistortPoints(
                image_points.reshape(-1,1,2), 
                camera_matrix, 
                dist_coeffs, 
                P=camera_matrix
            ).reshape(-1,2)

        # Get camera intrinsics
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        cx = camera_matrix[0,2]
        cy = camera_matrix[1,2]

        # Get depth values directly
        xy = np.round(image_points).astype(int)

        Z = depth_image[xy[:, 1], xy[:, 0]]
        depth_mask = Z > 0
        Z = Z[depth_mask]
        filtered_points = image_points[depth_mask]

        # Back-project to 3D (vectorized)
        X = (filtered_points[:, 0] - cx) * Z / fx
        Y = (filtered_points[:, 1] - cy) * Z / fy
        
        return np.column_stack((X, Y, Z)), depth_mask
    

    def visualize_3d_points(self, feature_points=None, current_points=None):
        """
        Visualize stored and current 3D points.
        Args:
            feature_points: Optional list of stored feature points to visualize
            current_points: Optional numpy array of current 3D points to visualize
        """
        # Create new figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot current points if provided
        if current_points is not None and len(current_points) > 0:
            ax.scatter(current_points[:, 0], 
                      current_points[:, 1], 
                      current_points[:, 2],
                      c='red', s=75,
                      label=f'Current ({len(current_points)})')
    
        # Plot stored points
        if feature_points is not None and len(feature_points) > 0:
            stored_points = np.array([f['point3d'] for f in feature_points])
            ax.scatter(stored_points[:, 0], 
                    stored_points[:, 1], 
                    stored_points[:, 2],
                    c='green', s=20, 
                    alpha=0.5,
                    label=f'Stored ({len(stored_points)})')

        all_points = np.vstack([stored_points, current_points]) if feature_points is not None else current_points

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

        # Auto-scale limits
        
        min_val = np.min(all_points)
        max_val = np.max(all_points)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_zlim([min_val, max_val])

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
    
    def estimate_transform(self, obj_points, est_points, ransac_n=5, correspondence_threshold=0.01):
        """
        Estimate rigid transform between object points and estimated points using Open3D's RANSAC.
        
        Args:
            obj_points: Nx3 array of 3D points in object frame
            est_points: Nx3 array of estimated 3D points in camera frame
            ransac_threshold: RANSAC inlier threshold in meters
            
        Returns:
            R: 3x3 rotation matrix from camera to object frame
            t: 3x1 translation vector from camera to object frame
        """
        
        if len(obj_points) != len(est_points) or len(obj_points) < 3:
            return None, None

        # Ensure points are float32
        obj_points = obj_points.astype(np.float32)
        est_points = est_points.astype(np.float32)

        # Convert points to Open3D format
        pcd_obj = o3d.geometry.PointCloud()
        pcd_est = o3d.geometry.PointCloud()
        
        pcd_obj.points = o3d.utility.Vector3dVector(obj_points)
        pcd_est.points = o3d.utility.Vector3dVector(est_points)

        # Create correspondences (assuming points are already matched)
        corres = np.array([[i, i] for i in range(len(obj_points))])

        # Perform RANSAC registration
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source=pcd_obj,
            target=pcd_est,
            corres=o3d.utility.Vector2iVector(corres),
            max_correspondence_distance=correspondence_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=ransac_n,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=10000000,
                confidence=0.99
            )
        )

        if result.transformation is None:
            return None, None
        
        inlier_mask = np.zeros(len(obj_points), dtype=bool)
        correspondence_set = np.asarray(result.correspondence_set)
        inlier_mask[correspondence_set[:, 0]] = True

        # Extract rotation and translation from transformation matrix
        transformation = result.transformation
        R = transformation[:3, :3]
        t = transformation[:3, 3]

        return R, t, inlier_mask

    def compute_transform_error(self, obj_points, est_points, R, t):
        """
        Compute error between transformed object points and estimated points.
        Note: obj_points are source, est_points are destination (matching estimateAffine3D)
        
        Args:
            obj_points: Nx3 array of object points (source)
            est_points: Nx3 array of estimated points (destination)
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            
        Returns:
            errors: Nx1 array of point-wise distances
            rmse: Root mean square error
        """
        # Convert inputs to numpy arrays and ensure correct shape
        R = np.asarray(R)
        t = np.asarray(t).reshape(3)
        obj_points = np.asarray(obj_points)
        est_points = np.asarray(est_points)
        
        # Transform object points
        transformed_points = (R @ obj_points.T).T + t
        
        # Compute distances to estimated points
        errors = np.linalg.norm(est_points - transformed_points, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        return errors, rmse

    def visualize_alignment(self, obj_points, est_points, R, t):
        """
        Visualize alignment between transformed object points and estimated points.
        Note: obj_points are source, est_points are destination (matching estimateAffine3D)
        
        Args:
            obj_points: Nx3 array of object points (source, green)
            est_points: Nx3 array of estimated points (destination, blue)
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Transform object points
        transformed_points = (R @ obj_points.T).T + t

        # Plot transformed object points
        ax.scatter(transformed_points[:, 0], 
                  transformed_points[:, 1], 
                  transformed_points[:, 2],
                  c='green', s=25, alpha=0.5,
                  label='Transformed Object Points')

        # Plot estimated points
        ax.scatter(est_points[:, 0], 
                  est_points[:, 1], 
                  est_points[:, 2],
                  c='red', s=50, alpha=1.0,
                  label='Estimated Points (destination)')

        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])

        # Auto-scale limits
        all_points = np.vstack([obj_points, transformed_points, est_points])
        min_val = np.min(all_points)
        max_val = np.max(all_points)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_zlim([min_val, max_val])

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Alignment')
        
        # Add transformation info
        euler_angles = cv2.RQDecomp3x3(R)[0]
        info_text = f'Rotation (deg): {euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}\n'
        info_text += f'Translation (m): {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}'
        plt.figtext(0.02, 0.98, info_text, fontsize=10, va='top')
        
        ax.legend()
        plt.show()

    def draw_frame_axes(self, image, R, t, camera_matrix, dist_coeffs=None, axis_length=0.1):
        """
        Draw coordinate axes in image using estimated transformation.
        
        Args:
            image: Input image
            transform: 3x4 transformation matrix [R|t]
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (optional)
            axis_length: Length of axes in meters
            
        Returns:
            image_with_axes: Image with drawn coordinate axes
        """
        # Convert R to rotation vector
        rvec = cv2.Rodrigues(R)[0]
        tvec = t

        # Draw axes
        img_axes = cv2.drawFrameAxes(
            image.copy(), 
            camera_matrix, 
            dist_coeffs if dist_coeffs is not None else np.zeros(4),
            rvec, 
            tvec, 
            axis_length,
            2  # Line thickness
        )

        # Add text labels for pose
        # Convert rotation vector to euler angles (in degrees)
        euler_angles = cv2.RQDecomp3x3(R)[0]
        pose_text = [
            f"Rotation (deg): {euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}",
            f"Translation (m): {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}"
        ]

        y_offset = 30
        for text in pose_text:
            cv2.putText(
                img_axes,
                text,
                (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30

        return img_axes
    

# Example usage:
if __name__ == "__main__":
    # Initialize object with stored features
    obj = Object3D("examples/yellow_mustard.npz")

    # Load a test image
    image = cv2.imread("mustard0/rgb/1581120424100262102.png")
    depth_image = cv2.imread("mustard0/depth/1581120424100262102.png", cv2.IMREAD_UNCHANGED)

    # Define the camera matrix
    camera_matrix = np.array([
        [3.195820007324218750e+02, 0.000000000000000000e+00, 3.202149847676955687e+02],
        [0.000000000000000000e+00, 4.171186828613281250e+02, 2.443486680871046701e+02],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ], dtype=np.float32)
    mask = cv2.imread("mustard0/masks/1581120424100262102.png", cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Convert depth image to float32 if necessary
    if depth_image.dtype != np.float32:
        print("Converting depth image to float32")
        depth_image = depth_image.astype(np.float32) / 1000.0  # Assuming depth is in millimeters
    
    # Get matching points
    img_pts, obj_pts = obj.match_image_points(image, mask)
    
    if img_pts is not None:
        print(f"Found {len(img_pts)} matching points")
        
        # Visualize 2D matches
        vis_img = obj.visualize_matches(image, img_pts)
        cv2.imshow("Matches", vis_img)
        cv2.waitKey(1)

        # Estimate 3D points
        real_pts, valid_indices = obj.estimate3d(img_pts, depth_image, camera_matrix, None)

        # Visualize 3D points
        obj_pts = obj_pts[valid_indices]
        obj.visualize_3d_points(obj.feature_points, obj_pts)
        obj.visualize_3d_points(current_points=real_pts)

        R, t, inliers = obj.estimate_transform(
        obj_points=obj_pts,
        est_points=real_pts,
        correspondence_threshold=0.01  # 1cm threshold
        )

        if R is not None:
            # Compute alignment error
            errors, rmse = obj.compute_transform_error(
                obj_points=obj_pts[inliers],
                est_points=real_pts[inliers],
                R=R,
                t=t
            )
            print(f"RMSE: {rmse*1000:.6f}mm")
            print(t)

            img_axes = obj.draw_frame_axes(
            image=image,
            R=R,
            t=t,
            camera_matrix=camera_matrix,
            dist_coeffs=None,
            axis_length=0.1  # 10cm axes
            )

            cv2.imshow("Pose", img_axes)
            cv2.waitKey(0)

            stored_points = np.array([f['point3d'] for f in obj.feature_points])
            # Visualize alignment
            obj.visualize_alignment(
                obj_points=stored_points,
                est_points=real_pts[inliers],
                R=R,
                t=t
            )

