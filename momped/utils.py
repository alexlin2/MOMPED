import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.spatial.transform import Rotation
from scipy import stats

class FeatureManager:
    def __init__(self):
        self.feature_points = []
        self.stored_3d_points = []
        self.total_features = 0

    def add_features(self, keypoints, descriptors, points3d):
        """Add new features to storage."""
        new_features = []
        for kp, desc, point3d in zip(keypoints, descriptors, points3d):
            feature_info = {
                'id': self.total_features,
                'descriptor': desc,
                'point3d': point3d,
                'keypoint': kp,
                'response': kp.response,
                'size': kp.size,
                'angle': kp.angle,
                'octave': kp.octave
            }
            new_features.append(feature_info)
            self.total_features += 1
        
        self.feature_points.extend(new_features)
        self.stored_3d_points.extend(points3d)
        
        return len(new_features)

    def decluster_points(self, voxel_size=0.05):
        """Decluster points using a voxel grid approach."""
        if not self.stored_3d_points or not self.feature_points:
            print("No features to decluster")
            return

        points = np.array(self.stored_3d_points)
        voxel_indices = np.floor(points / voxel_size).astype(int)
        voxel_dict = {}
        
        for idx, (point, feature) in enumerate(zip(points, self.feature_points)):
            voxel_idx = tuple(voxel_indices[idx])
            if voxel_idx not in voxel_dict:
                voxel_dict[voxel_idx] = []
            voxel_dict[voxel_idx].append((point, feature))

        new_points = []
        new_features = []
        
        for voxel_points in voxel_dict.values():
            voxel_points.sort(key=lambda x: x[1]['response'], reverse=True)
            best_point, best_feature = voxel_points[0]
            new_points.append(best_point)
            new_features.append(best_feature)

        self.stored_3d_points = new_points
        self.feature_points = new_features

    def save_features(self, filepath):
        """Save feature points to a .npz file."""
        save_data = []
        for feat in self.feature_points:
            kp = feat['keypoint']
            keypoint_data = {
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            }
            
            save_feat = {
                'id': feat['id'],
                'descriptor': feat['descriptor'],
                'point3d': feat['point3d'],
                'keypoint': keypoint_data,
                'response': feat['response'],
                'size': feat['size'],
                'angle': feat['angle'],
                'octave': feat['octave']
            }
            save_data.append(save_feat)
        
        np.savez_compressed(
            filepath,
            feature_points=save_data,
            stored_3d_points=np.array(self.stored_3d_points),
            total_features=self.total_features
        )

    def load_features(self, filepath):
        """Load feature points from a .npz file."""
        data = np.load(f"{filepath}.npz", allow_pickle=True)
        loaded_data = data['feature_points']
        self.feature_points = []
        
        for feat_dict in loaded_data:
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
        
        self.stored_3d_points = data['stored_3d_points'].tolist()
        self.total_features = int(data['total_features'])

class Visualization:
    @staticmethod
    def draw_features(image, keypoints, matched_indices=None):
        """Draw features on image with optional matching information."""
        img_with_keypoints = image.copy()
        
        if matched_indices is None:
            cv2.drawKeypoints(image, keypoints, img_with_keypoints, 
                            color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            unmatched_kp = [kp for i, kp in enumerate(keypoints) if i not in matched_indices]
            cv2.drawKeypoints(image, unmatched_kp, img_with_keypoints, 
                            color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            matched_kp = [kp for i, kp in enumerate(keypoints) if i in matched_indices]
            cv2.drawKeypoints(img_with_keypoints, matched_kp, img_with_keypoints, 
                            color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        total_features = len(keypoints)
        matched_count = len(matched_indices) if matched_indices is not None else 0
        
        info_text = [
            f'Total Features: {total_features}',
            f'Matched: {matched_count}',
            f'Match Rate: {(matched_count/total_features*100):.1f}%' if total_features > 0 else '0%'
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(img_with_keypoints, text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y_offset += 25
        
        return img_with_keypoints

    @staticmethod
    def plot_3d_points(verts=None, stored_points=None, current_points=None):
        """Plot 3D points with different categories."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        if verts is not None:
            stride = max(1, len(verts) // 1000)
            ax.scatter(verts[::stride, 0], 
                      verts[::stride, 1], 
                      verts[::stride, 2], 
                      c='gray', alpha=0.2, s=1, label='Mesh')

        if stored_points is not None and len(stored_points) > 0:
            stored_points = np.array(stored_points)
            ax.scatter(stored_points[:, 0], 
                      stored_points[:, 1], 
                      stored_points[:, 2],
                      c='green', s=20, 
                      alpha=0.5,
                      label=f'Stored ({len(stored_points)})')

        if current_points is not None and len(current_points) > 0:
            ax.scatter(current_points[:, 0], 
                      current_points[:, 1], 
                      current_points[:, 2],
                      c='red', s=50, 
                      label=f'Current ({len(current_points)})')

        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Visualization')
        ax.legend()
        
        plt.show()

class ModelUtils:
    @staticmethod
    def get_rotation_matrix(elevation, azimuth, yaw, device):
        """Calculate combined rotation matrix."""
        elevation_rad = torch.deg2rad(torch.tensor(elevation))
        azimuth_rad = torch.deg2rad(torch.tensor(azimuth))
        yaw_rad = torch.deg2rad(torch.tensor(yaw))

        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(elevation_rad), -torch.sin(elevation_rad)],
            [0, torch.sin(elevation_rad), torch.cos(elevation_rad)]
        ], device=device, dtype=torch.float32)

        Ry = torch.tensor([
            [torch.cos(azimuth_rad), 0, torch.sin(azimuth_rad)],
            [0, 1, 0],
            [-torch.sin(azimuth_rad), 0, torch.cos(azimuth_rad)]
        ], device=device, dtype=torch.float32)

        Rz = torch.tensor([
            [torch.cos(yaw_rad), -torch.sin(yaw_rad), 0],
            [torch.sin(yaw_rad), torch.cos(yaw_rad), 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)

        R = Rz @ Ry @ Rx
        return R.unsqueeze(0)

    @staticmethod
    def auto_scale_and_center(meshes):
        """Auto-scale and center the mesh."""
        verts = meshes.verts_packed()
        center = verts.mean(dim=0)
        scale = verts.abs().max().item()
        
        meshes.offset_verts_(-center)
        meshes.scale_verts_((1.0 / scale))
        return meshes

def detect_sift_features(image, percentile=50, filter_size=3.0):
    """
    Detect SIFT features with balanced parameters.
    Args:
        image: Input image
        percentile: Percentile threshold for response strength
        filter_size: Minimum size threshold for keypoints
    Returns:
        keypoints: List of keypoints
        descriptors: Numpy array of descriptors
    """
    # Convert current image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector with balanced parameters
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        print("No features detected")
        return None, None
    
    # Filter based on response strength - keep top 20%
    min_response = np.percentile([kp.response for kp in keypoints], percentile)
    
    filtered_keypoints = []
    filtered_descriptors = []
    
    for kp, desc in zip(keypoints, descriptors):
        # Filter by response
        if kp.response < min_response:
            continue
            
        # Minimal size filtering
        if kp.size < filter_size:  # Minimum size threshold
            continue
            
        filtered_keypoints.append(kp)
        filtered_descriptors.append(desc)
        
    if not filtered_keypoints:
        print("No features passed filtering")
        return None, None
        
    filtered_descriptors = np.array(filtered_descriptors)
    print(f"Detected {len(filtered_keypoints)} filtered SIFT features")
    
    return filtered_keypoints, filtered_descriptors

def find_matching_points(feature_points, query_descriptors, ratio_threshold=0.85):
    """
    Find 3D points corresponding to query feature descriptors using basic FLANN matcher.
    Args:
        query_descriptors: numpy array of SIFT descriptors to match
    Returns:
        matches: list of (query_idx, stored_feature_info) tuples
    """
    matches = []
    
    if feature_points is None or query_descriptors is None:
        return matches

    # Get stored descriptors
    stored_descriptors = np.array([f['descriptor'] for f in feature_points])
    
    if len(stored_descriptors) == 0:
        return matches

    # Convert to float32
    query_descriptors = query_descriptors.astype(np.float32)
    stored_descriptors = stored_descriptors.astype(np.float32)

    # Configure FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=50)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find k=2 nearest matches for each descriptor
    matches_flann = flann.knnMatch(query_descriptors, stored_descriptors, k=2)
    
    # Apply basic ratio test
    for i, (m, n) in enumerate(matches_flann):
        if m.distance < ratio_threshold * n.distance:
            matches.append((i, feature_points[m.trainIdx]))
    
    print(f"Found {len(matches)} matches from {len(matches_flann)} potential matches")
    return matches

def visualize_3d_points(feature_points=None, current_points=None):
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

def visualize_alignment(obj_points, est_points, R, t):
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

def draw_frame_axes(image, R, t, camera_matrix, dist_coeffs=None, axis_length=0.1):
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

def compute_reprojection_error(image_points, object_points, R, t, camera_matrix, 
                            dist_coeffs=None, image=None, visualize=False):
    """
    Compute and optionally visualize reprojection error.
    
    Args:
        image_points: Nx2 array of measured image points
        object_points: Nx3 array of corresponding 3D object points
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients (optional)
        image: Optional input image for visualization
        visualize: Whether to create visualization
        
    Returns:
        errors: Nx1 array of point-wise reprojection errors
        rmse: Root mean square reprojection error
        visualization: Optional visualization image if requested
    """
    # Convert rotation matrix to vector
    rvec = cv2.Rodrigues(R)[0]
    tvec = t.reshape(3, 1)
    
    # If no distortion coefficients provided, use zero distortion
    if dist_coeffs is None:
        dist_coeffs = np.zeros(4, dtype=np.float32)
    
    # Project 3D points to image plane
    proj_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    proj_points = proj_points.reshape(-1, 2)
    
    # Compute euclidean distances between measured and projected points
    errors = np.linalg.norm(proj_points - image_points, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    
    vis_img = None
    if visualize and image is not None:
        # Create visualization
        vis_img = image.copy()
        
        # Draw measured points in green
        for pt in image_points:
            cv2.circle(vis_img, tuple(pt.astype(int)), 3, (0,255,0), -1)
            
        # Draw projected points in red and connect with lines
        for i, (proj, meas) in enumerate(zip(proj_points, image_points)):
            # Draw projected point
            cv2.circle(vis_img, tuple(proj.astype(int)), 3, (0,0,255), -1)
            
            # Draw line connecting measured and projected
            cv2.line(vis_img, 
                    tuple(meas.astype(int)), 
                    tuple(proj.astype(int)),
                    (255,0,0), 1)
            
            # Optionally display error value
            error = errors[i]
            cv2.putText(vis_img, 
                    f'{error:.1f}px',
                    tuple((proj + [5, 5]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1)
        
        # Add overall RMSE to image
        cv2.putText(vis_img,
                f'RMSE: {rmse:.2f}px',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)
        
        # Create error histogram
        plt.figure(figsize=(10, 4))
        plt.hist(errors, bins=30, color='blue', alpha=0.7)
        plt.title('Reprojection Error Distribution')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Count')
        plt.axvline(rmse, color='red', linestyle='--', label=f'RMSE: {rmse:.2f}px')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return errors, rmse, vis_img

def compute_transform_error(obj_points, est_points, R, t):
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

def get_transform_distance(T1, T2):
    """
    Calculate the distance between two transforms.
    Returns rotation difference in degrees and translation difference in units.
    
    Args:
        T1, T2: 4x4 transformation matrices
        
    Returns:
        tuple: (rotation_diff_degrees, translation_diff_magnitude)
    """
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    
    R = np.dot(R1.T, R2)
    # Get rotation difference in radians
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    # Get translation difference magnitude
    trans_diff = np.linalg.norm(t1 - t2)
    
    return theta, trans_diff

def rotation_to_rpy_camera(T):
    """
    Convert rotation matrix from transform matrix T to roll-pitch-yaw angles
    about the camera frame, assuming object z-axis up.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        roll, pitch, yaw angles in radians
    """
    # Extract rotation matrix from transform
    R = T[:3, :3]
    
    # Calculate pitch (rotation about y-axis)
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    
    # Calculate yaw (rotation about z-axis)
    # Handle singularity when pitch = ±90°
    if np.abs(pitch) > np.pi/2 - 1e-8:
        # Gimbal lock case
        yaw = 0  # Set yaw to zero as it's arbitrary in gimbal lock
        # Calculate roll considering gimbal lock
        if pitch > 0:
            roll = np.arctan2(R[0,1], R[1,1])
        else:
            roll = -np.arctan2(R[0,1], R[1,1])
    else:
        # Normal case
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    
    return roll, pitch, yaw

def filter_errors_simple(errors, min_inliers=3):
    """
    Filter error array using ensemble of statistical methods.
    
    Args:
        errors: Array of error values
        min_inliers: Minimum number of points to keep
        
    Returns:
        mask: Boolean array, True for inliers
    """
    if len(errors) < min_inliers:
        return np.ones(len(errors), dtype=bool)
    
    # Method 1: Z-score
    z_scores = np.abs(stats.zscore(errors))
    mask1 = z_scores < 2.0
    
    # Method 2: IQR
    Q1 = np.percentile(errors, 25)
    Q3 = np.percentile(errors, 75)
    IQR = Q3 - Q1
    mask2 = errors < (Q3 + 1.5 * IQR)
    
    # Method 3: Median
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    mask3 = errors < (median + 3 * mad)
    
    # Combine masks using majority voting
    votes = mask1.astype(int) + mask2.astype(int) + mask3.astype(int)
    final_mask = votes >= 2  # Point is inlier if at least 2 methods agree
    
    # Ensure minimum number of inliers
    if np.sum(final_mask) < min_inliers:
        # Fall back to method that kept the most inliers
        counts = [np.sum(mask1), np.sum(mask2), np.sum(mask3)]
        best_mask = [mask1, mask2, mask3][np.argmax(counts)]
        return best_mask
    
    return final_mask