import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_sift_features(image):
    """
    Detect SIFT features with balanced parameters.
    Args:
        nfeatures: Maximum number of features to detect
        contrast_threshold: Threshold for feature strength
        edge_threshold: Threshold for edge filtering
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
    
    # Filter based on response strength - keep top 75%
    min_response = np.percentile([kp.response for kp in keypoints], 25)
    
    filtered_keypoints = []
    filtered_descriptors = []
    
    for kp, desc in zip(keypoints, descriptors):
        # Filter by response
        if kp.response < min_response:
            continue
            
        # Minimal size filtering
        if kp.size < 2.0:  # Minimum size threshold
            continue
            
        filtered_keypoints.append(kp)
        filtered_descriptors.append(desc)
        
    if not filtered_keypoints:
        print("No features passed filtering")
        return None, None
        
    filtered_descriptors = np.array(filtered_descriptors)
    print(f"Detected {len(filtered_keypoints)} filtered SIFT features")
    
    return filtered_keypoints, filtered_descriptors

def find_matching_points(feature_points, query_descriptors):
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
        if m.distance < 0.85 * n.distance:
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