import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
from model3d import Model3D
from utils import ModelUtils

def depth_to_point_cloud(depth_image, camera_matrix, subsample_factor=4):
    """
    Convert depth image to point cloud using camera intrinsics.
    Subsamples points to reduce density.
    
    Args:
        depth_image: HxW depth image in meters
        camera_matrix: 3x3 camera intrinsic matrix
        subsample_factor: Factor to subsample points (higher = fewer points)
    
    Returns:
        Nx3 array of 3D points
    """
    # Get focal length and principal point from camera matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Create pixel coordinate grid
    rows, cols = depth_image.shape
    x_grid, y_grid = np.meshgrid(np.arange(0, cols, subsample_factor),
                                np.arange(0, rows, subsample_factor))
    
    # Flatten grid and depth
    x = x_grid.flatten()
    y = y_grid.flatten()
    z = depth_image[y_grid, x_grid].flatten()
    
    # Remove points with invalid depth
    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]
    
    # Convert to 3D points
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    
    return np.column_stack([X, Y, Z])

def fit_cuboid(points, n_iterations=5, inlier_thresh=2.0):
    """
    Fit a cuboid to a point cloud using iteratively refined PCA.
    
    Args:
        points: Nx3 array of points
        n_iterations: Number of refinement iterations
        inlier_thresh: Threshold for inlier detection in standard deviations
        
    Returns:
        dict containing:
            - center: 3D center point
            - dimensions: 3D dimensions
            - rotation: 3x3 rotation matrix
            - error: fitting error
    """
    points = np.asarray(points)
    if len(points) < 4:
        return None
        
    # Initial center estimate using median for robustness
    best_error = float('inf')
    best_params = None
    center = np.median(points, axis=0)
    current_points = points - center
    
    for iteration in range(n_iterations):
        if len(current_points) < 4:  # Need at least 4 points for PCA
            break
            
        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(current_points)
        
        # Get rotation matrix from PCA
        rotation = pca.components_
        
        # Transform points to PCA space
        local_points = current_points @ rotation.T
        
        # Initialize mask for this iteration
        inlier_mask = np.ones(len(current_points), dtype=bool)
        dimensions = np.zeros(3)
        
        # Filter points along each dimension
        for dim in range(3):
            points_1d = local_points[inlier_mask, dim]
            if len(points_1d) < 4:
                break
                
            median = np.median(points_1d)
            mad = np.median(np.abs(points_1d - median))
            sigma = mad * 1.4826  # Convert MAD to standard deviation estimate
            
            # Avoid issues with constant values
            if sigma < 1e-6:
                continue
                
            # Update mask for this dimension
            dim_inliers = np.abs(points_1d - median) < (inlier_thresh * sigma)
            inlier_mask[inlier_mask] = dim_inliers
            
            # Calculate dimension based on robust statistics
            valid_points = points_1d[dim_inliers]
            if len(valid_points) > 0:
                dimensions[dim] = np.max(valid_points) - np.min(valid_points)
        
        # Skip if we don't have enough inliers
        if np.sum(inlier_mask) < 4:
            continue
        
        # Calculate error for this iteration
        # Mean squared distance from points to cuboid surface
        half_dims = dimensions / 2
        dx = np.abs(local_points[:, 0]) - half_dims[0]
        dy = np.abs(local_points[:, 1]) - half_dims[1]
        dz = np.abs(local_points[:, 2]) - half_dims[2]
        
        outside_dist = np.sqrt(np.maximum(dx, 0)**2 + 
                             np.maximum(dy, 0)**2 + 
                             np.maximum(dz, 0)**2)
        inside_dist = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0)
        distances = outside_dist + inside_dist
        error = np.mean(distances**2)
        
        if error < best_error:
            best_error = error
            best_params = {
                'center': center,
                'rotation': rotation,
                'dimensions': dimensions,
                'error': error
            }
            
        # Update points for next iteration
        current_points = current_points[inlier_mask]
    
    return best_params

def compute_fitting_error(local_points, dimensions):
    """Compute mean squared distance from points to cuboid surface."""
    half_dims = dimensions / 2
    dx = np.abs(local_points[:, 0]) - half_dims[0]
    dy = np.abs(local_points[:, 1]) - half_dims[1]
    dz = np.abs(local_points[:, 2]) - half_dims[2]
    
    outside_dist = np.sqrt(np.maximum(dx, 0)**2 + 
                         np.maximum(dy, 0)**2 + 
                         np.maximum(dz, 0)**2)
    inside_dist = np.minimum(np.maximum(np.maximum(dx, dy), dz), 0)
    
    distances = outside_dist + inside_dist
    return np.mean(distances**2)

def get_cuboid_corners(center, dimensions, rotation):
    """Get the 8 corners of a cuboid."""
    half_dims = dimensions / 2
    corners_local = np.array([
        [-1, -1, -1],  # 0: left  bottom back
        [-1, -1,  1],  # 1: left  bottom front
        [-1,  1, -1],  # 2: left  top    back
        [-1,  1,  1],  # 3: left  top    front
        [ 1, -1, -1],  # 4: right bottom back
        [ 1, -1,  1],  # 5: right bottom front
        [ 1,  1, -1],  # 6: right top    back
        [ 1,  1,  1]   # 7: right top    front
    ]) * half_dims
    
    return corners_local @ rotation + center

def visualize_fit(image, cuboid_params, camera_matrix, R=None, t=None):
    """
    Draw the fitted cuboid on the image.
    """
    # Get corners in world coordinates
    corners = get_cuboid_corners(
        cuboid_params['center'],
        cuboid_params['dimensions'],
        cuboid_params['rotation']
    )

    # Transform corners if R and t are provided
    if R is not None and t is not None:
        corners = (R @ corners.T).T + t

    # Project corners to image space
    corners_img = cv2.projectPoints(
        corners, 
        np.zeros(3), np.zeros(3),  # Already in camera frame
        camera_matrix, 
        None
    )[0].reshape(-1, 2).astype(int)

    # Define edges for visualization
    edges = [
        # Bottom face
        (0, 1), (1, 5), (5, 4), (4, 0),
        # Top face
        (2, 3), (3, 7), (7, 6), (6, 2),
        # Vertical edges
        (0, 2), (1, 3), (5, 7), (4, 6)
    ]

    # Draw edges
    vis_img = image.copy()
    for i, j in edges:
        cv2.line(vis_img, 
                tuple(corners_img[i]), 
                tuple(corners_img[j]), 
                (0, 255, 0), 2)

    # Add text with dimensions
    dims = cuboid_params['dimensions']
    dim_text = f"Dims: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f}"
    cv2.putText(vis_img, dim_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis_img

def plot_3d_fit(points, cuboid_params, title="3D Cuboid Fit"):
    """Plot points and fitted cuboid in 3D."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='b', marker='.', alpha=0.1, label='Points')
    
    # Plot fitted cuboid
    corners = get_cuboid_corners(
        cuboid_params['center'],
        cuboid_params['dimensions'],
        cuboid_params['rotation']
    )
    
    # Define edges
    edges = [
        # Bottom face
        (0, 1), (1, 5), (5, 4), (4, 0),
        # Top face
        (2, 3), (3, 7), (7, 6), (6, 2),
        # Vertical edges
        (0, 2), (1, 3), (5, 7), (4, 6)
    ]
    
    # Plot edges
    for i, j in edges:
        ax.plot3D([corners[i,0], corners[j,0]],
                 [corners[i,1], corners[j,1]],
                 [corners[i,2], corners[j,2]], 'r-')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make scaling uniform
    all_points = np.vstack([points, corners])
    max_range = np.array([
        all_points[:,0].max() - all_points[:,0].min(),
        all_points[:,1].max() - all_points[:,1].min(),
        all_points[:,2].max() - all_points[:,2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:,0].max() + all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max() + all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max() + all_points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_box_aspect([1,1,1])
    plt.legend()
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description='3D Model Shape Fitting Validator')
    parser.add_argument('--obj', required=True, help='Path to the OBJ file')
    parser.add_argument('--scan_distance', type=float, default=6.0, help='Distance to scan')
    args = parser.parse_args()
    
    try:
        # Initialize viewer
        viewer = Model3D(args.obj, auto_scan=False, scan_distance=args.scan_distance)
        
        WINDOW_NAME = '3D Shape Fitting Validator'
        DEPTH_WINDOW = 'Depth View'
        RESULT_WINDOW = 'Fitting Result'
        
        # Initial window setup
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
        
        # Set initial window sizes
        cv2.imshow(WINDOW_NAME, np.zeros((1024, 1024, 3), dtype=np.uint8))
        cv2.imshow(DEPTH_WINDOW, np.zeros((512, 512, 3), dtype=np.uint8))
        cv2.imshow(RESULT_WINDOW, np.zeros((512, 512, 3), dtype=np.uint8))
        
        ModelUtils.auto_scale_and_center(viewer.meshes)
        
        print("\nControls:")
        print("Left/Right Arrow: Rotate azimuth (Y-axis)")
        print("Up/Down Arrow: Rotate elevation (X-axis)")
        print("A/D: Rotate yaw (Z-axis)")
        print("W/S: Zoom in/out")
        print("R: Reset view")
        print("Space: Fit shape to current view")
        print("Q: Quit")
        
        camera_matrix, width, height = viewer.get_camera_intrinsics()
        
        while True:
            # Render current frame and get depth
            image = viewer.render_frame()
            depth_colored, depth = viewer.get_depth_image()
            
            # Get camera transforms
            world_to_camera, camera_to_world = viewer.get_camera_transforms()
            camera_to_world = viewer.transform_to_ros_coord_system(camera_to_world)
            
            # Show images
            cv2.imshow(WINDOW_NAME, image)
            if depth_colored is not None:
                cv2.imshow(DEPTH_WINDOW, depth_colored)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('w'):
                viewer.distance = max(0.5, viewer.distance - 0.1)
            elif key == ord('s'):
                viewer.distance += 0.1
            elif key == 81:  # Left arrow
                viewer.azimuth -= 5
            elif key == 83:  # Right arrow
                viewer.azimuth += 5
            elif key == 82:  # Up arrow
                viewer.elevation = min(90, viewer.elevation + 5)
            elif key == 84:  # Down arrow
                viewer.elevation = max(-90, viewer.elevation - 5)
            elif key == ord('a'):
                viewer.yaw -= 5
            elif key == ord('d'):
                viewer.yaw += 5
            elif key == ord('r'):
                viewer.distance = viewer.scan_distance
                viewer.elevation = 0.0
                viewer.azimuth = 0.0
                viewer.yaw = 0.0
            elif key == ord(' '):
                if depth is not None:
                    # Convert depth to point cloud with subsampling
                    rows, cols = depth.shape
                    x_grid, y_grid = np.meshgrid(np.arange(0, cols, 4),
                                                np.arange(0, rows, 4))
                    
                    # Get camera parameters
                    fx = camera_matrix[0, 0]
                    fy = camera_matrix[1, 1]
                    cx = camera_matrix[0, 2]
                    cy = camera_matrix[1, 2]
                    
                    # Sample points from depth
                    z = depth[y_grid, x_grid]
                    x = (x_grid - cx) * z / fx
                    y = (y_grid - cy) * z / fy
                    
                    # Stack coordinates and filter valid points
                    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
                    valid_mask = z.reshape(-1) > 0
                    valid_pts = points[valid_mask]
                    
                    if len(valid_pts) > 0:
                        # Fit cuboid
                        cuboid_params = fit_cuboid(valid_pts)
                        
                        # Print results
                        print("\nFitting results:")
                        print(f"Error: {cuboid_params['error']:.6f}")
                        print(f"Dimensions: {cuboid_params['dimensions']}")
                        print(f"Center: {cuboid_params['center']}")
                        
                        # Show 2D visualization
                        result_img = visualize_fit(image, cuboid_params, camera_matrix)
                        cv2.imshow(RESULT_WINDOW, result_img)
                        
                        # Show 3D visualization
                        plot_3d_fit(valid_pts, cuboid_params)
                        plt.show()
                
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e
    
if __name__ == '__main__':
    main()