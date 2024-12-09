import argparse
import numpy as np
import cv2
from model3d import Model3D
from object3d import Object3D
from utils import get_transform_distance, visualize_alignment, compute_transform_error, rotation_to_rpy_camera, ModelUtils
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def visualize_validation_results(df, save_dir=None):
    """
    Create visualizations of validation results.
    
    Args:
        df: DataFrame with validation results
        save_dir: Optional directory to save plots
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    # Use a basic style
    plt.style.use('default')
    
    # Set general plot parameters
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    
    # 1. Success rate vs pose parameters
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, param in zip([ax1, ax2, ax3], ['elevation', 'azimuth', 'yaw']):
        success_rate = df.groupby(param)['estimation_success'].mean() * 100
        ax.plot(success_rate.index, success_rate.values, 'o-', color='blue')
        ax.set_xlabel(f'{param.capitalize()} (degrees)')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Success Rate vs {param.capitalize()}')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/success_rates.png")
    plt.show()
    
    # 2. Error distributions
    successful_df = df[df['estimation_success']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rotation error
    ax1.hist(successful_df['rotation_error_deg'], bins=20, color='blue', alpha=0.7)
    ax1.set_title('Rotation Error Distribution')
    ax1.set_xlabel('Rotation Error (degrees)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Translation error
    ax2.hist(successful_df['translation_error'], bins=20, color='blue', alpha=0.7)
    ax2.set_title('Translation Error Distribution')
    ax2.set_xlabel('Translation Error (units)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # In-plane error
    ax3.hist(successful_df['inplane_error_deg'], bins=20, color='blue', alpha=0.7)
    ax3.set_title('In-Plane Rotation Error')
    ax3.set_xlabel('Error (degrees)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Out-of-plane errors scatter plot
    ax4.scatter(successful_df['outofplane_x_error_deg'], 
               successful_df['outofplane_y_error_deg'], 
               alpha=0.5)
    ax4.set_title('Out-of-Plane Rotation Errors')
    ax4.set_xlabel('X Error (degrees)')
    ax4.set_ylabel('Y Error (degrees)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/error_distributions.png")
    plt.show()
    
    # 3. Correlation matrix
    error_cols = ['rotation_error_deg', 'translation_error', 
                 'inplane_error_deg', 'outofplane_x_error_deg', 
                 'outofplane_y_error_deg', 'matching_points', 'inlier_ratio']
    
    correlation_matrix = successful_df[error_cols].corr()
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # Add correlation values as text
    for i in range(len(error_cols)):
        for j in range(len(error_cols)):
            text = f'{correlation_matrix.iloc[i, j]:.2f}'
            plt.text(j, i, text, ha='center', va='center')
    
    plt.xticks(range(len(error_cols)), error_cols, rotation=45, ha='right')
    plt.yticks(range(len(error_cols)), error_cols)
    plt.title('Error Metric Correlations')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/correlation_matrix.png")
    plt.show()

def run_grid_validation(viewer, detector, rotation_ranges, translation_ranges=None, steps=5,
                       visualization_dir=None, save_example_images=False):
    """
    Run systematic validation across a grid of poses.
    
    Args:
        viewer: Model3D instance
        detector: Object3D instance
        rotation_ranges: dict with keys 'elevation', 'azimuth', 'yaw', each containing (min, max)
        translation_ranges: Optional dict with keys 'x', 'y', 'z', each containing (min, max)
        steps: Number of steps for each range
        visualization_dir: Directory to save visualizations
        save_example_images: Whether to save example images for each pose
        
    Returns:
        pd.DataFrame with validation statistics
    """
    camera_matrix, width, height = viewer.get_camera_intrinsics()
    
    if visualization_dir:
        Path(visualization_dir).mkdir(parents=True, exist_ok=True)
        if save_example_images:
            image_dir = Path(visualization_dir) / "example_images"
            image_dir.mkdir(exist_ok=True)
    
    # Store original pose
    orig_elevation = viewer.elevation
    orig_azimuth = viewer.azimuth
    orig_yaw = viewer.yaw
    orig_distance = viewer.distance
    
    results = []
    
    # Generate test poses
    elevation_range = np.linspace(rotation_ranges['elevation'][0], 
                                rotation_ranges['elevation'][1], steps)
    azimuth_range = np.linspace(rotation_ranges['azimuth'][0], 
                               rotation_ranges['azimuth'][1], steps)
    yaw_range = np.linspace(rotation_ranges['yaw'][0], 
                           rotation_ranges['yaw'][1], steps)
    
    total_iterations = len(elevation_range) * len(azimuth_range) * len(yaw_range)
    pbar = tqdm(total=total_iterations, desc="Validating poses")
    
    try:
        for elevation in elevation_range:
            for azimuth in azimuth_range:
                for yaw in yaw_range:
                    # Set pose
                    viewer.elevation = elevation
                    viewer.azimuth = azimuth
                    viewer.yaw = yaw
                    
                    # Render frame and get depth
                    image = viewer.render_frame()
                    _, depth = viewer.get_depth_image()
                    
                    # Get ground truth pose
                    _, camera_to_world_gt = viewer.get_camera_transforms()
                    camera_to_world_gt = viewer.transform_to_ros_coord_system(camera_to_world_gt)
                    
                    # Create mask from rendered image
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    
                    # Run pose estimation
                    img_pts, obj_pts = detector.match_image_points(image, mask)
                    
                    result = {
                        'elevation': elevation,
                        'azimuth': azimuth,
                        'yaw': yaw,
                        'matching_points': len(img_pts) if img_pts is not None else 0,
                        'estimation_success': False,
                        'good_estimate': False,
                        'rotation_error_deg': None,
                        'translation_error': None,
                        'inplane_error_deg': None,
                        'outofplane_x_error_deg': None,
                        'outofplane_y_error_deg': None,
                        'inlier_ratio': None
                    }
                    
                    # Create visualization image
                    vis_image = image.copy()
                    
                    # Draw ground truth axes
                    rvec_gt = cv2.Rodrigues(camera_to_world_gt[:3, :3])[0]
                    tvec_gt = camera_to_world_gt[:3, 3]
                    cv2.drawFrameAxes(vis_image, camera_matrix, None, rvec_gt, tvec_gt, 1.2)
                    
                    if img_pts is not None and len(img_pts) >= 4:
                        # Get 3D points from depth
                        real_pts, valid_indices = detector.estimate3d(img_pts, depth, camera_matrix)
                        
                        # Filter points
                        obj_pts = obj_pts[valid_indices]
                        real_pts = real_pts[valid_indices]
                        img_pts = img_pts[valid_indices]
                        
                        # Draw matching points
                        for pt in img_pts:
                            cv2.circle(vis_image, tuple(pt.astype(int)), 3, (0, 255, 0), -1)
                        
                        # Estimate pose
                        R, t, inliers = detector.estimate_transform(
                            real_pts=real_pts,
                            obj_pts=obj_pts,
                            img_pts=img_pts,
                            camera_matrix=camera_matrix,
                        )

                        
                        if R is not None:
                            result['estimation_success'] = True
                            
                            # Draw estimated pose axes
                            rvec_est = cv2.Rodrigues(R)[0]
                            cv2.drawFrameAxes(vis_image, camera_matrix, None, rvec_est, t, 1.5)
                            
                            # Compute transform error
                            T_est = np.eye(4)
                            T_est[:3, :3] = R
                            T_est[:3, 3] = t.squeeze()
                            
                            rot_diff, trans_diff = get_transform_distance(camera_to_world_gt, T_est)
                            result['rotation_error_deg'] = np.rad2deg(rot_diff)
                            result['translation_error'] = trans_diff
                            if np.rad2deg(rot_diff) < 5.0 and result['translation_error'] < 0.02:
                                result['good_estimate'] = True
                            
                            # Compute in-plane and out-of-plane errors
                            out_of_plane_x, out_of_plane_y, in_plane = rotation_to_rpy_camera(T_est)
                            out_of_plane_x_gt, out_of_plane_y_gt, in_plane_gt = rotation_to_rpy_camera(camera_to_world_gt)
                            
                            result['inplane_error_deg'] = np.rad2deg(np.arctan2(
                                np.sin(in_plane - in_plane_gt), 
                                np.cos(in_plane - in_plane_gt)))
                            result['outofplane_x_error_deg'] = np.rad2deg(np.arctan2(
                                np.sin(out_of_plane_x - out_of_plane_x_gt),
                                np.cos(out_of_plane_x - out_of_plane_x_gt)))
                            result['outofplane_y_error_deg'] = np.rad2deg(np.arctan2(
                                np.sin(out_of_plane_y - out_of_plane_y_gt),
                                np.cos(out_of_plane_y - out_of_plane_y_gt)))
                            
                            result['inlier_ratio'] = np.sum(inliers) / len(inliers) if inliers is not None else 1.0
                            
                            # Add error text to image
                            error_text = [
                                f"Rot err: {result['rotation_error_deg']:.1f}",
                                f"Trans err: {result['translation_error']:.3f}",
                                f"Matches: {result['matching_points']}"
                            ]
                            y_offset = 30
                            for text in error_text:
                                cv2.putText(vis_image, text, (10, y_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                y_offset += 30

                    cv2.imshow('3D Model Validator', vis_image)
                    cv2.waitKey(1)
                    
                    if save_example_images and visualization_dir:
                        image_path = image_dir / f"pose_e{elevation:.0f}_a{azimuth:.0f}_y{yaw:.0f}.png"
                        cv2.imwrite(str(image_path), vis_image)
                    
                    results.append(result)
                    pbar.update(1)
    
    finally:
        # Restore original pose
        viewer.elevation = orig_elevation
        viewer.azimuth = orig_azimuth
        viewer.yaw = orig_yaw
        viewer.distance = orig_distance
        pbar.close()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Compute summary statistics
    summary = {
        'total_poses': len(df),
        'successful_estimations': df['estimation_success'].sum(),
        'success_rate': df['estimation_success'].mean() * 100,
        'good_estimates': df['good_estimate'].sum(),
        'good_estimate_rate': df['good_estimate'].sum() / df['estimation_success'].sum() * 100,
        'avg_matching_points': df['matching_points'].mean(),
        'avg_rotation_error': df[df['estimation_success']]['rotation_error_deg'].mean(),
        'avg_translation_error': df[df['estimation_success']]['translation_error'].mean(),
        'avg_inplane_error': df[df['estimation_success']]['inplane_error_deg'].mean(),
        'avg_outofplane_x_error': df[df['estimation_success']]['outofplane_x_error_deg'].mean(),
        'avg_outofplane_y_error': df[df['estimation_success']]['outofplane_y_error_deg'].mean(),
        'avg_inlier_ratio': df[df['estimation_success']]['inlier_ratio'].mean()
    }
    
    print("\nValidation Summary:")
    print(f"Total poses tested: {summary['total_poses']}")
    print(f"Successful estimations: {summary['successful_estimations']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Good estimates: {summary['good_estimates']}")
    print(f"Good estimate rate: {summary['good_estimate_rate']:.1f}%")
    print(f"Average matching points: {summary['avg_matching_points']:.1f}")
    print("\nError Metrics (successful estimations only):")
    print(f"Average rotation error: {summary['avg_rotation_error']:.2f}°")
    print(f"Average translation error: {summary['avg_translation_error']:.3f} units")
    print(f"Average in-plane rotation error: {summary['avg_inplane_error']:.2f}°")
    print(f"Average out-of-plane rotation error X: {summary['avg_outofplane_x_error']:.2f}°")
    print(f"Average out-of-plane rotation error Y: {summary['avg_outofplane_y_error']:.2f}°")
    print(f"Average inlier ratio: {summary['avg_inlier_ratio']:.3f}")
    
    # Create visualizations if directory provided
    if visualization_dir:
        visualize_validation_results(df, visualization_dir)
    
    return df, summary

def main():
    parser = argparse.ArgumentParser(description='3D Model Pose Estimation Validator')
    parser.add_argument('--obj', required=True, help='Path to the OBJ file')
    parser.add_argument('--model', required=True, help='Path to model NPZ file')
    parser.add_argument('--scan_distance', type=float, default=6.0, help='Distance to scan')
    parser.add_argument('--validate', action='store_true', help='Run grid validation')
    parser.add_argument('--output', type=str, help='Path to save validation results CSV')
    parser.add_argument('--vis_dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--save_images', action='store_true', help='Save example images')
    parser.add_argument('--elevation_range', type=float, nargs=2, default=(-45, 45),
                       help='Min and max elevation angles')
    parser.add_argument('--azimuth_range', type=float, nargs=2, default=(-45, 45),
                       help='Min and max azimuth angles')
    parser.add_argument('--yaw_range', type=float, nargs=2, default=(-45, 45),
                       help='Min and max yaw angles')
    parser.add_argument('--steps', type=int, default=5,
                       help='Number of steps for each angle range')
    args = parser.parse_args()
    
    try:
        # Initialize both renderable model and object detector
        viewer = Model3D(args.obj, auto_scan=False, scan_distance=args.scan_distance)
        detector = Object3D(args.model)

        ModelUtils.auto_scale_and_center(viewer.meshes)

        if args.validate:
            # Define rotation ranges for validation
            rotation_ranges = {
                'elevation': (args.elevation_range[0], args.elevation_range[1]),
                'azimuth': (args.azimuth_range[0], args.azimuth_range[1]),
                'yaw': (args.yaw_range[0], args.yaw_range[1])
            }

            # Run validation
            results_df, summary = run_grid_validation(
                viewer=viewer,
                detector=detector,
                rotation_ranges=rotation_ranges,
                steps=args.steps,
                visualization_dir=args.vis_dir,
                save_example_images=args.save_images
            )
            
            # Save results if output path provided
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")
            
            return
        
        WINDOW_NAME = '3D Model Validator'
        DEPTH_WINDOW = 'Depth View'
        RESULT_WINDOW = 'Pose Estimation Result'
        
        cv2.imshow(WINDOW_NAME, np.zeros((1024, 1024, 3), dtype=np.uint8))
        cv2.setMouseCallback(WINDOW_NAME, viewer.mouse_callback)
        
        print("\nControls:")
        print("Left/Right Arrow: Rotate azimuth (Y-axis)")
        print("Up/Down Arrow: Rotate elevation (X-axis)")
        print("A/D: Rotate yaw (Z-axis)")
        print("W/S: Zoom in/out")
        print("R: Reset view")
        print("Space: Run pose estimation")
        print("Q: Quit")

        camera_matrix, _, _ = viewer.get_camera_intrinsics()

        while True:
            # Render current frame and get corresponding depth
            image = viewer.render_frame()
            depth_colored, depth = viewer.get_depth_image()
            
            # Get ground truth pose
            _, camera_to_world_gt = viewer.get_camera_transforms()

            camera_to_world_gt = viewer.transform_to_ros_coord_system(camera_to_world_gt)

            # Draw ground truth axes
            rvec_gt = cv2.Rodrigues(camera_to_world_gt[:3, :3])[0]
            tvec_gt = camera_to_world_gt[:3, 3]
            frame_gt = cv2.drawFrameAxes(image, camera_matrix, None, rvec_gt, tvec_gt, 1.2)

            # Show images
            cv2.imshow(WINDOW_NAME, frame_gt)
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
            elif key == ord(' '):  # Run pose estimation

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                # Get image points and object points
                img_pts, obj_pts = detector.match_image_points(image, mask)
                
                if img_pts is not None:
                    print(f"\nFound {len(img_pts)} matching points")
                    
                    # Get 3D points from depth
                    real_pts, valid_indices = detector.estimate3d(img_pts, depth, camera_matrix)
                    
                    # Filter points based on valid depth values
                    obj_pts = obj_pts[valid_indices]
                    real_pts = real_pts[valid_indices]
                    img_pts = img_pts[valid_indices]
                    
                    print(f"Valid 3D points: {len(real_pts)}")

                    # Estimate pose using selected method
                    R, t, inliers = detector.estimate_transform(
                        real_pts=real_pts,
                        obj_pts=obj_pts,
                        img_pts=img_pts,
                        camera_matrix=camera_matrix,
                    )

                    if R is not None:
                        
                        T_est = np.eye(4)
                        T_est[:3, :3] = R
                        T_est[:3, 3] = t.squeeze()
                        
                        # Draw estimated pose
                        rvec_est = cv2.Rodrigues(R)[0]
                        tvec_est = t
                        frame_result = frame_gt.copy()
                        pts = img_pts[inliers]
                        cv2.drawFrameAxes(frame_result, camera_matrix, None, rvec_est, tvec_est, 1.5)
                        matched_kp = [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) 
                            for pt in pts]
                        cv2.drawKeypoints(frame_result, matched_kp, frame_result, 
                                    color=(0,255,0), 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        # Compute and display transform error
                        rot_diff, trans_diff = get_transform_distance(camera_to_world_gt, T_est)
                        print(f"\nTransform Error:")
                        print(f"Rotation difference: {np.rad2deg(rot_diff):.2f} degrees")
                        print(f"Translation difference: {trans_diff:.3f} units")
                        
                        # compute in plane and out-of-plane errors
                        out_of_plane_rotation_x, out_of_plane_rotation_y, in_plane_rotation = rotation_to_rpy_camera(T_est)
                        out_of_plane_rotation_x_gt, out_of_plane_rotation_y_gt, in_plane_rotation_gt = rotation_to_rpy_camera(camera_to_world_gt)
                        error_out_of_plane_x = np.arctan2(np.sin(out_of_plane_rotation_x - out_of_plane_rotation_x_gt), np.cos(out_of_plane_rotation_x - out_of_plane_rotation_x_gt))
                        error_out_of_plane_y = np.arctan2(np.sin(out_of_plane_rotation_y - out_of_plane_rotation_y_gt), np.cos(out_of_plane_rotation_y - out_of_plane_rotation_y_gt))
                        error_in_plane = np.arctan2(np.sin(in_plane_rotation - in_plane_rotation_gt), np.cos(in_plane_rotation - in_plane_rotation_gt))
                        
                        
                        # Draw error information on image
                        error_text = [
                            f"Rot diff: {np.rad2deg(rot_diff):.2f} deg",
                            f"Trans diff: {trans_diff:.3f} m",
                            f"IP rot: {np.rad2deg(error_in_plane):.2f} deg",
                            f"OP rot x: {np.rad2deg(error_out_of_plane_x):.2f} deg",
                            f"OP rot y: {np.rad2deg(error_out_of_plane_y):.2f} deg"
                        ]
                        y_offset = 30
                        for text in error_text:
                            cv2.putText(frame_result, text, (10, y_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_offset += 30

                        # Show result
                        cv2.imshow(RESULT_WINDOW, frame_result)

                        # Show alignment visualization
                        filtered_obj_pts = obj_pts[inliers] if inliers is not None else obj_pts
                        filtered_real_pts = real_pts[inliers] if inliers is not None else real_pts
                        visualize_alignment(filtered_obj_pts, filtered_real_pts, R, t)
                    else:
                        print("Pose estimation failed")
                else:
                    print("No matches found")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()