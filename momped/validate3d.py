import argparse
import numpy as np
import cv2
import torch
from model3d import Model3D
from object3d import Object3D
from utils import get_transform_distance, visualize_alignment, compute_transform_error, ModelUtils

def main():
    parser = argparse.ArgumentParser(description='3D Model Pose Estimation Validator')
    parser.add_argument('--obj', required=True, help='Path to the OBJ file')
    parser.add_argument('--model', required=True, help='Path to model NPZ file')
    parser.add_argument('--scan_distance', type=float, default=6.0, help='Distance to scan')
    
    args = parser.parse_args()
    
    try:
        # Initialize both renderable model and object detector
        viewer = Model3D(args.obj, auto_scan=False, scan_distance=args.scan_distance)
        detector = Object3D(args.model)
        
        WINDOW_NAME = '3D Model Validator'
        DEPTH_WINDOW = 'Depth View'
        RESULT_WINDOW = 'Pose Estimation Result'
        
        ModelUtils.auto_scale_and_center(viewer.meshes)

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

        camera_matrix, width, height = viewer.get_camera_intrinsics()

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
                        img_pts=img_pts,
                        obj_pts=obj_pts,
                        real_pts=real_pts,
                        camera_matrix=camera_matrix
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

                        # Draw error information on image
                        error_text = [
                            f"Rot diff: {np.rad2deg(rot_diff):.2f} deg",
                            f"Trans diff: {trans_diff:.3f} m"
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
                        # visualize_alignment(filtered_obj_pts, filtered_real_pts, R, t)
                    else:
                        print("Pose estimation failed")
                else:
                    print("No matches found")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()