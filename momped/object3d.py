import numpy as np
import cv2
from momped.utils import (
    detect_sift_features, 
    find_matching_points, 
    visualize_alignment, 
    compute_reprojection_error, 
    draw_frame_axes, 
    compute_transform_error,
    filter_errors_simple
)
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import open3d as o3d
import time

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
                'point3d': feat_dict['point3d'] / 10.0, # Scale down to meters when in real world
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
        Match image features with stored 3D points using mask as ROI.
        Args:
            image: Input image
            mask: Binary mask for ROI
        Returns:
            image_points: 2D points in the image
            obj_pts: Corresponding 3D points in object space
        """
        if mask is None:
            return None, None

        # Find ROI from mask
        x, y, w, h = cv2.boundingRect(mask)
        
        # Extract ROI from image and mask
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        # Apply mask to ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)
        
        # Use utility function to detect features on ROI
        keypoints, descriptors = detect_sift_features(masked_roi)
        
        if keypoints is None:
            return None, None
        
        # Adjust keypoint coordinates back to original image space
        for kp in keypoints:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        
        # Use utility function to find matches
        matches = find_matching_points(self.feature_points, descriptors)
        
        if not matches:
            return None, None
        
        # Extract corresponding points
        image_points = []  # 2D points in image
        obj_pts = []  # 3D points in object space
        
        for query_idx, matched_feature in matches:
            # Get 2D point from keypoint
            image_points.append(keypoints[query_idx].pt)
            # Get corresponding 3D point
            obj_pts.append(matched_feature['point3d'])
        
        return np.array(image_points), np.array(obj_pts)
    
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

        # Back-project to 3D (vectorized)
        X = (image_points[:, 0] - cx) * Z / fx
        Y = (image_points[:, 1] - cy) * Z / fy
        
        return np.column_stack((X, Y, Z)), depth_mask
 
    def estimate_transform(self, obj_pts, real_pts, img_pts, camera_matrix, inlier_threshold=0.01):
        '''
        Estimate 6D pose by first filtering out mis-matched SIFT features by using 
        relative distance error between real points and object points. Then using Kabsch algorithm
        to find the rigid transformation between the filtered points.
        '''
        
        rel_dist_obj = np.linalg.norm(obj_pts[:, np.newaxis] - obj_pts, axis=2)
        np.fill_diagonal(rel_dist_obj, np.inf)  # Ignore self-distance
        
        rel_dist_real = np.linalg.norm(real_pts[:, np.newaxis] - real_pts, axis=2)
        np.fill_diagonal(rel_dist_real, np.inf)  # Ignore self-distance

        dist_error = np.abs(rel_dist_obj - rel_dist_real)
        inliers_dist = np.sum(dist_error < inlier_threshold, axis=1) >= 5

        R_pnp, t_pnp, inliers_pnp = self.estimate_pnp_ransac(
                            img_pts=img_pts,
                            obj_pts=obj_pts,
                            camera_matrix=camera_matrix,
                            ransac_threshold=6.0,
                            confidence=0.99,
                            max_iters=1000
                        )
        
        if inliers_pnp is None:
            return None, None, None

        inliers = inliers_dist & inliers_pnp

        filtered_obj_pts = obj_pts[inliers]
        filtered_real_pts = real_pts[inliers]

        if len(filtered_obj_pts) < 5:
            return None, None, None

        R_rigid, t_rigid = self.estimate_rigid_transform_3d(filtered_obj_pts, filtered_real_pts)

        transformed_points = (R_rigid @ filtered_obj_pts.T).T + t_rigid

        error_rigid = np.linalg.norm(filtered_real_pts - transformed_points, axis=1)

        second_filtered_mask = filter_errors_simple(error_rigid, 3)

        R, t = self.estimate_rigid_transform_3d(filtered_obj_pts[second_filtered_mask], filtered_real_pts[second_filtered_mask])

        combined_mask = np.zeros(len(obj_pts), dtype=bool)  # Initialize mask for all original points
        combined_mask[inliers] = second_filtered_mask

        return R, t, combined_mask

    def estimate_pnp_ransac(self, img_pts, obj_pts, camera_matrix, dist_coeffs=None, 
                       ransac_threshold=10.0, confidence=0.99, max_iters=1000):
        """
        Estimate 6D pose using PnP RANSAC.
        """

        if len(img_pts) != len(obj_pts) or len(img_pts) < 4:
            return None, None, None
            
        # Ensure points are float32
        img_pts = img_pts.astype(np.float32)
        obj_pts = obj_pts.astype(np.float32)
        
        # If no distortion coefficients provided, use zero distortion
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float32)
            
        try:
            # Estimate pose using RANSAC
            
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=max_iters,
                reprojectionError=ransac_threshold,
                confidence=confidence,
            )
            
            if not retval or inliers is None:
                print("PnP RANSAC failed to converge")
                return None, None, None
                
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3)
            
            # Create inlier mask
            inlier_mask = np.zeros(len(img_pts), dtype=bool)
            inlier_mask[inliers.ravel()] = True
            
            return R, t, inlier_mask
        
        except cv2.error as e:
            print(f"OpenCV Error in solvePnPRansac: {e}")
            return None, None, None
        except Exception as e:
            print(f"Error in estimate_pnp_ransac: {e}")
            return None, None, None
    
    def estimate_rigid_transform_3d(self, obj_pts, real_pts):
        """
        Estimate a 3D rigid transformation (rotation and translation) that aligns 
        set of points A to set of points B.
        Parameters:
            A (np.ndarray): Source points, shape (N, 3).
            B (np.ndarray): Destination points, shape (N, 3).
        Returns:
            R (np.ndarray): 3x3 rotation matrix.
            t (np.ndarray): 3x1 translation vector.
        """
        assert obj_pts.shape == real_pts.shape, "A and B must be of the same shape"
        # Calculate centroids
        centroid_A = np.mean(obj_pts, axis=0)
        centroid_B = np.mean(real_pts, axis=0)
        # Center the points
        AA = obj_pts - centroid_A
        BB = real_pts - centroid_B
        # Calculate covariance matrix
        H = AA.T @ BB
        # Perform SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Handle special reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        # Calculate translation
        t = centroid_B - R @ centroid_A
        return R, t

    def estimate_transform_registration_based_on_correspondence(self, obj_points, est_points, correspondence_threshold=0.05):
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
        
        if len(obj_points) != len(est_points) or len(obj_points) < 5:
            return None, None, None

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

        best_rmse = np.inf
        for n in range(3, len(obj_points)-1):
            # Perform RANSAC registration
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source=pcd_obj,
                target=pcd_est,
                corres=o3d.utility.Vector2iVector(corres),
                max_correspondence_distance=correspondence_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=n,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    max_iteration=1000,
                    confidence=0.99
                )
            )
            
            inlier_mask = np.zeros(len(obj_points), dtype=bool)
            correspondence_set = np.asarray(result.correspondence_set)
            inlier_mask[correspondence_set[:, 0]] = True

            # Extract rotation and translation from transformation matrix
            transformation = result.transformation
            R = transformation[:3, :3]
            t = transformation[:3, 3]

            # Compute alignment error
            _, rmse = compute_transform_error(
                obj_points=obj_points,
                est_points=est_points,
                R=R,
                t=t
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_R = R
                best_t = t
                best_inliers = inlier_mask

        return best_R, best_t, best_inliers

# Example usage:
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Object pose estimation using different methods')
    parser.add_argument('--method', type=str, default='pnp',
                       choices=['combined','pnp', 'ransac', 'rigid'],
                       help='Pose estimation method to use')
    parser.add_argument('--rgb', type=str, required=True,
                       help='Path to RGB image')
    parser.add_argument('--depth', type=str, required=True,
                       help='Path to depth image')
    parser.add_argument('--mask', type=str, required=True,
                       help='Path to mask image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model NPZ file')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    
    args = parser.parse_args()

    # Initialize object with stored features
    obj = Object3D(args.model)

    # Load images
    image = cv2.imread(args.rgb)
    depth_image = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    
    if image is None or depth_image is None or mask is None:
        print("Error loading images")
        exit(1)

    # Process mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Define the camera matrix (should be loaded from calibration)
    camera_matrix = np.array([
        [3.195820007324218750e+02, 0.000000000000000000e+00, 3.202149847676955687e+02],
        [0.000000000000000000e+00, 4.171186828613281250e+02, 2.443486680871046701e+02],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ], dtype=np.float32)

    # Convert depth image to meters
    if depth_image.dtype != np.float32:
        depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters

    start_time = time.time()

    # Get matching points
    img_pts, obj_pts = obj.match_image_points(image, mask)
    
    if img_pts is not None:
        print(f"Found {len(img_pts)} matching points")

        # Get 3D points from depth
        real_pts, valid_indices = obj.estimate3d(img_pts, depth_image, camera_matrix)
        
        # Filter points based on valid depth values
        obj_pts = obj_pts[valid_indices]
        real_pts = real_pts[valid_indices]
        img_pts = img_pts[valid_indices]
        
        print(f"Valid 3D points: {len(real_pts)}")

        # Estimate pose using selected method
        if args.method == 'combined':
            R, t, inliers = obj.estimate_transform(
                img_pts=img_pts,
                obj_pts=obj_pts,
                real_pts=real_pts,
                camera_matrix=camera_matrix
            )
        elif args.method == 'pnp':
            R, t, inliers = obj.estimate_pnp_ransac(
                image_points=img_pts,
                obj_pts=obj_pts,
                camera_matrix=camera_matrix,
                ransac_threshold=10.0,
                confidence=0.99
            )
        elif args.method == 'ransac':
            R, t, inliers = obj.estimate_transform_registration_based_on_correspondence(
                obj_points=obj_pts,
                est_points=real_pts,
                correspondence_threshold=0.05
            )
        else:  # rigid transform
            R, t = obj.estimate_rigid_transform_3d(obj_pts, real_pts)
            inliers = np.ones(len(obj_pts), dtype=bool)  # All points used

        end_time = time.time()
        print(f"Pose estimation ({args.method}) took {end_time - start_time:.4f} seconds")

        if R is not None:
            # Compute and visualize reprojection error
            if args.method in ['pnp', 'combined']:
                errors, rmse, vis_img = compute_reprojection_error(
                    image_points=img_pts[inliers] if args.method == 'pnp' else img_pts,
                    obj_pts=obj_pts[inliers] if args.method == 'pnp' else obj_pts,
                    R=R,
                    t=t,
                    camera_matrix=camera_matrix,
                    image=image,
                    visualize=args.visualize
                )
                print(f"Reprojection RMSE: {rmse:.3f} pixels")
                cv2.imshow("Reprojection", vis_img)
            
            # Draw coordinate axes
            img_axes = draw_frame_axes(
                image=image,
                R=R,
                t=t,
                camera_matrix=camera_matrix,
                axis_length=0.1
            )

            if args.visualize:
                # Show matching points
                pts = img_pts[inliers] if args.method == 'pnp' else img_pts
                matched_kp = [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) 
                            for pt in pts]
                cv2.drawKeypoints(img_axes, matched_kp, img_axes, 
                                    color=(0,255,0), 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                # Show pose
                cv2.imshow("Pose", img_axes)
                
                # Show alignment visualization
                visualize_alignment(
                    obj_points=obj_pts if args.method == 'rigid' else obj_pts[inliers],
                    est_points=real_pts if args.method == 'rigid' else real_pts[inliers],
                    R=R,
                    t=t
                )
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Pose estimation failed")
    else:
        print("No matches found")

