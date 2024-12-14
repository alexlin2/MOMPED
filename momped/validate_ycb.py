import argparse
import numpy as np
import cv2
import os
import time
import csv
from pathlib import Path
from tqdm import tqdm
from object3d import Object3D
from utils import get_transform_distance, rotation_to_rpy_camera, visualize_alignment

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import visualization

class YCBBatchEvaluator:
    def __init__(self, dataset_path, model_root):
        self.dataset_path = dataset_path
        self.model_root = Path(model_root)
        self.window_name = "YCB-Video Pose Estimation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # For storing BOP format results
        self.results = []
        # For storing statistics
        self.statistics = []

    def load_frame_data(self, scene_id, frame_id, object_id):
        """Load RGB, depth, mask, and pose data for a frame."""
        # Scene paths
        scene_path = os.path.join(self.dataset_path, 'test', f'{scene_id:06d}')
        rgb_file = os.path.join(scene_path, 'rgb', f'{frame_id:06d}.png')
        depth_file = os.path.join(scene_path, 'depth', f'{frame_id:06d}.png')
        
        # Load scene info
        scene_camera = inout.load_scene_camera(os.path.join(scene_path, 'scene_camera.json'))
        scene_gt = inout.load_scene_gt(os.path.join(scene_path, 'scene_gt.json'))
        
        if scene_camera is None or scene_gt is None:
            print(f"Failed to load scene info for scene {scene_id}")
            return None, None, None, None, None, None

        str_frame_id = int(frame_id)

        # Load RGB image
        rgb = cv2.imread(rgb_file)
        if rgb is None:
            print(f"Failed to load RGB image for frame {frame_id}")
            return None, None, None, None, None, None
            
        # Load depth image
        depth = cv2.imread(depth_file, -1)  # Load depth as-is
        depth_scale = scene_camera[str_frame_id]['depth_scale']
        depth = depth.astype(np.float32) * depth_scale / 1000.0  # Convert to meters
        if depth is None:
            print(f"Failed to load depth image for frame {frame_id}")
            return None, None, None, None, None, None
        
        # Get camera matrix from scene_camera
        cam_info = scene_camera[str_frame_id]
        camera_matrix = np.array(cam_info['cam_K']).reshape(3, 3)

        # Find our object in the ground truth
        gt_found = False
        mask = None
        R = None
        t = None
        
        if str_frame_id in scene_gt:
            for gt_id, gt in enumerate(scene_gt[str_frame_id]):
                if gt['obj_id'] == object_id:
                    # Load mask
                    mask_file = os.path.join(scene_path, 'mask_visib', 
                                           f'{frame_id:06d}_{gt_id:06d}.png')
                    mask = cv2.imread(mask_file, -1)
                    
                    if mask is None:
                        print(f"Failed to load mask for frame {frame_id}, object {gt_id}")
                        continue

                    # Get pose from GT
                    R = np.array(gt['cam_R_m2c']).reshape(3, 3)
                    t = np.array(gt['cam_t_m2c']) / 1000.0  # Convert to meters
                    gt_found = True
                    break

        if not gt_found:
            return None, None, None, None, None, None
            
        return rgb, depth, mask, R, t, camera_matrix

    def process_frame(self, rgb, depth, mask, gt_R, gt_t, camera_matrix, detector):
        """Process a single frame and estimate pose."""
        # Start timing
        start_time = time.time()
        
        # Get matching points using the dataset mask
        img_pts, obj_pts = detector.match_image_points(rgb, mask)
        
        if img_pts is None or len(img_pts) < 4:
            return None, None, None, None, None, None, 0.0
            
        # Get 3D points from depth
        real_pts, valid_indices = detector.estimate3d(
            img_pts, depth, camera_matrix
        )
        
        # Filter points
        obj_pts = obj_pts[valid_indices]
        real_pts = real_pts[valid_indices]
        img_pts = img_pts[valid_indices]
        
        # Estimate pose
        R, t, inliers = detector.estimate_transform(
            real_pts=real_pts,
            obj_pts=obj_pts,
            img_pts=img_pts,
            camera_matrix=camera_matrix,
        )
        
        # Calculate total processing time
        process_time = time.time() - start_time
        score = 1.0 if R is not None else 0.0
        
        return R, t, inliers, img_pts, obj_pts, real_pts, score, process_time
    
    def visualize_all(self, rgb, mask, depth, img_pts, frame_id):
        # Create mask visualization
        mask_vis = np.zeros_like(rgb)
        mask_vis[mask > 0] = [0, 255, 0]  # Green mask

        rgb_overlay = rgb.copy()
        # Draw keypoints on the overlay
        if img_pts is not None and len(img_pts) > 0:
            for pt in img_pts:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(rgb_overlay, (x, y), 3, (0, 0, 255), -1)  # Red dots for keypoints

        # Normalize and colorize depth for visualization
        depth_vis = np.zeros_like(rgb)
        if depth is not None:
            depth_valid = depth > 0
            if depth_valid.sum() > 0:
                depth_norm = depth.copy()
                depth_norm[depth_valid] = (depth_norm[depth_valid] - depth_norm[depth_valid].min()) / \
                                        (depth_norm[depth_valid].max() - depth_norm[depth_valid].min())
                depth_vis = (depth_norm * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Add frame counter to RGB image
        cv2.putText(rgb_overlay, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        # Stack images horizontally
        vis_img = np.hstack([rgb_overlay, depth_vis, mask_vis])
        
        # Display
        cv2.imshow('BOP Dataset Viewer (RGB | Depth | Mask)', vis_img)
    
    def visualize_frame(self, frame, mask, R_pred, t_pred, R_gt, t_gt, camera_matrix, inliers=None):
        """Visualize pose estimation results."""
        # Create visualization image
        vis_img = frame.copy()
        
        # Draw mask overlay
        if mask is not None:
            mask_overlay = vis_img.copy()
            mask_overlay[mask > 0] = (0, 255, 0)
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_overlay, 0.3, 0)
        
        # Create transformation matrices
        T_pred = np.eye(4)
        T_pred[:3, :3] = R_pred
        T_pred[:3, 3] = t_pred
        
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = t_gt.flatten()
        
        # Compute errors
        rot_diff, trans_diff = get_transform_distance(T_gt, T_pred)
        rot_diff_deg = np.rad2deg(rot_diff)
        
        # Get RPY angles
        rpy_pred = rotation_to_rpy_camera(T_pred)
        rpy_gt = rotation_to_rpy_camera(T_gt)
        
        # Draw coordinate axes
        rvec_pred = cv2.Rodrigues(R_pred)[0]
        rvec_gt = cv2.Rodrigues(R_gt)[0]
        
        # Draw predicted pose in red
        cv2.drawFrameAxes(vis_img, camera_matrix, None, 
                         rvec_pred, t_pred, 0.05, 2)
        
        # Draw ground truth pose in green
        cv2.drawFrameAxes(vis_img, camera_matrix, None,
                         rvec_gt, t_gt, 0.05, 2)
        
        # Add error information
        info_text = [
            f"Rotation Error: {rot_diff_deg:.2f} deg",
            f"Translation Error: {trans_diff*1000:.1f} mm",
            f"Pred RPY: {np.rad2deg(rpy_pred[0]):.1f}, {np.rad2deg(rpy_pred[1]):.1f}, {np.rad2deg(rpy_pred[2]):.1f}",
            f"GT RPY: {np.rad2deg(rpy_gt[0]):.1f}, {np.rad2deg(rpy_gt[1]):.1f}, {np.rad2deg(rpy_gt[2]):.1f}",
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(vis_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            
        cv2.imshow(self.window_name, vis_img)
        
        return vis_img
        
    def evaluate_scene_range(self, start_scene, end_scene, visualize=False):
        """Evaluate a range of scenes."""
        for scene_id in range(start_scene, end_scene + 1):
            print(f"\nProcessing scene {scene_id}")
            self.evaluate_scene(scene_id, visualize)
            
    def evaluate_scene(self, scene_id, visualize=False):
        """Evaluate all objects in a scene."""
        scene_path = os.path.join(self.dataset_path, 'test', f'{scene_id:06d}')
        scene_gt = inout.load_scene_gt(os.path.join(scene_path, 'scene_gt.json'))
        
        if not scene_gt:
            print(f"No ground truth found for scene {scene_id}")
            return
            
        # Get unique object IDs in the scene
        object_ids = set()
        for frame_data in scene_gt.values():
            for gt in frame_data:
                object_ids.add(gt['obj_id'])
                
        print(f"Found {len(object_ids)} unique objects in scene {scene_id}")
        
        # Process each object
        for obj_id in object_ids:
            print(f"\nProcessing object {obj_id} in scene {scene_id}")
            self.evaluate_object_in_scene(scene_id, obj_id, visualize)
            
    def evaluate_object_in_scene(self, scene_id, object_id, visualize=False):
        """Evaluate a single object across all frames in a scene."""
        # Initialize object detector
        model_path = self.model_root / f"obj_{object_id:06d}.npz"
        if not model_path.exists():
            print(f"Model file not found for object {object_id}")
            return
            
        detector = Object3D(str(model_path))
        
        # Collect frame statistics
        angle_errors = []
        trans_errors = []
        total_frames = 0
        successful_estimates = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Process each frame
        scene_path = os.path.join(self.dataset_path, 'test', f'{scene_id:06d}')
        rgb_path = os.path.join(scene_path, 'rgb')
        frames = sorted([int(f.split('.')[0]) for f in os.listdir(rgb_path) if f.endswith('.png')])
        
        for frame_id in tqdm(frames, desc=f"Scene {scene_id} Object {object_id}"):
            rgb, depth, mask, gt_R, gt_t, camera_matrix = self.load_frame_data(
                scene_id, frame_id, object_id)
                
            if rgb is None:
                continue
                
            total_frames += 1
            
            # Process frame
            results = self.process_frame(
                rgb, depth, mask, gt_R, gt_t, camera_matrix, detector)
            
            if visualize:
                self.visualize_all(rgb, mask, depth, results[3], frame_id)
                
            if results[0] is not None:
                R_pred, t_pred, inliers = results[:3]
                successful_estimates += 1
                
                # Calculate errors
                T_pred = np.eye(4)
                T_pred[:3, :3] = R_pred
                T_pred[:3, 3] = t_pred
                
                T_gt = np.eye(4)
                T_gt[:3, :3] = gt_R
                T_gt[:3, 3] = gt_t.flatten()
                
                rot_diff, trans_diff = get_transform_distance(T_gt, T_pred)
                rot_diff_deg = np.rad2deg(rot_diff)
                
                angle_errors.append(rot_diff_deg)
                trans_errors.append(trans_diff * 1000)  # Convert to mm

                if visualize:
                    self.visualize_frame(rgb, mask, R_pred, t_pred, gt_R, gt_t, camera_matrix)
                
                # Update precision/recall metrics
                if rot_diff_deg < 10.0 and trans_diff < 0.05:  # 10 degrees and 5cm thresholds
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                false_negatives += 1
            
            cv2.waitKey(1)
                
        # Calculate statistics
        if angle_errors:
            mean_angle_error = np.mean(angle_errors)
            mean_trans_error = np.mean(trans_errors)
            std_angle_error = np.std(angle_errors)
            std_trans_error = np.std(trans_errors)
            percent_success = (successful_estimates / total_frames) * 100
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            self.statistics.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'mean_angle_error': mean_angle_error,
                'mean_trans_error': mean_trans_error,
                'std_angle_error': std_angle_error,
                'std_trans_error': std_trans_error,
                'percent_success_estimate': percent_success,
                'precision_rate': precision * 100,
                'recall_rate': recall * 100
            })

    def save_bop_results(self, output_path):
        """Save results in BOP challenge format CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
            
            for result in self.results:
                # Format rotation matrix: 9 values with 6 decimal places
                R_str = ' '.join([f"{x:.6f}" for x in result['R'].flatten()])
                
                # Format translation vector: 3 values with 6 decimal places
                t_str = ' '.join([f"{x:.6f}" for x in result['t'].flatten()])
                
                # Write row with exactly matching format
                writer.writerow([
                    result['scene_id'],
                    result['im_id'],
                    result['obj_id'],
                    f"{result['score']:.2f}",  # Score to 2 decimal places
                    R_str,
                    t_str,
                    f"{result['time']:.4f}"   # Time to 4 decimal places
                ])
            
    def save_statistics(self, output_path):
        """Save statistics to CSV file."""
        fieldnames = ['scene_id', 'object_id', 'mean_angle_error', 'mean_trans_error',
                     'std_angle_error', 'std_trans_error', 'percent_success_estimate',
                     'precision_rate', 'recall_rate']
                     
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stat in self.statistics:
                writer.writerow(stat)

def main():
    parser = argparse.ArgumentParser(description='YCB-Video Dataset Batch Evaluation')
    parser.add_argument('--dataset_path', required=True, help='Path to YCB-Video dataset root')
    parser.add_argument('--models', required=True, help='Path to YCB object models')
    parser.add_argument('--start_scene', type=int, required=True, help='Starting scene ID')
    parser.add_argument('--end_scene', type=int, required=True, help='Ending scene ID')
    parser.add_argument('--output', required=True, help='Path to save BOP format results')
    parser.add_argument('--stats_output', required=True, help='Path to save statistics CSV')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    
    args = parser.parse_args()
    
    evaluator = YCBBatchEvaluator(args.dataset_path, args.models)
    
    # Process all scenes
    evaluator.evaluate_scene_range(args.start_scene, args.end_scene, args.visualize)
    
    # Save results
    evaluator.save_bop_results(args.output)
    evaluator.save_statistics(args.stats_output)
    
    print(f"\nResults saved in BOP format to: {args.output}")
    print(f"Statistics saved to: {args.stats_output}")

if __name__ == "__main__":
    main()