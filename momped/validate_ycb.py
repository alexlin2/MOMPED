import argparse
import numpy as np
import cv2
import json
import torch
import os
from pathlib import Path
from tqdm import tqdm
from object3d import Object3D
from utils import get_transform_distance, rotation_to_rpy_camera, visualize_alignment

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import visualization

class YCBEvaluator:
    def __init__(self, dataset_path, object_id, model_root):
        """
        Initialize YCB-Video dataset evaluator.
        
        Args:
            dataset_path: Path to YCB-Video dataset root
            object_id: Object ID to evaluate
            model_root: Path to YCB object models
        """
        self.dataset_path = dataset_path
        self.object_id = int(object_id)
        self.object_id_str = str(object_id).zfill(6)
        
        # Load model
        model_path = Path(model_root) / f"obj_{self.object_id_str}"
        
        # Initialize model and detector
        self.detector = Object3D(str(model_path) + ".npz")
        
        # For visualization
        self.window_name = "YCB-Video Pose Estimation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def load_frame_data(self, scene_id, frame_id):
        """Load RGB, depth, mask, and pose data for a frame."""
        # Scene paths using exact structure from ycb_loader.py
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
                if gt['obj_id'] == self.object_id:
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
            print(f"Object {self.object_id} not found in frame {frame_id}")
            return None, None, None, None, None, None
            
        return rgb, depth, mask, R, t, camera_matrix

    def process_frame(self, rgb, depth, mask, gt_R, gt_t, camera_matrix):
        """Process a single frame and estimate pose."""
        # Get matching points using the dataset mask
        img_pts, obj_pts = self.detector.match_image_points(rgb, mask)
        
        if img_pts is None or len(img_pts) < 4:
            return None, None, None, None, None, None
            
        # Get 3D points from depth
        real_pts, valid_indices = self.detector.estimate3d(
            img_pts, depth, camera_matrix
        )
        
        # Filter points
        obj_pts = obj_pts[valid_indices]
        real_pts = real_pts[valid_indices]
        img_pts = img_pts[valid_indices]
        
        # Estimate pose
        R, t, inliers = self.detector.estimate_transform(
            real_pts=real_pts,
            obj_pts=obj_pts,
            img_pts=img_pts,
            camera_matrix=camera_matrix,
        )
        
        return R, t, inliers, img_pts, obj_pts, real_pts

    def visualize_results(self, frame, mask, R_pred, t_pred, R_gt, t_gt, camera_matrix, inliers=None):
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
            f"Object ID: {self.object_id}",
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
            
        return vis_img

    def evaluate_sequence(self, scene_id, start_frame=1, end_frame=None):
        """Evaluate a sequence of frames."""
        errors = []
        
        # Get frame paths
        scene_path = os.path.join(self.dataset_path, 'test', f'{scene_id:06d}')
        rgb_path = os.path.join(scene_path, 'rgb')
        frame_paths = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
        
        if end_frame is None:
            end_frame = len(frame_paths)
            
        frame_indices = range(start_frame, end_frame)
        
        for frame_idx in tqdm(frame_indices, desc=f"Processing scene {scene_id}"):
            # Load frame data
            rgb, depth, mask, gt_R, gt_t, camera_matrix = self.load_frame_data(
                scene_id, frame_idx
            )
            
            if rgb is None:
                continue
                
            # Process frame
            R_pred, t_pred, inliers, img_pts, obj_pts, real_pts = self.process_frame(
                rgb, depth, mask, gt_R, gt_t, camera_matrix
            )

            vis_img = rgb.copy()
            if img_pts is not None:
                for pt in img_pts:
                    cv2.circle(vis_img, tuple(map(int, pt)), 5, (255, 0, 0), -1)
            if R_pred is not None:
                vis_img = self.visualize_results(
                    vis_img, mask, R_pred, t_pred, gt_R, gt_t, camera_matrix, inliers
                )
                #visualize_alignment(obj_pts, real_pts, R_pred, t_pred)
            else:
                print(f"Failed to estimate pose for frame {frame_idx}")
                
            # Visualize results
            
            cv2.imshow(self.window_name, vis_img)
            depth_vis = depth.copy()
            valid_mask = mask > 0.0
            valid_depths = depth[valid_mask]
            min_depth = valid_depths.min()
            max_depth = valid_depths.max()
            
            depth_vis[valid_mask] = ((depth_vis[valid_mask] - min_depth) / 
                                    (max_depth - min_depth) * 255)
            depth_vis = depth_vis.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            # Add min/max depth text overlay
            cv2.putText(depth_colored, f"Min depth: {min_depth:.3f}m", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colored, f"Max depth: {max_depth:.3f}m",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('depth', depth_colored)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            if R_pred is None:
                continue
            # Calculate errors
            T_pred = np.eye(4)
            T_pred[:3, :3] = R_pred
            T_pred[:3, 3] = t_pred
            
            T_gt = np.eye(4)
            T_gt[:3, :3] = gt_R
            T_gt[:3, 3] = gt_t.flatten()
            
            rot_diff, trans_diff = get_transform_distance(T_gt, T_pred)
            
            errors.append({
                'frame': frame_idx,
                'rotation_error_deg': np.rad2deg(rot_diff),
                'translation_error_mm': trans_diff * 1000,
            })
                
        return errors

def main():
    parser = argparse.ArgumentParser(description='YCB-Video Dataset Pose Estimation Evaluation')
    parser.add_argument('--dataset_path', required=True, help='Path to YCB-Video dataset root')
    parser.add_argument('--models', required=True, help='Path to YCB object models')
    parser.add_argument('--object_id', required=True, type=int, help='Object ID to evaluate')
    parser.add_argument('--scene_id', type=int, required=True, help='Scene ID to evaluate')
    parser.add_argument('--start', type=int, default=1, help='Start frame')
    parser.add_argument('--end', type=int, default=None, help='End frame')
    parser.add_argument('--output', help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    evaluator = YCBEvaluator(args.dataset_path, args.object_id, args.models)
    errors = evaluator.evaluate_sequence(args.scene_id, args.start, args.end)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(errors, f, indent=2)
            
        # Print summary statistics
        rot_errors = [e['rotation_error_deg'] for e in errors]
        trans_errors = [e['translation_error_mm'] for e in errors]
        
        print("\nEvaluation Summary:")
        print(f"Frames processed: {len(errors)}")
        print(f"Average rotation error: {np.mean(rot_errors):.2f}° ± {np.std(rot_errors):.2f}°")
        print(f"Average translation error: {np.mean(trans_errors):.1f} ± {np.std(trans_errors):.1f} mm")
        print(f"Median rotation error: {np.median(rot_errors):.2f}°")
        print(f"Median translation error: {np.median(trans_errors):.1f} mm")

if __name__ == "__main__":
    main()