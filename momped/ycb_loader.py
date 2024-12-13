import os
import sys
import numpy as np
import cv2
import json
import argparse
from pathlib import Path

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import visualization

def load_scene_camera(scene_path):
    """Load camera parameters for the scene."""
    camera_path = os.path.join(scene_path, 'scene_camera.json')
    if os.path.exists(camera_path):
        return inout.load_json(camera_path)
    return None

def load_scene_gt(scene_path):
    """Load ground truth annotations for the scene."""
    gt_path = os.path.join(scene_path, 'scene_gt.json')
    if os.path.exists(gt_path):
        return inout.load_json(gt_path)
    return None

def main():
    parser = argparse.ArgumentParser(description='Simple BOP YCB-Video Dataset Visualization')
    parser.add_argument('--dataset_path', required=True, help='Path to YCB-Video BOP dataset')
    parser.add_argument('--scene_id', type=int, required=True, help='Scene ID to visualize')
    parser.add_argument('--object_id', type=int, help='Specific object ID to track (optional)')
    args = parser.parse_args()

    # Scene paths
    scene_path = os.path.join(args.dataset_path, 'test', f'{args.scene_id:06d}')
    rgb_path = os.path.join(scene_path, 'rgb')
    depth_path = os.path.join(scene_path, 'depth')
    
    # Load scene info
    scene_camera = load_scene_camera(scene_path)
    scene_gt = load_scene_gt(scene_path)

    if scene_camera is None:
        print("Error: Could not load scene camera parameters")
        return
    
    # Get sorted list of frames
    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
    frame_ids = [int(f.split('.')[0]) for f in rgb_files]

    print("\nViewing dataset...")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  Left/Right Arrow: Previous/Next frame when paused")
    print("  Q: Quit")

    frame_idx = 0
    paused = False

    while True:
        frame_id = frame_ids[frame_idx]
        str_frame_id = str(frame_id)

        # Load RGB image
        rgb = cv2.imread(os.path.join(rgb_path, f'{frame_id:06d}.png'))
        
        # Load depth image
        depth_path_frame = os.path.join(depth_path, f'{frame_id:06d}.png')
        if os.path.exists(depth_path_frame):
            depth = cv2.imread(depth_path_frame, -1)  # Load depth as-is
            depth_scale = scene_camera[str_frame_id]['depth_scale']
            depth = depth * depth_scale
        else:
            depth = None

        # Create mask visualization
        mask_vis = np.zeros_like(rgb)
        
        # Draw masks for all objects (or specific object if specified)
        if scene_gt and str_frame_id in scene_gt:
            for gt_id, gt in enumerate(scene_gt[str_frame_id]):
                if args.object_id is not None and gt['obj_id'] != args.object_id:
                    continue
                    
                mask_path = os.path.join(scene_path, 'mask_visib', f'{frame_id:06d}_{gt_id:06d}.png')
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, -1)
                    mask_vis[mask > 0] = [0, 255, 0]  # Green mask

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
        cv2.putText(rgb, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        # Stack images horizontally
        vis_img = np.hstack([rgb, depth_vis, mask_vis])
        
        # Display
        cv2.imshow('BOP Dataset Viewer (RGB | Depth | Mask)', vis_img)
        
        # Handle keyboard input
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 83:  # Right arrow
            frame_idx = min(frame_idx + 1, len(frame_ids) - 1)
        elif key == 81:  # Left arrow
            frame_idx = max(frame_idx - 1, 0)
            
        # Advance frame if not paused
        if not paused:
            frame_idx += 1
            if frame_idx >= len(frame_ids):
                frame_idx = 0

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()