import torch
import numpy as np
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
)
import math

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import detect_sift_features, find_matching_points

class ModelViewer:
    def __init__(self, obj_path, auto_scan=False, scan_distance=6.0):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        print("Loading OBJ file...")
        self.meshes = load_objs_as_meshes([obj_path], device=self.device)
        
        if self.meshes.isempty():
            raise ValueError("Failed to load mesh!")

        print(f"Loaded mesh with {self.meshes.verts_packed().shape[0]} vertices")
        print(f"Number of faces: {self.meshes.faces_packed().shape[0]}")

        self.auto_scan = auto_scan
        # Initialize view parameters
        self.scan_distance = scan_distance
        self.distance = self.scan_distance
        self.elevation = 0.0
        self.azimuth = 0.0
        self.yaw = 0.0

        # Store last render data
        self.last_cameras = None
        self.last_fragments = None
        self.current_image = None

        # Initialize feature storage
        self.feature_points = []  # List of dictionaries containing feature info
        self.total_features = 0   # Counter for total features stored
        self.stored_3d_points = []

        self.init_renderer()

    def init_renderer(self):
        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=60.0
        )

        self.raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None
        )

        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, -0.3]],
            ambient_color=[[1.0, 1.0, 1.0]],
        )

        # Create separate rasterizer and shader for access to fragments
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, 
            raster_settings=self.raster_settings
        )

        self.shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=self.lights
        )

        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def auto_scale_and_center(self):
        verts = self.meshes.verts_packed()
        center = verts.mean(dim=0)
        scale = verts.abs().max().item()
        
        self.meshes.offset_verts_(-center)
        self.meshes.scale_verts_((1.0 / scale))

    def get_rotation_matrix(self):
        elevation_rad = math.radians(self.elevation)
        azimuth_rad = math.radians(self.azimuth)
        yaw_rad = math.radians(self.yaw)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(elevation_rad), -math.sin(elevation_rad)],
            [0, math.sin(elevation_rad), math.cos(elevation_rad)]
        ], device=self.device)

        Ry = torch.tensor([
            [math.cos(azimuth_rad), 0, math.sin(azimuth_rad)],
            [0, 1, 0],
            [-math.sin(azimuth_rad), 0, math.cos(azimuth_rad)]
        ], device=self.device)

        Rz = torch.tensor([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ], device=self.device)

        R = Rz @ Ry @ Rx
        return R.unsqueeze(0)

    def get_point_3d(self, x, y):
        """
        Get 3D point from 2D pixel coordinates.
        Args:
            x (int): pixel x coordinate
            y (int): pixel y coordinate
        Returns:
            point3d (torch.Tensor): 3D point in world coordinates, or None if no point found
        """
        if self.last_fragments is None:
            return None

        # Get depth and face index for the selected pixel
        depth = self.last_fragments.zbuf[0, y, x, 0].item()
        face_idx = self.last_fragments.pix_to_face[0, y, x, 0].item()

        # Check if we hit a face
        if face_idx < 0 or depth == -1:
            return None

        # Get barycentric coordinates
        barycentric = self.last_fragments.bary_coords[0, y, x, 0]
        
        # Get vertex indices for the face
        face_vertices = self.meshes.faces_packed()[face_idx]
        
        # Get the actual vertices
        vertices = self.meshes.verts_packed()[face_vertices]
        
        # Compute 3D point using barycentric coordinates
        point3d = (barycentric.unsqueeze(0) @ vertices).squeeze(0)

        return point3d.cpu().numpy()

    def render_frame(self):
        R = self.get_rotation_matrix()
        T = torch.tensor([[0, 0, self.distance]], device=self.device)

        R_xz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ], device=self.device).unsqueeze(0)

        # Combine the rotations
        R = R @ R_xz

        self.last_cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T
        )

        # Get fragments from rasterizer
        meshes_batch = self.meshes.extend(1)
        fragments = self.rasterizer(meshes_batch, cameras=self.last_cameras)
        self.last_fragments = fragments

        # Render using shader
        images = self.shader(fragments, meshes_batch, cameras=self.last_cameras)

        image = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        self.current_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.current_image

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point3d = self.get_point_3d(x, y)
            if point3d is not None:
                print(f"3D point at pixel ({x}, {y}): {point3d}")
            else:
                print(f"No point found at pixel ({x}, {y})")
    
    def visualize_features(self, keypoints, matched_indices=None):
        """
        Visualize features with additional information.
        """
        img_with_keypoints = self.current_image.copy()
        
        if matched_indices is None:
            cv2.drawKeypoints(self.current_image, keypoints, img_with_keypoints, 
                            color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            # Draw unmatched keypoints in green
            unmatched_kp = [kp for i, kp in enumerate(keypoints) if i not in matched_indices]
            cv2.drawKeypoints(self.current_image, unmatched_kp, img_with_keypoints, 
                            color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Draw matched keypoints in red
            matched_kp = [kp for i, kp in enumerate(keypoints) if i in matched_indices]
            cv2.drawKeypoints(img_with_keypoints, matched_kp, img_with_keypoints, 
                            color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Add detailed information
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
    
    def visualize_3d_points(self, current_points):

        # Create new figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get mesh vertices
        verts = self.meshes.verts_packed().cpu().numpy()
        
        # Plot subsampled mesh vertices
        stride = max(1, len(verts) // 1000)  # Subsample vertices
        ax.scatter(verts[::stride, 0], 
                  verts[::stride, 1], 
                  verts[::stride, 2], 
                  c='gray', alpha=0.2, s=1, label='Mesh')

        # Plot stored points if any exist
        if self.stored_3d_points:
            stored_points = np.array(self.stored_3d_points)
            ax.scatter(stored_points[:, 0], 
                      stored_points[:, 1], 
                      stored_points[:, 2],
                      c='green', s=20, 
                      alpha=0.5,
                      label=f'Stored ({len(stored_points)})')

        # Plot current points
        ax.scatter(current_points[:, 0], 
                  current_points[:, 1], 
                  current_points[:, 2],
                  c='red', s=50, 
                  label=f'Current ({len(current_points)})')

        
        ax.set_box_aspect([1,1,1])

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Visualization')
        
        # Add legend
        ax.legend()
        
        # Show the plot
        plt.show()

    def decluster_points(self, voxel_size=0.05):
        """
        Decluster points using a voxel grid approach.
        Args:
            voxel_size: Size of voxels for declustering
        """
        if not self.stored_3d_points or not self.feature_points:
            print("No features to decluster")
            return

        print(f"\nDeclustering with voxel size: {voxel_size}")
        print(f"Initial points: {len(self.stored_3d_points)}")

        points = np.array(self.stored_3d_points)

        # Create voxel grid indices
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Create dictionary to store points and corresponding features for each voxel
        voxel_dict = {}
        
        for idx, (point, feature) in enumerate(zip(points, self.feature_points)):
            voxel_idx = tuple(voxel_indices[idx])
            if voxel_idx not in voxel_dict:
                voxel_dict[voxel_idx] = []
            voxel_dict[voxel_idx].append((point, feature))

        # Keep strongest feature in each voxel
        new_points = []
        new_features = []
        
        for voxel_points in voxel_dict.values():
            # Sort points in this voxel by response strength
            voxel_points.sort(key=lambda x: x[1]['response'], reverse=True)
            # Keep the strongest point
            best_point, best_feature = voxel_points[0]
            new_points.append(best_point)
            new_features.append(best_feature)

        # Update stored points and features
        self.stored_3d_points = new_points
        self.feature_points = new_features

        print(f"Points after declustering: {len(self.stored_3d_points)}")
        print(f"Reduction ratio: {len(self.stored_3d_points)/len(points):.2f}")

        # Visualize the declustered points
        self.visualize_3d_points(np.array([[0.0, 0.0, 0.0]]))
        
    def compute_feature_3d_points(self, keypoints, descriptors):
        """
        Compute 3D points for given keypoints and store feature information.
        Args:
            keypoints: List of keypoints
            descriptors: Numpy array of descriptors
        Returns:
            points3d: List of 3D points corresponding to valid features
            valid_descriptors: List of descriptors for valid features
            valid_keypoints: List of keypoints for valid features
        """
        if keypoints is None or descriptors is None:
            return None, None, None

        points3d = []
        valid_descriptors = []
        valid_keypoints = []
        new_features = []
        
        for kp, desc in zip(keypoints, descriptors):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            point3d = self.get_point_3d(x, y)
            
            if point3d is not None:
                points3d.append(point3d)
                valid_descriptors.append(desc)
                valid_keypoints.append(kp)
                
                # Store feature information
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
        
        if not points3d:
            print("No 3D points found for features")
            return None, None, None
        
        # Add new features to storage
        self.feature_points.extend(new_features)
        self.stored_3d_points.extend(points3d)
        
        # print(f"Found {len(points3d)} valid features with 3D points")
        # print(f"Total stored features: {len(self.feature_points)}")
        
        return np.array(points3d), np.array(valid_descriptors), valid_keypoints
    
    def grid_capture(self, elevation_range=(-45, 45), elevation_steps=5,
                         yaw_range=(-45, 45), yaw_steps=5,
                         save_path="grid_features"):
        """
        Systematically capture features from a grid of viewpoints.
        Args:
            elevation_range: (min, max) elevation angles in degrees
            elevation_steps: number of elevation steps
            yaw_range: (min, max) yaw angles in degrees
            yaw_steps: number of yaw steps
            save_path: path to save features
        """
        print("\nStarting systematic capture...")
        print(f"Elevation range: {elevation_range}, steps: {elevation_steps}")
        print(f"Yaw range: {yaw_range}, steps: {yaw_steps}")

        # Clear existing features
        self.feature_points = []
        self.stored_3d_points = []
        self.total_features = 0

        # Create grid of viewpoints
        elevations = np.arange(elevation_range[0], elevation_range[1], elevation_steps)
        yaws = np.arange(yaw_range[0], yaw_range[1], yaw_steps)

        total_views = len(elevations) * len(yaws)
        view_count = 0

        # Store original view parameters
        orig_elevation = self.elevation
        orig_yaw = self.yaw

        try:
            for elevation in elevations:
                for yaw in yaws:
                    view_count += 1
                    print(f"\nProcessing view {view_count}/{total_views}")
                    print(f"Elevation: {elevation:.1f}, Yaw: {yaw:.1f}")

                    # Set view parameters
                    self.elevation = elevation
                    self.yaw = yaw

                    # Render the view
                    image = self.render_frame()
                    
                    # Detect and store features
                    keypoints, descriptors = detect_sift_features(image)
                    if keypoints is not None:
                        points3d, descriptors, keypoints = self.compute_feature_3d_points(
                            keypoints, descriptors)
                        if points3d is not None:
                            image_with_keypoints = self.visualize_features(keypoints)
                            cv2.imshow("Saved Features", image_with_keypoints)
                            print(f"Added {len(points3d)} features from this view")

                    # Optional: show current view
                    cv2.imshow("Grid Sampling", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow early termination
                        raise KeyboardInterrupt

            # After capturing all views
            print("\nCapture complete!")
            print(f"Total features collected: {len(self.feature_points)}")
            
            # Optional: decluster points
            if len(self.feature_points) > 0:
                print("\nDeclustering points...")
                self.decluster_points()
                
                # Save features
                print(f"\nSaving features to {save_path}")
                self.save_feature_points(save_path)

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")

        finally:
            # Restore original view
            self.elevation = orig_elevation
            self.yaw = orig_yaw
            self.render_frame()
    
    def save_feature_points(self, filepath):
        """
        Save feature points to a .npz file
        Args:
            filepath: Path to save the feature points (without extension)
        """
        # Convert feature points to saveable format
        save_data = []
        for feat in self.feature_points:
            # Convert keypoint to its basic attributes
            kp = feat['keypoint']
            keypoint_data = {
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            }
            
            # Create saveable feature dict
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
        
        # Save to file
        np.savez_compressed(
            filepath,
            feature_points=save_data,
            stored_3d_points=np.array(self.stored_3d_points),
            total_features=self.total_features
        )
        print(f"Saved {len(self.feature_points)} features to {filepath}.npz")

    def load_feature_points(self, filepath):
        """
        Load feature points from a .npz file
        Args:
            filepath: Path to the saved feature points (without extension)
        """
        # Load the compressed file
        data = np.load(f"{filepath}.npz", allow_pickle=True)
        
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
                'point3d': feat_dict['point3d'],
                'keypoint': keypoint,
                'response': feat_dict['response'],
                'size': feat_dict['size'],
                'angle': feat_dict['angle'],
                'octave': feat_dict['octave']
            }
            self.feature_points.append(feature)
        
        # Load 3D points and total features
        self.stored_3d_points = data['stored_3d_points'].tolist()
        self.total_features = int(data['total_features'])
        
        print(f"Loaded {len(self.feature_points)} features from {filepath}.npz")
        
        # Visualize loaded points
        self.visualize_3d_points(np.array([[0.0, 0.0, 0.0]]))

    def display(self):
        WINDOW_NAME = '3D Model Viewer (PyTorch3D)'
        
        self.auto_scale_and_center()
        
        cv2.imshow(WINDOW_NAME, np.zeros((1024, 1024, 3), dtype=np.uint8))
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        
        print("\nControls:")
        print("Left/Right Arrow: Rotate azimuth (Y-axis)")
        print("Up/Down Arrow: Rotate elevation (X-axis)")
        print("A/D: Rotate yaw (Z-axis)")
        print("W/S: Zoom in/out")
        print("R: Reset view")
        print("Space: Detect and store ORB features")
        print("M: Match features with last detected")
        print("Left Click: Sample 3D point")
        print("Q: Quit")

        last_keypoints = None
        last_descriptors = None

        if self.auto_scan:
            self.grid_capture(
                elevation_range=(-25, 25),
                elevation_steps=20,
                yaw_range=(0, 360),
                yaw_steps=5,
                save_path="grid_features"
            )

        while True:
            image = self.render_frame()
            cv2.imshow(WINDOW_NAME, image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('w'):
                self.distance = max(0.5, self.distance - 0.1)
                print(f"Distance: {self.distance}")
            elif key == ord('s'):
                self.distance += 0.1
                print(f"Distance: {self.distance}")
            elif key == 81:  # Left arrow
                self.azimuth -= 5
                print(f"Azimuth: {self.azimuth}")
            elif key == 83:  # Right arrow
                self.azimuth += 5
                print(f"Azimuth: {self.azimuth}")
            elif key == 82:  # Up arrow
                self.elevation = min(90, self.elevation + 5)
                print(f"Elevation: {self.elevation}")
            elif key == 84:  # Down arrow
                self.elevation = max(-90, self.elevation - 5)
                print(f"Elevation: {self.elevation}")
            elif key == ord('a'):
                self.yaw -= 5
                print(f"Yaw: {self.yaw}")
            elif key == ord('d'):
                self.yaw += 5
                print(f"Yaw: {self.yaw}")
            elif key == ord('r'):
                self.distance = self.scan_distance
                self.elevation = 0.0
                self.azimuth = 0.0
                self.yaw = 0.0
            elif key == ord('c'):  # Save feature points
                self.save_feature_points("grid_features")
            elif key == ord('v'):  # Load feature points
                self.load_feature_points("grid_features")
            elif key == ord('o'):  # Filter and keep best features
                self.decluster_points()
            elif key == ord(' '):  # Space bar - detect and store SIFT features
                last_keypoints, last_descriptors = detect_sift_features(image)
                if last_keypoints is not None:
                    points3d, descriptors, keypoints = self.compute_feature_3d_points(
                        last_keypoints, last_descriptors)
                    if points3d is not None:
                        image_with_keypoints = self.visualize_features(keypoints)
                        cv2.imshow("Saved Features", image_with_keypoints)
            elif key == ord('m'):  # Match features
                keypoints, descriptors = detect_sift_features(image)
                if keypoints is not None and descriptors is not None:
                    matches = find_matching_points(self.feature_points, descriptors)
                    if matches:
                        matched_indices = [m[0] for m in matches]
                        image_with_keypoints_matched = self.visualize_features(keypoints, matched_indices)
                        self.visualize_3d_points(np.array([m[1]['point3d'] for m in matches]))
                        cv2.imshow("Matched Features", image_with_keypoints_matched)
                    else:
                        print("No matches found")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D Model Viewer using PyTorch3D')
    parser.add_argument('--obj', required=True, help='Path to the OBJ file')
    parser.add_argument('--auto_scan', action='store_true', help='Automatically scan the object and save the feature points')
    parser.add_argument('--scan_distance', type=float, default=6.0, help='Distance to scan the object')
    
    args = parser.parse_args()
    
    try:
        viewer = ModelViewer(args.obj, args.auto_scan, args.scan_distance)
        viewer.display()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()