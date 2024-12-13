import torch
import numpy as np
import cv2
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesUV,
    TexturesVertex
)
import matplotlib
matplotlib.use('TkAgg')
import trimesh

from utils import (
    detect_sift_features, 
    find_matching_points,
    FeatureManager,
    Visualization,
    ModelUtils
)

class Model3D:
    def __init__(self, model_path, auto_scan=False, scan_distance=6.0):
        """
        Initialize Model3D with support for PLY and OBJ files, and texture mapping.
        
        Args:
            model_path: Path to the model file (.ply or .obj)
            auto_scan: Whether to automatically scan the object
            scan_distance: Distance for scanning/viewing
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")

        print("Loading model file...")
        self.meshes = self.load_model(model_path)
        
        if self.meshes.isempty():
            raise ValueError("Failed to load mesh!")

        print(f"Loaded mesh with {self.meshes.verts_packed().shape[0]} vertices")
        print(f"Number of faces: {self.meshes.faces_packed().shape[0]}")

        self.auto_scan = auto_scan
        self.scan_distance = scan_distance
        self.distance = self.scan_distance
        self.elevation = 0.0
        self.azimuth = 0.0
        self.yaw = 0.0

        # Store rendering data
        self.last_fragments = None
        self.current_image = None
        
        # Initialize feature manager
        self.feature_manager = FeatureManager()
        self.init_renderer()

    def load_model(self, model_path):
        """
        Load model from PLY or OBJ file with optional texture mapping.
        
        Args:
            model_path: Path to model file (.ply or .obj)
            texture_path: Optional path to texture image (.png)
            
        Returns:
            Meshes object with loaded model
        """
        if model_path.lower().endswith('.ply'):
            # Load PLY file
            verts, faces = load_ply(model_path)
            mesh_trimesh = trimesh.load(model_path)

            # Move vertices and faces to specified device
            verts = verts.to(self.device)
            faces = faces.to(self.device)

            texture_path = model_path.replace('.ply', '.png')
            
            # Load and process texture image
            texture_image = cv2.imread(texture_path)
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
            texture_image = torch.from_numpy(texture_image).float().to(self.device) / 255.0

            # Get UV coordinates
            verts_uvs = torch.tensor(mesh_trimesh.visual.uv, dtype=torch.float32, device=self.device)
            
            # YCB-Video dataset uses face-vertex UV format, so we need to create the faces_uvs
            faces_uvs = faces.clone()

            # Create TexturesUV
            textures = TexturesUV(
                maps=[texture_image],
                faces_uvs=[faces_uvs],
                verts_uvs=[verts_uvs]
            )

            # Create mesh with vertices, faces and textures
            meshes = Meshes(
                verts=[verts.to(self.device)],
                faces=[faces.to(self.device)],
                textures=textures
            )

        else:  # Assume OBJ file
            meshes = load_objs_as_meshes([model_path], device=self.device)

        return meshes

    def init_renderer(self):
        """Initialize the PyTorch3D renderer with the current settings."""
        self.cameras = FoVPerspectiveCameras(device=self.device, fov=60.0)
        
        self.raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, -0.3]],
            ambient_color=[[1.0, 1.0, 1.0]],
        )

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

    def get_point_3d(self, x, y):
        """Get 3D point from 2D pixel coordinates."""
        if self.last_fragments is None:
            return None

        depth = self.last_fragments.zbuf[0, y, x, 0].item()
        face_idx = self.last_fragments.pix_to_face[0, y, x, 0].item()

        if face_idx < 0 or depth == -1:
            return None

        barycentric = self.last_fragments.bary_coords[0, y, x, 0]
        face_vertices = self.meshes.faces_packed()[face_idx]
        vertices = self.meshes.verts_packed()[face_vertices]
        point3d = (barycentric.unsqueeze(0) @ vertices).squeeze(0)

        return point3d.cpu().numpy()

    def render_frame(self):
        """Render current frame."""
        R = ModelUtils.get_rotation_matrix(self.elevation, self.azimuth, self.yaw, self.device)
        T = torch.tensor([[0, 0, self.distance]], device=self.device)
        
        R_xz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ], device=self.device).unsqueeze(0)

        R = R @ R_xz
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        meshes_batch = self.meshes.extend(1)
        self.last_fragments = self.rasterizer(meshes_batch, cameras=cameras)
        images = self.shader(self.last_fragments, meshes_batch, cameras=cameras)

        image = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        image[np.all(image == [255, 255, 255], axis=-1)] = [0, 0, 0]
        self.current_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.current_image
    
    def get_depth_image(self):
        """Get depth image from the last rendered frame."""
        if self.last_fragments is None:
            return None, None

        depth = self.last_fragments.zbuf[0, ..., 0]  
        depth_np = depth.cpu().numpy()
        valid_mask = depth_np != -1
        
        if not valid_mask.any():
            return None, None

        depth_vis = depth_np.copy()
        valid_depths = depth_vis[valid_mask]
        min_depth = valid_depths.min()
        max_depth = valid_depths.max()
        
        depth_vis[valid_mask] = ((depth_vis[valid_mask] - min_depth) / 
                                (max_depth - min_depth) * 255)
        depth_vis[~valid_mask] = 0
        depth_vis = depth_vis.astype(np.uint8)
        
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.putText(depth_colored, 
                   f"Depth range: {min_depth:.2f} to {max_depth:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return depth_colored, depth_np
    
    def get_camera_transforms(self):
        """Get the camera transformation matrices."""
        R = ModelUtils.get_rotation_matrix(self.elevation, self.azimuth, self.yaw, self.device)
        T = torch.tensor([[0, 0, self.distance]], device=self.device)
        
        R_xz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ], device=self.device).unsqueeze(0)
        
        R = R @ R_xz
        R = R.squeeze().cpu().numpy()
        T = T.squeeze().cpu().numpy()
        
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = R
        world_to_camera[:3, 3] = -R @ T
        
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = R.T
        camera_to_world[:3, 3] = T

        return world_to_camera, camera_to_world
    
    def transform_to_ros_coord_system(self, T):
        
        R, t = T[:3, :3], T[:3, 3]

        R_x = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        R_y = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])

        R = R_x @ R
        t = R_x @ t
        R = R_y @ R
        t = R_y @ t

        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def get_camera_intrinsics(self):
        """Get camera intrinsic parameters in OpenCV format."""
        width = height = self.raster_settings.image_size
        fov_degrees = 60.0
        
        fov_radians = np.deg2rad(fov_degrees)
        focal_length_pixels = (width/2) / np.tan(fov_radians/2)
        
        cx = width / 2
        cy = height / 2
        
        K = np.array([
            [focal_length_pixels, 0, cx],
            [0, focal_length_pixels, cy],
            [0, 0, 1]
        ])
        
        return K, width, height

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            point3d = self.get_point_3d(x, y)
            if point3d is not None:
                print(f"3D point at pixel ({x}, {y}): {point3d}")
            else:
                print(f"No point found at pixel ({x}, {y})")

    def compute_feature_3d_points(self, keypoints, descriptors):
        """Compute 3D points for given keypoints."""
        if keypoints is None or descriptors is None:
            return None, None, None

        points3d = []
        valid_descriptors = []
        valid_keypoints = []
        
        for kp, desc in zip(keypoints, descriptors):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            point3d = self.get_point_3d(x, y)
            
            if point3d is not None:
                points3d.append(point3d)
                valid_descriptors.append(desc)
                valid_keypoints.append(kp)
        
        if not points3d:
            return None, None, None
        
        points3d = np.array(points3d)
        self.feature_manager.add_features(valid_keypoints, valid_descriptors, points3d)
        
        return points3d, np.array(valid_descriptors), valid_keypoints

    def grid_capture(self, elevation_range=(-60, 60), elevation_steps=10,
                    yaw_range=(-45, 45), yaw_steps=5,
                    save_path="grid_features"):
        """Systematically capture features from a grid of viewpoints."""
        print("\nStarting systematic capture...")
        
        self.feature_manager = FeatureManager()
        elevations = np.arange(elevation_range[0], elevation_range[1], elevation_steps)
        yaws = np.arange(yaw_range[0], yaw_range[1], yaw_steps)

        total_views = len(elevations) * len(yaws)
        view_count = 0

        orig_elevation = self.elevation
        orig_yaw = self.yaw

        try:
            for elevation in elevations:
                for yaw in yaws:
                    view_count += 1
                    print(f"\nProcessing view {view_count}/{total_views}")
                    print(f"Elevation: {elevation:.1f}, Yaw: {yaw:.1f}")

                    self.elevation = elevation
                    self.yaw = yaw
                    image = self.render_frame()
                    
                    keypoints, descriptors = detect_sift_features(image)
                    if keypoints is not None:
                        points3d, descriptors, keypoints = self.compute_feature_3d_points(
                            keypoints, descriptors)
                        if points3d is not None:
                            image_with_keypoints = Visualization.draw_features(image, keypoints)
                            cv2.imshow("Saved Features", image_with_keypoints)

                    cv2.imshow("Grid Sampling", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

            print("\nCapture complete!")
            print(f"Total features collected: {len(self.feature_manager.feature_points)}")
            
            if len(self.feature_manager.feature_points) > 0:
                print("\nDeclustering points...")
                self.feature_manager.decluster_points()
                print(f"\nSaving features to {save_path}")
                self.feature_manager.save_features(save_path)

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")

        finally:
            self.elevation = orig_elevation
            self.yaw = orig_yaw
            self.render_frame()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D Model Viewer using PyTorch3D')
    parser.add_argument('--obj', required=True, help='Path to the OBJ file')
    parser.add_argument('--auto_scan', action='store_true', help='Automatically scan the object')
    parser.add_argument('--scan_distance', type=float, default=6.0, help='Distance to scan')
    parser.add_argument('--features_path', type=str, default=None, help='Path to save/load features')
    
    args = parser.parse_args()


    
    try:
        viewer = Model3D(args.obj, args.auto_scan, args.scan_distance)
        
        WINDOW_NAME = '3D Model Viewer (PyTorch3D)'
        DEPTH_WINDOW = 'Depth View'
        
        ModelUtils.auto_scale_and_center(viewer.meshes)

        cv2.imshow(WINDOW_NAME, np.zeros((1024, 1024, 3), dtype=np.uint8))
        cv2.setMouseCallback(WINDOW_NAME, viewer.mouse_callback)
        
        print("\nControls:")
        print("Left/Right Arrow: Rotate azimuth (Y-axis)")
        print("Up/Down Arrow: Rotate elevation (X-axis)")
        print("A/D: Rotate yaw (Z-axis)")
        print("W/S: Zoom in/out")
        print("R: Reset view")
        print("Space: Detect and store SIFT features")
        print("M: Match features with last detected")
        print("Left Click: Sample 3D point")
        print("Q: Quit")

        if viewer.auto_scan:
            viewer.grid_capture(
                elevation_range=(-60, 60),
                elevation_steps=20,
                yaw_range=(0, 360),
                yaw_steps=20,
                save_path=args.features_path
            )

        camera_matrix, width, height = viewer.get_camera_intrinsics()

        while True:
            image = viewer.render_frame()
            depth_colored, raw_depth = viewer.get_depth_image()
            _, camera_to_world = viewer.get_camera_transforms()

            ros_camera_to_world = viewer.transform_to_ros_coord_system(camera_to_world)

            rvec = cv2.Rodrigues(ros_camera_to_world[:3, :3])[0]
            tvec = ros_camera_to_world[:3, 3]
            cv2.drawFrameAxes(image, camera_matrix, None, rvec, tvec, 0.5)
            cv2.imshow(DEPTH_WINDOW, depth_colored)
            cv2.imshow(WINDOW_NAME, image)
            
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
            elif key == ord('c'):
                viewer.feature_manager.save_features(args.features_path)
            elif key == ord('v'):
                viewer.feature_manager.load_features(args.features_path)
            elif key == ord('o'):
                viewer.feature_manager.decluster_points()
            elif key == ord(' '):
                keypoints, descriptors = detect_sift_features(image)
                if keypoints is not None:
                    points3d, descriptors, keypoints = viewer.compute_feature_3d_points(
                        keypoints, descriptors)
                    if points3d is not None:
                        image_with_keypoints = Visualization.draw_features(keypoints)
                        cv2.imshow("Saved Features", image_with_keypoints)
            elif key == ord('m'):
                keypoints, descriptors = detect_sift_features(image)
                if keypoints is not None and descriptors is not None:
                    matches = find_matching_points(viewer.feature_manager.feature_points, descriptors)
                    if matches:
                        matched_indices = [m[0] for m in matches]
                        image_with_keypoints_matched = Visualization.draw_features(
                            image, keypoints, matched_indices)
                        Visualization.plot_3d_points(
                            verts=viewer.meshes.verts_packed().cpu().numpy(),
                            stored_points=viewer.feature_manager.stored_3d_points,
                            current_points=np.array([m[1]['point3d'] for m in matches])
                        )
                        cv2.imshow("Matched Features", image_with_keypoints_matched)
                    else:
                        print("No matches found")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()