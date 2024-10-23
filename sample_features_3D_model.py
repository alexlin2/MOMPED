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

class ModelViewer:
    def __init__(self, obj_path):
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

        # Initialize view parameters
        self.distance = 3.0
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

        self.init_renderer()

    def init_renderer(self):
        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=60.0
        )

        self.raster_settings = RasterizationSettings(
            image_size=1024, 
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
        self.distance = 3.0

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

    def detect_sift_features(self, nfeatures=1000, contrast_threshold=0.05, edge_threshold=10):
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
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector with balanced parameters
        sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrast_threshold,  
            edgeThreshold=edge_threshold,          
            sigma=1.6                             
        )
        
        # Find keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) == 0:
            print("No features detected")
            return None, None
        
        # Filter based on response strength - keep top 75%
        min_response = np.percentile([kp.response for kp in keypoints], 50)
        
        filtered_keypoints = []
        filtered_descriptors = []
        
        for kp, desc in zip(keypoints, descriptors):
            # Filter by response
            if kp.response < min_response:
                continue
                
            # Minimal size filtering
            if kp.size < 1.6:  # Minimum size threshold
                continue
                
            filtered_keypoints.append(kp)
            filtered_descriptors.append(desc)
            
        if not filtered_keypoints:
            print("No features passed filtering")
            return None, None
            
        filtered_descriptors = np.array(filtered_descriptors)
        print(f"Detected {len(filtered_keypoints)} filtered SIFT features")
        
        return filtered_keypoints, filtered_descriptors

    def find_matching_points(self, query_descriptors):
        """
        Find 3D points corresponding to query feature descriptors using basic FLANN matcher.
        Args:
            query_descriptors: numpy array of SIFT descriptors to match
        Returns:
            matches: list of (query_idx, stored_feature_info) tuples
        """
        matches = []
        
        if not self.feature_points or query_descriptors is None:
            return matches

        # Get stored descriptors
        stored_descriptors = np.array([f['descriptor'] for f in self.feature_points])
        
        if len(stored_descriptors) == 0:
            return matches

        # Convert to float32
        query_descriptors = query_descriptors.astype(np.float32)
        stored_descriptors = stored_descriptors.astype(np.float32)

        # Configure FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # Create FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Find k=2 nearest matches for each descriptor
        matches_flann = flann.knnMatch(query_descriptors, stored_descriptors, k=2)
        
        # Apply basic ratio test
        for i, (m, n) in enumerate(matches_flann):
            if m.distance < 0.75 * n.distance:
                matches.append((i, self.feature_points[m.trainIdx]))
        
        print(f"Found {len(matches)} matches from {len(matches_flann)} potential matches")
        return matches
    
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
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 25
        
        return img_with_keypoints

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
        
        print(f"Found {len(points3d)} valid features with 3D points")
        print(f"Total stored features: {len(self.feature_points)}")
        
        return np.array(points3d), np.array(valid_descriptors), valid_keypoints

    def display(self):
        WINDOW_NAME = '3D Model Viewer (PyTorch3D)'
        FEATURE_WINDOW = 'Feature Detection'
        
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
        print("D: Detect ORB features only")
        print("C: Compute 3D points for current features")
        print("Left Click: Sample 3D point")
        print("Q: Quit")

        last_keypoints = None
        last_descriptors = None
        
        while True:
            image = self.render_frame()
            cv2.imshow(WINDOW_NAME, image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('w'):
                self.distance = max(0.5, self.distance - 0.1)
            elif key == ord('s'):
                self.distance += 0.1
            elif key == 81:  # Left arrow
                self.azimuth -= 5
            elif key == 83:  # Right arrow
                self.azimuth += 5
            elif key == 82:  # Up arrow
                self.elevation = min(90, self.elevation + 5)
            elif key == 84:  # Down arrow
                self.elevation = max(-90, self.elevation - 5)
            elif key == ord('a'):
                self.yaw -= 5
            elif key == ord('d'):
                self.yaw += 5
            elif key == ord('r'):
                self.distance = 3.0
                self.elevation = 0.0
                self.azimuth = 0.0
                self.yaw = 0.0
            elif key == ord(' '):  # Space bar - detect and store SIFT features
                last_keypoints, last_descriptors = self.detect_sift_features()
                if last_keypoints is not None:
                    points3d, descriptors, keypoints = self.compute_feature_3d_points(
                        last_keypoints, last_descriptors)
                    if points3d is not None:
                        image_with_keypoints = self.visualize_features(keypoints)
                        cv2.imshow("Saved Features", image_with_keypoints)
            elif key == ord('m'):  # Match features
                last_keypoints, last_descriptors = self.detect_sift_features()
                if last_keypoints is not None and last_descriptors is not None:
                    matches = self.find_matching_points(last_descriptors)
                    if matches:
                        matched_indices = [m[0] for m in matches]
                        image_with_keypoints_matched = self.visualize_features(last_keypoints, matched_indices)
                        cv2.imshow("Matched Features", image_with_keypoints_matched)
                    else:
                        print("No matches found")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D Model Viewer using PyTorch3D')
    parser.add_argument('--obj', required=True, help='Path to the OBJ file')
    
    args = parser.parse_args()
    
    try:
        viewer = ModelViewer(args.obj)
        viewer.display()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()