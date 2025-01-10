import numpy as np
import open3d as o3d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ShapeFitter:
    @staticmethod
    def fit_cylinder(points, initial_radius=0.1, initial_height=0.2):
        """
        Fit a cylinder to a point cloud using optimization.
        
        Args:
            points: Nx3 array of points
            initial_radius: Initial guess for cylinder radius
            initial_height: Initial guess for cylinder height
            
        Returns:
            dict with cylinder parameters (radius, height, center, direction)
        """
        # Convert points to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Get the principal axes using PCA
        covariance = np.cov(points.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        
        # Sort eigenvalues and eigenvectors
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # The principal axis is the eigenvector with the largest eigenvalue
        principal_axis = eigenvectors[:, 0]
        
        # Project points onto the principal axis to find the actual center
        points_centered = points - np.mean(points, axis=0)
        projections = np.dot(points_centered, principal_axis)
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # Find the center along the axis
        center_on_axis = np.mean(points, axis=0) + principal_axis * (min_proj + max_proj) / 2
        
        def cylinder_distance(params):
            """
            Calculate sum of squared distances from points to cylinder surface.
            params: [radius, cx, cy, cz]
            """
            radius, dx, dy, dz = params
            center = np.array([dx, dy, dz])
            
            # For each point, calculate distance to cylinder axis
            point_vectors = points - center
            
            # Project points onto cylinder axis
            projections = np.dot(point_vectors, principal_axis)[:, np.newaxis] * principal_axis
            
            # Get perpendicular vectors from axis to points
            perp_vectors = point_vectors - projections
            
            # Calculate distances from points to cylinder surface
            distances = np.abs(np.linalg.norm(perp_vectors, axis=1) - radius)
            
            return np.sum(distances**2)
        
        # Initial parameters [radius, cx, cy, cz]
        # Estimate initial radius from points perpendicular to axis
        points_perp = points_centered - np.dot(points_centered, principal_axis)[:, np.newaxis] * principal_axis
        initial_radius = np.mean(np.linalg.norm(points_perp, axis=1))
        
        initial_params = [
            initial_radius,
            center_on_axis[0], center_on_axis[1], center_on_axis[2]
        ]
        
        # Optimize cylinder parameters
        result = minimize(
            cylinder_distance,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if result.success:
            radius = result.x[0]
            center = result.x[1:4]
            
            # Compute height using projections
            point_vectors = points - center
            projections = np.dot(point_vectors, principal_axis)
            height = np.max(projections) - np.min(projections)
            
            # The center should be at the middle of the height
            center = center + principal_axis * (np.min(projections) + height/2)
            
            return {
                'radius': radius,
                'height': height,
                'center': center,
                'direction': principal_axis,
                'error': result.fun
            }
        else:
            return None

    @staticmethod
    def fit_sphere(points):
        """
        Fit a sphere to point cloud using least squares.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            dict with sphere parameters (radius, center)
        """
        # Convert points to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Initial guess for center is point cloud centroid
        center_initial = np.mean(points, axis=0)
        
        def sphere_distance(params):
            """Calculate sum of squared distances from points to sphere surface."""
            cx, cy, cz, radius = params
            center = np.array([cx, cy, cz])
            
            # Calculate distances from points to sphere center
            distances = np.linalg.norm(points - center, axis=1) - radius
            return np.sum(distances**2)
        
        # Initial parameters [cx, cy, cz, radius]
        initial_radius = np.mean(np.linalg.norm(points - center_initial, axis=1))
        initial_params = [
            center_initial[0], center_initial[1], center_initial[2],
            initial_radius
        ]
        
        # Optimize sphere parameters
        result = minimize(
            sphere_distance,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if result.success:
            return {
                'center': result.x[:3],
                'radius': result.x[3],
                'error': result.fun
            }
        else:
            return None

    @staticmethod
    def fit_cuboid(points):
        """
        Fit an oriented bounding box (cuboid) to a point cloud.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            dict with cuboid parameters (center, dimensions, rotation)
        """
        # Convert points to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Compute oriented bounding box
        obb = pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)  # Set to green
        
        # Get cuboid parameters
        center = obb.get_center()
        extent = obb.extent
        R = obb.R

        # Create a box mesh
        box_mesh = o3d.geometry.TriangleMesh.create_box(
            width=extent[0], 
            height=extent[1], 
            depth=extent[2]
        )
        
        # Move box center to origin before rotation
        box_mesh.translate((-extent[0]/2, -extent[1]/2, -extent[2]/2))
        
        # Transform the box mesh
        box_mesh.rotate(R, center=(0, 0, 0))
        box_mesh.translate(center)
        
        return {
            'center': np.asarray(center),
            'dimensions': np.asarray(extent),
            'rotation': np.asarray(R),
            'box_mesh': box_mesh,
            'oriented_box': obb
        }

    @staticmethod
    def visualize_fit(points, shape_params, shape_type='cylinder'):
        """
        Visualize point cloud and fitted shape.
        
        Args:
            points: Nx3 array of points
            shape_params: Parameters returned from fit_cylinder, fit_sphere, or fit_cuboid
            shape_type: 'cylinder', 'sphere', or 'cuboid'
        """
        # Create point cloud geometry
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1, 0, 0])  # Red points
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Configure render options
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0, 0, 0])  # black background
        render_option.point_size = 3.0  # larger points
        render_option.mesh_show_wireframe = True
        render_option.line_width = 2.0
        render_option.mesh_show_back_face = True
        
        geometries = []
        
        if shape_type == 'cylinder':
            # Create cylinder mesh
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=shape_params['radius'],
                height=shape_params['height']
            )
            
            # Center the cylinder at origin before rotation
            bbox = cylinder.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            cylinder.translate(-center)
            
            # Align cylinder with direction
            z_axis = np.array([0, 0, 1])
            direction = shape_params['direction']
            direction = direction / np.linalg.norm(direction)
            
            rotation_axis = np.cross(z_axis, direction)
            if np.any(rotation_axis):
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * rotation_angle
                )
                cylinder.rotate(R, center=(0, 0, 0))
            
            # Move to final position
            cylinder.translate(shape_params['center'])
            
            # Set semi-transparent green color
            cylinder.paint_uniform_color([0, 0.8, 0])  # lighter green for better transparency
            cylinder.compute_vertex_normals()
            geometries.append(cylinder)
            
        elif shape_type == 'sphere':
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=shape_params['radius']
            )
            sphere.translate(shape_params['center'])
            sphere.paint_uniform_color([0, 0.8, 0])  # lighter green for better transparency
            sphere.compute_vertex_normals()
            geometries.append(sphere)
            
        elif shape_type == 'cuboid':
            box_mesh = shape_params['box_mesh']
            box_mesh.paint_uniform_color([0, 0.8, 0])  # lighter green for better transparency
            box_mesh.compute_vertex_normals()
            geometries.append(box_mesh)

        # Add all geometries to visualizer
        for geom in geometries:
            vis.add_geometry(geom)

        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        
        # Add a light to better see the transparency
        vis.get_view_control().set_zoom(0.8)
        vis.get_view_control().set_lookat(shape_params['center'])
        
        # Run visualization
        vis.run()
        vis.destroy_window()

def example_shape_fitting():
    # Generate sample points for different shapes
    n_points = 1000
    noise_scale = 0.02
    
    def add_noise(points):
        return points + np.random.normal(0, noise_scale, points.shape)
    
    # Example 1: Cylinder
    print("\nFitting cylinder...")
    # Parameters
    n_points = 1000
    noise_scale = 0.02
    radius = 0.5
    height = 2.0

    # Generate points only for a partial view of the cylinder
    theta_min = -np.pi/2  # -45 degrees
    theta_max = np.pi/2   # 45 degrees
    theta = np.random.uniform(theta_min, theta_max, n_points)

    # Generate height values with some parts missing
    z_min = height * 0.2  # Start at 20% of height
    z_max = height * 0.8  # End at 80% of height
    z = np.random.uniform(z_min, z_max, n_points)

    # Generate x, y coordinates for cylinder surface
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Add points on the top and bottom circles (partial)
    n_circle_points = n_points // 4
    theta_circle = np.random.uniform(theta_min, theta_max, n_circle_points)
    r_circle = np.random.uniform(0, radius, n_circle_points)

    # Top circle points
    x_top = r_circle * np.cos(theta_circle)
    y_top = r_circle * np.sin(theta_circle)
    z_top = np.full_like(x_top, z_max)

    # Bottom circle points
    x_bottom = r_circle * np.cos(theta_circle)
    y_bottom = r_circle * np.sin(theta_circle)
    z_bottom = np.full_like(x_bottom, z_min)

    # Combine all points
    x = np.concatenate([x, x_top, x_bottom])
    y = np.concatenate([y, y_top, y_bottom])
    z = np.concatenate([z, z_top, z_bottom])

    # Add noise
    points = np.column_stack([x, y, z])
    cylinder_points = points + np.random.normal(0, noise_scale, points.shape)
    # Example 2: Cuboid
    print("\nFitting cuboid...")
    length, width, height = 1.0, 0.8, 1.5
    x = np.random.uniform(-length/2, length/2, n_points)
    y = np.random.uniform(-width/2, width/2, n_points)
    z = np.random.uniform(-height/2, height/2, n_points)
    cuboid_points = add_noise(np.column_stack([x, y, z]))
    
    # Fit shapes
    fitter = ShapeFitter()
    
    # Fit and visualize cylinder
    cylinder_params = fitter.fit_cylinder(cylinder_points)
    if cylinder_params:
        print("Cylinder fitting results:")
        print(f"Radius: {cylinder_params['radius']:.3f}")
        print(f"Height: {cylinder_params['height']:.3f}")
        print(f"Center: {cylinder_params['center']}")
        fitter.visualize_fit(cylinder_points, cylinder_params, 'cylinder')
    
    # Fit and visualize cuboid
    cuboid_params = fitter.fit_cuboid(cuboid_points)
    if cuboid_params:
        print("\nCuboid fitting results:")
        print(f"Dimensions: {cuboid_params['dimensions']}")
        print(f"Center: {cuboid_params['center']}")
        fitter.visualize_fit(cuboid_points, cuboid_params, 'cuboid')

if __name__ == "__main__":
    example_shape_fitting()