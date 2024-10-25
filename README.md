# MOMPED: Multi-Object Multimodal Pose Estimation and Detection

<img src="assets/mom_on_a_moped.webp" alt="MOMPED" width="300"/>

MOMPED is a Python library for 3D object pose estimation using multimodal data (RGB-D and 3D models). It provides tools for:
- Feature extraction from 3D models
- RGB-D based 3D point estimation
- 2D-3D feature matching
- Robust pose estimation

## Features

- **3D Model Feature Sampling**
  - Interactive visualization of 3D models
  - Automatic feature sampling from multiple viewpoints
  - SIFT feature detection and description
  - Storage of 3D feature points and descriptors

- **Object Detection and Pose Estimation**
  - RGB-D based 3D point estimation
  - Feature matching between live views and stored models
  - Robust pose estimation with RANSAC
  - Visual feedback and error metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/alexlin2/momped.git
cd momped

# Install dependencies
pip install numpy opencv-python matplotlib pytorch3d
```

## Usage

### 1. Sample Features from 3D Model
```bash
python sample_features_3D_model.py --obj path/to/model.obj --auto_scan
```
Options:
- `--obj`: Path to the OBJ file
- `--auto_scan`: Automatically scan the object from multiple viewpoints
- `--scan_distance`: Distance for scanning (default: 6.0)

### 2. Detect Objects and Estimate Pose
```python
from momped import Object3D

# Initialize object with stored features
obj = Object3D("path/to/features.npz")

# Match with new image and estimate pose
img_pts, obj_pts = obj.match_image_points(rgb_image, mask)
real_pts, valid_indices = obj.estimate3d(img_pts, depth_image, camera_matrix)

# Estimate transformation
R, t = obj.estimate_transform(obj_pts[valid_indices], real_pts)
```

## File Structure

- `object3d.py`: Main class for object detection and pose estimation
- `sample_features_3D_model.py`: Tool for sampling features from 3D models
- `utils.py`: Utility functions for feature detection and matching

## Requirements

- Python 3.6+
- PyTorch3D
- OpenCV
- NumPy
- Matplotlib

## Example Output

The library provides various visualization tools:
- 3D feature point visualization
- Feature matching visualization
- Pose estimation results
- Error metrics and alignment quality

## Contributing

Feel free to open issues or submit pull requests.

## License

MIT License

Copyright (c) 2024 Alex Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

If you use this work in your research, please cite:
```
@software{momped2024,
  author = {Alex Lin},
  title = {MOMPED: Multi-Object Multimodal Pose Estimation and Detection},
  year = {2024},
  url = {https://github.com/alexlin2/momped}
}
```