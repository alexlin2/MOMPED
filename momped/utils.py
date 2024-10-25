import numpy as np
import cv2

def detect_sift_features(image):
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector with balanced parameters
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        print("No features detected")
        return None, None
    
    # Filter based on response strength - keep top 75%
    min_response = np.percentile([kp.response for kp in keypoints], 25)
    
    filtered_keypoints = []
    filtered_descriptors = []
    
    for kp, desc in zip(keypoints, descriptors):
        # Filter by response
        if kp.response < min_response:
            continue
            
        # Minimal size filtering
        if kp.size < 2.0:  # Minimum size threshold
            continue
            
        filtered_keypoints.append(kp)
        filtered_descriptors.append(desc)
        
    if not filtered_keypoints:
        print("No features passed filtering")
        return None, None
        
    filtered_descriptors = np.array(filtered_descriptors)
    print(f"Detected {len(filtered_keypoints)} filtered SIFT features")
    
    return filtered_keypoints, filtered_descriptors

def find_matching_points(feature_points, query_descriptors):
    """
    Find 3D points corresponding to query feature descriptors using basic FLANN matcher.
    Args:
        query_descriptors: numpy array of SIFT descriptors to match
    Returns:
        matches: list of (query_idx, stored_feature_info) tuples
    """
    matches = []
    
    if feature_points is None or query_descriptors is None:
        return matches

    # Get stored descriptors
    stored_descriptors = np.array([f['descriptor'] for f in feature_points])
    
    if len(stored_descriptors) == 0:
        return matches

    # Convert to float32
    query_descriptors = query_descriptors.astype(np.float32)
    stored_descriptors = stored_descriptors.astype(np.float32)

    # Configure FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=16)
    search_params = dict(checks=100)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find k=2 nearest matches for each descriptor
    matches_flann = flann.knnMatch(query_descriptors, stored_descriptors, k=2)
    
    # Apply basic ratio test
    for i, (m, n) in enumerate(matches_flann):
        if m.distance < 0.9 * n.distance:
            matches.append((i, feature_points[m.trainIdx]))
    
    print(f"Found {len(matches)} matches from {len(matches_flann)} potential matches")
    return matches