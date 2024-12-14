#!/bin/bash

# Set base paths
MOMPED_PATH="/home/alexlin/pose_estimation/momped"
DATASET_PATH="/home/alexlin/dataset/ycbv"
MODEL_PATH="${DATASET_PATH}/models"

# Ensure output directories exist
mkdir -p "${MODEL_PATH}"

# Function to process a single object
process_object() {
    local obj_id=$1
    local obj_id_padded=$(printf "%06d" $obj_id)
    
    echo "Processing object ${obj_id_padded}..."
    
    # Construct paths
    local model_file="${MODEL_PATH}/obj_${obj_id_padded}.ply"
    local output_file="${MODEL_PATH}/obj_${obj_id_padded}.npz"
    
    # Check if model file exists
    if [ ! -f "$model_file" ]; then
        echo "Warning: Model file not found: ${model_file}"
        return 1
    fi
    
    # Run feature capture
    python "${MOMPED_PATH}/model3d.py" \
        --obj "${model_file}" \
        --auto_scan \
        --scan_distance 4.0 \
        --features_path "${output_file}"
    
    # Check if feature capture was successful
    if [ $? -eq 0 ]; then
        echo "Successfully captured features for object ${obj_id_padded}"
    else
        echo "Error capturing features for object ${obj_id_padded}"
    fi
}

# Main processing loop
echo "Starting feature capture for YCB-Video objects..."

for obj_id in $(seq 1 21); do
    process_object $obj_id
    
    # Add a small delay between objects
    sleep 2
done

echo "Feature capture complete!"