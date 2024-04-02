new_dataset_info = {
    "joints": [
        {"name": "nose", "id": 0, "color": "#ff0000", "type": "End Site"},
        {"name": "left_eye", "id": 1, "color": "#00ff00", "type": "End Site"},
        {"name": "right_eye", "id": 2, "color": "#0000ff", "type": "End Site"},
        {"name": "left_ear", "id": 3, "color": "#ffff00", "type": "End Site"},
        {"name": "right_ear", "id": 4, "color": "#ff00ff", "type": "End Site"},
        {"name": "left_shoulder", "id": 5, "color": "#00ffff", "type": "Intermediate"},
        {"name": "right_shoulder", "id": 6, "color": "#ffffff", "type": "Intermediate"},
        {"name": "left_elbow", "id": 7, "color": "#ff8000", "type": "Intermediate"},
        {"name": "right_elbow", "id": 8, "color": "#ff0080", "type": "Intermediate"},
        {"name": "left_wrist", "id": 9, "color": "#80ff00", "type": "End Site"},
        {"name": "right_wrist", "id": 10, "color": "#00ff80", "type": "End Site"},
        {"name": "left_hip", "id": 11, "color": "#8000ff", "type": "Intermediate"},
        {"name": "right_hip", "id": 12, "color": "#0080ff", "type": "Intermediate"},
        {"name": "left_knee", "id": 13, "color": "#800080", "type": "Intermediate"},
        {"name": "right_knee", "id": 14, "color": "#808000", "type": "Intermediate"},
        {"name": "left_ankle", "id": 15, "color": "#408080", "type": "End Site"},
        {"name": "right_ankle", "id": 16, "color": "#804080", "type": "End Site"}
    ],
    "skeleton": [
        {"start": "nose", "end": "left_eye"},
        {"start": "nose", "end": "right_eye"},
        {"start": "left_eye", "end": "left_ear"},
        {"start": "right_eye", "end": "right_ear"},
        {"start": "left_shoulder", "end": "right_shoulder"},
        {"start": "left_shoulder", "end": "left_elbow"},
        {"start": "right_shoulder", "end": "right_elbow"},
        {"start": "left_elbow", "end": "left_wrist"},
        {"start": "right_elbow", "end": "right_wrist"},
        {"start": "left_shoulder", "end": "left_hip"},
        {"start": "right_shoulder", "end": "right_hip"},
        {"start": "left_hip", "end": "right_hip"},
        {"start": "left_hip", "end": "left_knee"},
        {"start": "right_hip", "end": "right_knee"},
        {"start": "left_knee", "end": "left_ankle"},
        {"start": "right_knee", "end": "right_ankle"}
    ]
}

# Generate keypoint information
keypoints = [joint["name"] for joint in new_dataset_info["joints"]]

# Generate skeleton information
# We need the joint IDs for start and end points to match the COCO format
joint_name_to_id = {joint["name"]: joint["id"] for joint in new_dataset_info["joints"]}
skeleton = [[joint_name_to_id[connection["start"]], joint_name_to_id[connection["end"]]] for connection in new_dataset_info["skeleton"]]

# Assuming uniform joint weights and sigmas for simplicity
# In real scenarios, these values should be determined based on the dataset characteristics
joint_weights = [1.0] * len(keypoints)  # Uniform weight for all joints
sigmas = [0.025] * len(keypoints)  # Uniform sigma for all joints

keypoints, skeleton, joint_weights, sigmas