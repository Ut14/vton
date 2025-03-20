import os
import glob
import json
import os
import subprocess

# OpenPose body parts mapping
BODY_PARTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
}


def run_openpose(openpose_path, input_folder, output_json):
    os.makedirs(output_json, exist_ok=True)

    openpose_exe = os.path.join(openpose_path, "bin", "OpenPoseDemo.exe")
    model_folder = os.path.join(openpose_path, "models")  # Fix path

    if not os.path.exists(openpose_exe):
        print(f"❌ OpenPose executable not found: {openpose_exe}")
        return []

    command = [
        openpose_exe,
        "--image_dir", input_folder,
        "--write_json", output_json,
        "--render_pose", "1",
        "--display", "0",
        "--model_folder", model_folder,  # Corrected path
        "--write_images", os.path.join(output_json, "pose_images")
    ]

    print("Executing OpenPose command:\n", " ".join(command))
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"⚠ OpenPose execution failed: {e.stderr}")
        return []

    return glob.glob(os.path.join(output_json, "*.json"))



def extract_keypoints(json_files):
    """
    Extracts body keypoints from the first available OpenPose JSON output.
    """
    if not json_files:
        print("❌ No JSON files provided to extract keypoints.")
        return None

    json_file = json_files[0]  # Pick the first JSON file

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return None

    keypoints = {}
    if "people" in data and len(data["people"]) > 0:
        person = data["people"][0]  # Get first detected person
        body_keypoints = person["pose_keypoints_2d"]

        # Convert list to dictionary format
        for i, part in BODY_PARTS.items():
            keypoints[part] = (body_keypoints[i * 3], body_keypoints[i * 3 + 1])  # x, y coordinates

    return keypoints
