import os
import sys
import subprocess
import numpy as np
from PIL import Image
from bg_removal import PreprocessInput
from pose_estimation import run_openpose, extract_keypoints
from clothmask import apply_cloth_mask

# Paths
INPUT_IMAGE = "inputs/image/work.png"
CLOTH_IMAGE = "inputs/cloth/printed_tshirt.jpg"
OUTPUT_FOLDER = "output/bgremove"
HUMAN_PARSE_OUTPUT = "output/human_parsed"
SCHP_SCRIPT = "Self-Correction-Human-Parsing/simple_extractor.py"
SCHP_CHECKPOINT = "checkpoints/final.pth"
MASK_OUTPUT_FOLDER = "output/masked_cloth"
POSE_JSON_OUTPUT = "output\pose_output"
TRYON_OUTPUT = "output/final_tryon"
MODEL_IMAGE_DIR = "inputs/image"  # Directory where model images are stored
CLOTH_IMAGE_DIR = "inputs/cloth"  # Directory where cloth images are stored
PAIRS_FILE = "inputs/test_pairs.txt"  # File to store model and cloth image pairs

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(HUMAN_PARSE_OUTPUT, exist_ok=True)
os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRYON_OUTPUT, exist_ok=True)

# -------------------------------
# 1️⃣ Background Removal
# -------------------------------
print("🚀 Removing Background...")
preprocessor = PreprocessInput()
img_no_bg = preprocessor.remove_background(INPUT_IMAGE)

if isinstance(img_no_bg, np.ndarray):
    print("✅ Background removal successful.")
    bg_removed_path = os.path.join(OUTPUT_FOLDER, "lower_model_no_bg.png")
    Image.fromarray(img_no_bg).save(bg_removed_path)
    print(f"✅ Image saved at {bg_removed_path}")
else:
    print("❌ Background removal failed.")
    sys.exit(1)

# -------------------------------
# 2️⃣ Run OpenPose
# -------------------------------
print("\n🚀 Running OpenPose...")

OPENPOSE_PATH = r"D:\ml\newtryon\openpose"

if not os.path.exists(OPENPOSE_PATH):
    print(f"❌ OpenPose executable not found at {OPENPOSE_PATH}")
    sys.exit(1)

json_files = run_openpose(OPENPOSE_PATH, OUTPUT_FOLDER, POSE_JSON_OUTPUT)

if json_files:
    keypoints = extract_keypoints(json_files)
    print("\n✅ Extracted Keypoints:", keypoints)
else:
    print("\n❌ OpenPose execution failed. No keypoints extracted.")
    sys.exit(1)

# -------------------------------
# 3️⃣ Apply Human Parsing (SCHP)
# -------------------------------
print("\n🚀 Running Human Parsing (SCHP)...")

if not os.path.exists(SCHP_SCRIPT):
    print(f"❌ SCHP script not found at {SCHP_SCRIPT}")
    sys.exit(1)

schp_command = [
    sys.executable, SCHP_SCRIPT,
    "--dataset", "lip",
    "--model-restore", SCHP_CHECKPOINT,
    "--input-dir", OUTPUT_FOLDER,
    "--output-dir", HUMAN_PARSE_OUTPUT
]

try:
    subprocess.run(schp_command, check=True)
    print(f"✅ Human Parsing completed. Output saved at {HUMAN_PARSE_OUTPUT}")
except subprocess.CalledProcessError as e:
    print("❌ Human Parsing failed:", e)
    sys.exit(1)

# -------------------------------
# 4️⃣ Apply Clothing Segmentation Mask
# -------------------------------
print("\n🚀 Applying Clothing Mask...")

masked_cloth_path = apply_cloth_mask(CLOTH_IMAGE, MASK_OUTPUT_FOLDER)

if masked_cloth_path:
    print(f"✅ Cloth mask applied successfully. Output saved at {masked_cloth_path}")
else:
    print("❌ Cloth mask application failed.")
    sys.exit(1)

# -------------------------------
# 5️⃣ Pair Model and Cloth Images
# -------------------------------
print("\n🚀 Pairing Model and Cloth Images...")

model_image_files = os.listdir(MODEL_IMAGE_DIR)
cloth_image_files = os.listdir(CLOTH_IMAGE_DIR)

if len(model_image_files) != len(cloth_image_files):
    print(f"❌ Number of model images ({len(model_image_files)}) does not match the number of cloth images ({len(cloth_image_files)})")
    sys.exit(1)

# Pairing the images and saving to a text file
with open(PAIRS_FILE, 'w') as file:
    for model_image, cloth_image in zip(model_image_files, cloth_image_files):
        file.write(f"{model_image} {cloth_image}\n")

print(f"✅ Pairing completed. Pairs saved to {PAIRS_FILE}")

# -------------------------------
# 6️⃣ Making Predictions (Running test.py)
# -------------------------------
print("\n🚀 Running Predictions (test.py)...")

TEST_SCRIPT = "test.py"
CHECKPOINT_DIR = "checkpoints"
SAVE_DIR = "output"

# Ensure the required directories exist
pose_output_json = os.path.join(POSE_JSON_OUTPUT, "lower_model_no_bg_keypoints.json")
human_parse_output_img = os.path.join(HUMAN_PARSE_OUTPUT, "lower_model_no_bg.png")
masked_cloth_img = os.path.join(MASK_OUTPUT_FOLDER, "printed_tshirt.png")

print("\n🔍 Final verification before try-on:")
print(f"Model image exists: {os.path.exists(INPUT_IMAGE)}")
print(f"Cloth image exists: {os.path.exists(CLOTH_IMAGE)}")
print(f"Pose JSON exists: {os.path.exists(pose_output_json)}")
print(f"Human parse exists: {os.path.exists(human_parse_output_img)}")
print(f"Cloth mask exists: {os.path.exists(masked_cloth_img)}")

# Prepare required arguments for test.py
test_command = [
    sys.executable, TEST_SCRIPT,
    "--name", "final_tryon",  # Explicit output folder name
    "--dataset_dir", os.path.abspath("inputs"),  # Use absolute paths
    "--checkpoint_dir", os.path.abspath(CHECKPOINT_DIR),
    "--save_dir", os.path.abspath(SAVE_DIR),
    "--display_freq", "1",  # Show progress every batch
]

# Check if the paths for the required files exist
for file_path in [pose_output_json, human_parse_output_img, masked_cloth_img]:
    if not os.path.exists(file_path):
        print(f"❌ Required file not found: {file_path}")
        sys.exit(1)

try:
    subprocess.run(test_command, check=True)
    print(f"✅ Prediction completed. Output saved at {TRYON_OUTPUT}")
except subprocess.CalledProcessError as e:
    print("❌ Prediction failed:", e)
    sys.exit(1)
