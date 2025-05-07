# config.py
import os

# --- Robot Communication ---
ROBOT_IP = "10.35.129.5"  # Change this to your KUKA robot's IP address
# ROBOT_IP = "172.31.1.10"
PORT = 30000
BASE_URL = f"http://{ROBOT_IP}:{PORT}"

# --- KMR Physical Dimensions (Offsets in mm) ---
# Offsets for iiwa base relative to KMR center when KMR theta is 0
IIWA_BASE_X_OFFSET = 363
IIWA_BASE_Y_OFFSET = -184
IIWA_BASE_Z_OFFSET = 700 # Height of IIWA base from the floor/world origin Z

LONGEST_LENGTH_KMR = 1130
LONGEST_WIDTH_KMR = 630

# --- File Paths ---
# Ensure these paths are relative to your project root or use absolute paths
# Get the directory where this config file is located
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level if config.py is inside a 'communication' subdirectory
# _PROJECT_ROOT = os.path.dirname(_BASE_DIR) # Adjust if needed
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, os.pardir)) # Setting the project root to the parent directory of KMR_communication

IMAGE_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "image_processing")
COMMUNICATION_DIR = os.path.join(_PROJECT_ROOT, "communication") # If this structure exists
IMAGES_DIR = os.path.join(_PROJECT_ROOT, "images")

# Calibration and Pose Files
CAMERA_EXTRINSIC_FILE = os.path.join(IMAGE_PROCESSING_DIR, "calibration_data", "camera_extrinsic.json")
HAND_POSES_FILE = os.path.join(COMMUNICATION_DIR, "HandPoses.json") # Adjust path if needed
GO_AROUND_HAND_POSES_FILE = os.path.join(IMAGE_PROCESSING_DIR, "calibration_data", "GoAroundHandPoses.json") # Adjust path if needed

# Output Directories (can be overridden)
DEFAULT_GO_AROUND_OUTPUT_FOLDER = os.path.join(IMAGES_DIR, "GoAround")
DEFAULT_CALIBRATION_OUTPUT_FOLDER = os.path.join(IMAGES_DIR, "Calibration")
DEFAULT_PICK_OUTPUT_FOLDER = os.path.join(IMAGES_DIR, "JustPickIt")

# --- Image Processing Server ---
# Add server details if used by send_image_to_server/send_for_pose_estimation
DETECTION_SERVER_URL = "http://localhost:5001" # Example URL, replace if needed
POSE_ESTIMATION_SERVER_URL = "http://localhost:5002" # Example URL, replace if needed

# --- Other Constants ---
DEFAULT_ARM_SPEED = 0.2