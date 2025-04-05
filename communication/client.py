# communication/client.py
import requests
import os
import cv2
import numpy as np
import json # Needed for bbox formatting
from pathlib import Path
import logging

# --- Configuration ---
# Use absolute paths or paths relative to the project root
# Assuming client.py is in KMR_communication/communication
_CURRENT_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CURRENT_FILE_DIR.parent.parent

# Use server URLs from config if possible, or hardcode defaults
try:
     # If running with python -m communication.client, relative should work
     from . import config
     SERVER_URL_BASE = config.DETECTION_SERVER_URL # Assuming detection and pose are same base URL
except ImportError:
     print("Warning: Running client standalone. Using hardcoded URLs.")
     SERVER_URL_BASE = "http://localhost:5000" # Default

SERVER_URL_DETECTION = f"{SERVER_URL_BASE}/process"
SERVER_URL_POSE = f"{SERVER_URL_BASE}/estimate_pose"

# --- Test Data ---
# Adjust these paths and prompts for your testing scenario
DEFAULT_INPUT_FOLDER = PROJECT_ROOT / "images" / "JustPickIt"
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "images" / "JustPickIt" / "output_client_test"
DEFAULT_PROMPT = "foam brick" # Object type for detection
DEFAULT_OBJECT_NAME_POSE = "foam_brick" # Object name for MegaPose (must match folder)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - STANDALONE_CLIENT - %(levelname)s - %(message)s')

# Run me with: python -m communication.client
# Or: python KMR_communication/communication/client.py (if run from project root)

def send_detection_request(image_data_bytes, prompt):
    """ Sends image to detection server """
    logging.info(f"Sending detection request to {SERVER_URL_DETECTION} for prompt: '{prompt}'")
    try:
        response = requests.post(
            SERVER_URL_DETECTION,
            files={"image": ("image.png", image_data_bytes, "image/png")},
            data={"prompt": prompt},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        logging.info(f"Detection Response: {result}")
        return result.get("bounding_boxes", []) # Return list of boxes or empty list
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed detection request: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in detection request: {e}", exc_info=True)
        return []


def send_pose_request(image_data_bytes, object_name, bounding_box, visualize=False):
    """ Sends image and bbox to the pose estimation server """
    logging.info(f"Sending pose request to {SERVER_URL_POSE} for object: '{object_name}', Visualize: {visualize}")
    try:
        bbox_json_string = json.dumps(bounding_box)
        response = requests.post(
            SERVER_URL_POSE,
            files={"image": ("image.png", image_data_bytes, "image/png")},
            data={
                "object_name": object_name,
                "bbox": bbox_json_string,
                "visualize": str(visualize).lower()
            },
            timeout=240 # Long timeout for pose estimation
        )
        response.raise_for_status()
        result = response.json()
        logging.info(f"Pose Estimation Response: {result}")
        return result # Return the full dictionary (contains 'poses' or 'error')
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed pose estimation request: {e}")
        return {"error": f"Connection error: {e}"}
    except Exception as e:
        logging.error(f"Unexpected error in pose estimation request: {e}", exc_info=True)
        return {"error": f"Client-side error: {e}"}


def process_images_in_folder(input_folder=DEFAULT_INPUT_FOLDER,
                             output_folder=DEFAULT_OUTPUT_FOLDER,
                             detection_prompt=DEFAULT_PROMPT,
                             pose_object_name=DEFAULT_OBJECT_NAME_POSE,
                             run_pose_estimation=True,
                             request_visualization=False):
    """ Loads images, detects objects, optionally estimates pose, saves annotated images. """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".png", ".jpeg")]
    logging.info(f"Found {len(image_files)} images in {input_path}")

    all_pose_results = {}

    for image_file_path in image_files:
        logging.info(f"\n--- Processing image: {image_file_path.name} ---")
        try:
            # Read image bytes
            with open(image_file_path, "rb") as f:
                image_data_bytes = f.read()
            # Load image for drawing later
            image_cv = cv2.imread(str(image_file_path))
            if image_cv is None:
                 logging.warning(f"Could not read image file {image_file_path.name}, skipping.")
                 continue

            # 1. Run Object Detection
            bounding_boxes = send_detection_request(image_data_bytes, detection_prompt)

            if not bounding_boxes:
                logging.warning(f"'{detection_prompt}' not found in image: {image_file_path.name}")
                # Save original image to output if nothing found
                cv2.imwrite(str(output_path / image_file_path.name), image_cv)
                continue

            logging.info(f"'{detection_prompt}' found in {image_file_path.name}. BBoxes: {bounding_boxes}")

            # Draw bounding boxes
            for (x1, y1, x2, y2) in bounding_boxes:
                cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 2. Optionally Run 6D Pose Estimation (using the first bounding box)
            pose_result = None
            if run_pose_estimation and bounding_boxes:
                first_bbox = bounding_boxes[0]
                logging.info(f"Requesting 6D Pose Estimation for '{pose_object_name}' using bbox: {first_bbox}")
                pose_result = send_pose_request(image_data_bytes, pose_object_name, first_bbox, visualize=request_visualization)

                if pose_result and "error" not in pose_result:
                    logging.info(f"Pose Estimation SUCCESSFUL for {image_file_path.name}")
                    all_pose_results[image_file_path.name] = pose_result # Store successful results
                    # Optionally draw pose info? (Complex)
                elif pose_result: # Error dictionary returned
                    logging.error(f"Pose Estimation FAILED for {image_file_path.name}: {pose_result.get('error', 'Unknown error')}")
                else: # Should not happen if server returns error dict
                     logging.error(f"Pose Estimation FAILED for {image_file_path.name}: No response or unexpected return.")

            # Save image with annotations (bounding boxes)
            output_image_path = output_path / image_file_path.name
            cv2.imwrite(str(output_image_path), image_cv)
            logging.info(f"Annotated image saved to: {output_image_path}")

        except Exception as e:
            logging.error(f"Failed to process image {image_file_path.name}: {e}", exc_info=True)

    # Optionally save all collected pose results
    if all_pose_results:
         pose_summary_path = output_path / "_pose_estimation_summary.json"
         with open(pose_summary_path, 'w') as f_summary:
              json.dump(all_pose_results, f_summary, indent=4)
         logging.info(f"Summary of successful pose estimations saved to: {pose_summary_path}")

if __name__ == "__main__":
    process_images_in_folder(
        run_pose_estimation=True,
        request_visualization=False # Set to True to test visualization generation
    )