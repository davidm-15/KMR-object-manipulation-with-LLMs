import sys

sys.path.append('../megapose6d/src/megapose/scripts')

import run_inference_on_example
from pathlib import Path
from typing import List
import argparse
import json
from megapose.datasets.scene_dataset import ObjectData
from megapose.lib3d.transform import Transform
from image_processing.megapose_handler import MegaPoseHandler
import os
import torch
from communication.server import process_image, estimate_pose, initialize_handlers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

from transformers import AutoConfig
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent # Go up one level
if str(PROJECT_ROOT) not in sys.path:
    print(f"Adding project root to sys.path: {PROJECT_ROOT}")
    sys.path.append(str(PROJECT_ROOT))
# --- ---

try:
    # Import the functions needed
    from communication.server import initialize_handlers, process_image, estimate_pose_via_subprocess
except ImportError as e:
    print(f"Failed to import from communication.server: {e}")
    print("Check sys.path and ensure server.py is accessible.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# run me with python -m image_processing.Megapose_testing

def main():
    logging.info("Initializing object detector...")
    # Initialize only the detection model handler (e.g., glamm, grounding_dino)
    # initialize_handlers now only returns this one handler.
    model_handler = initialize_handlers("glamm") # Or whichever detector you use
    logging.info("Object detector initialized.")

    # Define image path and prompt
    image_file = "images/JustPickIt/img.png" # Make sure this path is correct relative to where you run the script
    prompt = "plug-in outlet expander" # This will be used for detection AND as the object_name for pose

    image_path = Path(image_file)
    if not image_path.is_file():
        logging.error(f"Input image file not found: {image_path.resolve()}")
        return

    logging.info(f"Running object detection for '{prompt}' on {image_path}...")
    # Run object detection - process_image expects a path string
    detection_result = process_image(model_handler, str(image_path), prompt)

    # Check detection results
    if not detection_result or "bounding_boxes" not in detection_result or not detection_result["bounding_boxes"]:
        logging.error(f"Object detection failed or found no objects for '{prompt}'. Result: {detection_result}")
        return
    logging.info(f"Detection result: {detection_result}")

    # Get the first bounding box
    bbox = detection_result["bounding_boxes"][0]
    logging.info(f"Using bounding box: {bbox}")

    # --- Call the subprocess-based pose estimation ---
    logging.info(f"Running pose estimation for '{prompt}' via subprocess...")
    # Pass the image *path* (as string) directly to the modified function
    # Pass visualize=True instead of DoVis=True
    estimate_pose_result = estimate_pose_via_subprocess(
        image_input=str(image_path), # Pass the string path
        object_name=prompt,          # Use the prompt as object name
        bbox=bbox,
        visualize=True               # Use 'visualize' argument
    )
    # --- ---

    logging.info("Pose estimation subprocess finished.")
    print("-------------------------------------")
    print("Final Pose Result from subprocess:")
    # Pretty print the JSON result if possible
    import json
    try:
        print(json.dumps(estimate_pose_result, indent=4))
    except Exception:
        print(estimate_pose_result) # Print as is if JSON fails
    print("-------------------------------------")



def test_megapose():
    model_name = "megapose-1.0-RGB-multi-hypothesis"
    example_dir = Path("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/mustard bottle")
    image, depth, camera_data = run_inference_on_example.load_observation(example_dir, load_depth=False)
    object_data = run_inference_on_example.load_object_data(example_dir / "inputs" / "object_data.json")



    output = run_inference_on_example.my_inference(image, depth, camera_data, object_data, model_name, example_dir)

    # print("Poses:", output.poses)
    # print("Poses Input:", output.poses_input)
    # print("K_crop:", output.K_crop)
    # print("K:", output.K)
    # print("Boxes Rend:", output.boxes_rend)
    # print("Boxes Crop:", output.boxes_crop)
    # print("Infos:", output.infos)

    out_path = Path("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/MustardBottle/6DPose")
    out_path = example_dir
    run_inference_on_example.save_predictions(out_path, output)


    labels = output.infos["label"]
    poses = output.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])



    run_inference_on_example.my_visualization(example_dir, out_path / "visualizations")

if __name__ == "__main__":
    # test_megapose()
    main()