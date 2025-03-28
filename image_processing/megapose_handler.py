import os
import json
import torch
import logging
from pathlib import Path
from PIL import Image
import sys

sys.path.append('../megapose6d/src/megapose/scripts')

import run_inference_on_example
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.lib3d.transform import Transform


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MegaPoseHandler:
    def __init__(self, device):
        self.device = device
        self.model_name = "megapose-1.0-RGB-multi-hypothesis"
        self.base_path = Path("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects")
        logging.info("MegaPoseHandler initialized.")

    def estimate_pose(self, image, object_name, bbox):
        """Estimate 6D pose for the given image and object name."""
        object_folder = self.base_path / object_name

        if not object_folder.exists():
            return {"error": f"Object folder {object_folder} not found"}

        # Ensure necessary folders exist
        os.makedirs(object_folder / "inputs", exist_ok=True)
        os.makedirs(object_folder / "outputs", exist_ok=True)

        # Save image
        image_path = object_folder / "image_rgb.png"
        image.save(image_path)

        # Save bounding box as object_data.json
        object_data = [{"label": object_name, "bbox_modal": bbox}]
        object_data_path = object_folder / "inputs" / "object_data.json"
        with open(object_data_path, "w") as f:
            json.dump(object_data, f)

        logging.info(f"Saved image and bounding box for {object_name} in {object_folder}.")

        print("Loading object data...")
        print("Object folder: ", object_folder)
        # Load required data
        image, depth, camera_data = run_inference_on_example.load_observation(object_folder, load_depth=False)
        object_data = run_inference_on_example.load_object_data(object_data_path)

        print("Running MegaPose inference...")
        # Run MegaPose inference
        output = run_inference_on_example.my_inference(image, depth, camera_data, object_data, self.model_name, object_folder)

        print("MegaPose inference finished.")
        # Save results
        out_path = object_folder / "outputs"
        run_inference_on_example.save_predictions(out_path, output)

        # Convert outputs to JSON-friendly format
        labels = output.infos["label"]
        poses = output.poses.cpu().numpy()
        object_data = [{"label": label, "pose": pose.tolist()} for label, pose in zip(labels, poses)]

        return {"poses": object_data}
