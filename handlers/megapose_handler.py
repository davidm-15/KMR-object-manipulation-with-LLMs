# image_processing/megapose_handler.py
import os
import json
import torch
import logging
from pathlib import Path
from PIL import Image
import sys
import numpy as np

sys.path.append('../megapose6d/src/megapose/scripts')

import run_inference_on_example
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.lib3d.transform import Transform
from PIL import ImageDraw
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# python -m groma.eval.run_groma     --model-name FoundationVision/groma-7b-finetune     --image-file ../KMR-object-manipulation-with-LLMs/images/ScannedObjects/ScannedObjects/Stationary/ScanObjects_5/image_1743174825.png     --query "mustard bottle"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6  python -m groma.eval.run_groma     --model-name FoundationVision/groma-7b-finetune     --image-file ../KMR-object-manipulation-with-LLMs/images/ScannedObjects/ScannedObjects/Stationary/ScanObjects_5/image_1743174825.png     --query "mustard bottle"

class MegaPoseHandler:
    def __init__(self, device, path):
        self.device = device
        self.model_name = "megapose-1.0-RGB-multi-hypothesis"
        self.base_path = Path(path)
        logging.info("MegaPoseHandler initialized.")


    def load_observation(self, folder, load_depth=False):
        intrinsic_path = Path("image_processing/calibration_data/camera_intrinsics.json")
        camera_data = CameraData.from_json((intrinsic_path).read_text())

        rgb = np.array(Image.open(folder / "image_rgb.png"), dtype=np.uint8)
        assert rgb.shape[:2] == camera_data.resolution

        depth = None
        if load_depth:
            depth = np.array(Image.open(folder / "image_depth.png"), dtype=np.float32) / 1000
            assert depth.shape[:2] == camera_data.resolution

        return rgb, depth, camera_data

    def load_object_data(self, path):
        object_data = json.loads(path.read_text())
        object_data = [ObjectData.from_json(d) for d in object_data]
        return object_data



    def estimate_pose(self, image, object_name, bbox, **kwargs):
        """Estimate 6D pose for the given image and object name."""
        DoVis = kwargs.get("DoVis", False)
        Depth = kwargs.get("Depth", None)

        object_folder = self.base_path / object_name

        if not object_folder.exists():
            return {"error": f"Object folder {object_folder} not found"}

        # Ensure necessary folders exist
        os.makedirs(object_folder / "inputs", exist_ok=True)
        os.makedirs(object_folder / "outputs", exist_ok=True)

        # Save image
        image_path = object_folder / "image_rgb.png"
        image.save(image_path)

        time = datetime.now().strftime("%H_%M_%S")

        if DoVis:
            # Draw bounding box on the image and save it
            draw = ImageDraw.Draw(image)
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            # Save the image with the bounding box overlay
            visualization_path = object_folder / "visualizations"
            os.makedirs(visualization_path, exist_ok=True)
            bbox_image_path = visualization_path / f"image_with_bbox_{time}.png"
            image.save(bbox_image_path)

            logging.info(f"Saved image with bounding box overlay to {bbox_image_path}.")

        # Save bounding box as object_data.json
        object_data = [{"label": object_name, "bbox_modal": bbox}]
        object_data_path = object_folder / "inputs" / "object_data.json"
        with open(object_data_path, "w") as f:
            json.dump(object_data, f)

        logging.info(f"Saved image and bounding box for {object_name} in {object_folder}.")

        print("Loading object data...")
        # Load required data
        if Depth is not None:
            # Save depth as image_depth.png
            depth_path = object_folder / "image_depth.png"
            depth_image = Image.fromarray((Depth * 1000).astype(np.uint16))
            depth_image.save(depth_path)
            logging.info(f"Saved depth image for {object_name} in {depth_path}.")

        if not Depth:
            image, depth, camera_data = self.load_observation(object_folder, load_depth=False)
        else:
            image, depth, camera_data = self.load_observation(object_folder, load_depth=True)
        object_data = self.load_object_data(object_data_path)

        print("Running MegaPose inference...")
        # Run MegaPose inference
        output = run_inference_on_example.my_inference(image, depth, camera_data, object_data, self.model_name, object_folder)

        print("MegaPose inference finished.")
        # Save results
        out_path = object_folder
        run_inference_on_example.save_predictions(out_path, output)

        # Convert outputs to JSON-friendly format
        labels = output.infos["label"]
        poses = output.poses.cpu().numpy()
        object_data = [{"label": label, "pose": pose.tolist()} for label, pose in zip(labels, poses)]

        if DoVis:
            vis = run_inference_on_example.my_visualization(object_folder, out_path / "visualizations", time=time)
            # return {"visualization": vis, "poses": object_data}

        return {"poses": object_data}
    


if __name__ == "__main__":
    megapose_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/object_models"
    megapose_handler = MegaPoseHandler(device, megapose_path)


    image = Image.open("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/cracker box/image_rgb.png")
    object_name="cracker box"
    bbox=[894, 1406, 1078, 1655]
    DoVis=True

    megapose_handler.estimate_pose(image, object_name, bbox)
