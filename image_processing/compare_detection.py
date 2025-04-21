from image_processing.glamm_handler import GLAMMHandler
from image_processing.rexseek_handler import RexSeekHandler
from image_processing.grounding_dino_handler import GroundingDINOHandler
from image_processing.lisa_handler import LISAHandler
from image_processing.midas_handler import MiDaSHandler
from image_processing.owlwit_handler import OwlViTHandler
from image_processing.owlv2_handler import Owlv2Handler

import json
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from PIL import ImageDraw

# Run me with python -m image_processing.compare_detection

MODEL_CLASSES = {
    "rexseek": RexSeekHandler,
    "owlvit": OwlViTHandler,
    "owlv2": Owlv2Handler,
    "grounding_dino": GroundingDINOHandler,
    "lisa": LISAHandler,
}

Objects = [
    "mustard bottle",
    "box of jello",
    "cracker box",
    "foam brick",
    "tuna fish can",
    "white lego brick",
    "Triple Outlet Plug"
]

def delete_detections():
    # Delete all "detections" folders
    base_path = Path("images/ScannedObjects/ScannedObjects/Stationary")
    for folder in base_path.iterdir():
        if folder.is_dir():
            detection_folder = folder / "detections"
            if detection_folder.exists():
                for file in detection_folder.iterdir():
                    file.unlink()  # Delete all files in the folder
                detection_folder.rmdir()  # Remove the empty folder


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "grounding_dino"
    model_handler = MODEL_CLASSES[model_name](device)
    
    base_path = Path("images/ScannedObjects/ScannedObjects/Stationary")
    
    for folder in base_path.iterdir():
        if folder.is_dir():
            detection_folder = folder / "detections"
            detection_folder.mkdir(exist_ok=True)
            detection_folder = detection_folder / model_name
            detection_folder.mkdir(exist_ok=True)

            detection_data = []
            for image_path in folder.glob("*.png"):
                image = Image.open(image_path)

                for object_name in Objects:
                    # Run inference
                    print(f"Running inference for {object_name} on {image_path}")
                    detections = model_handler.infer(image_path, object_name)

                    print("--" * 20)
                    print(f"{detections=}")
                    print(f"{image_path=}")
                    print("--" * 20)

                    if detections["bounding_boxes"] != []:


                        for detection in detections["bounding_boxes"]:
                            box = detection

                            # Draw the box on the image
                            image_with_box = image.copy()
                            draw = ImageDraw.Draw(image_with_box)
                            draw.rectangle(box, outline="red", width=3)
                            detection_name = f"{image_path.stem}_{object_name}.jpg"
                            detection_path = detection_folder / detection_name
                            image_with_box.save(detection_path)

                            # Add detection data
                            detection_data.append({
                                "original name": image_path.name,
                                "detection name": detection_name,
                                "box": box,
                                "object name": object_name
                            })

            # Save detection data to JSON
            json_path = detection_folder / "detections.json"
            with open(json_path, "w") as json_file:
                json.dump(detection_data, json_file, indent=4)
    



if __name__ == "__main__":
    main()