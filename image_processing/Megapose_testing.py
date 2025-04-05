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
from image_processing.glamm_handler import GLAMMHandler
from image_processing.rexseek_handler import RexSeekHandler
from image_processing.grounding_dino_handler import GroundingDINOHandler
from image_processing.lisa_handler import LISAHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

import logging
from transformers import AutoConfig

# run me with python -m image_processing.Megapose_testing

MODEL_CLASSES = {
    "glamm": GLAMMHandler,
    "rexseek": RexSeekHandler,
    "grounding_dino": GroundingDINOHandler,
    "lisa": LISAHandler,
}





def main(**kwargs):
    image_file = kwargs.get("image_file", "images/JustPickIt/img.png")
    prompt = kwargs.get("prompt", "foam brick")
    bbox = kwargs.get("bbox", None)
    DoVis = kwargs.get("DoVis", False)

    megapose_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects"

    pose_handler = MegaPoseHandler(device, megapose_path)

    print("bbox:", bbox)
    image = Image.open(image_file).convert("RGB")
    estimate_pose_result = pose_handler.estimate_pose(image, prompt, bbox, DoVis=DoVis)

    print("Pose Result:", estimate_pose_result)

    # Save the result to a JSON file in the megapose_path + prompt directory
    save_dir = os.path.join(megapose_path, prompt)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pose_result.json")
    

    
    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(estimate_pose_result, f, indent=2)
    
    print(f"Saved pose estimation result to: {save_path}")

    return estimate_pose_result


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
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Run object pose estimation')
    parser.add_argument('--model_name', type=str, default='glamm', help='Model name for detection')
    parser.add_argument('--image_file', type=str, default='images/JustPickIt/img.png', help='Path to the image file')
    parser.add_argument('--prompt', type=str, default='foam brick', help='Text prompt for detection')
    parser.add_argument('--bbox', type=str, help='Bounding box coordinates in format "x1,y1,x2,y2"')

    args = parser.parse_args()

    # # Process the bbox argument if provided
    # detection_result = None
    # if args.bbox:
    #     try:
    #         # Parse the bbox string into coordinates
    #         x1, y1, x2, y2 = map(float, args.bbox.split(','))
    #         detection_result = {
    #             "bounding_boxes": [
    #                 [x1, y1, x2, y2]
    #             ]
    #         }
    #     except ValueError:
    #         print("Error: Invalid bbox format. Expected 'x1,y1,x2,y2'")
    #         sys.exit(1)

    # bbox = None
    # if detection_result:
    #     bbox = detection_result["bounding_boxes"][0]
    # else:
    #     bbox = [0, 0, 0, 0]

    # print("Bounding box:", bbox)
    # x1, y1, x2, y2 = bbox
    # print("x1, y1, x2, y2:", x1, y1, x2, y2)

    main(
        model_name="glamm",
        image_file="images/JustPickIt/img.png",
        prompt="foam brick",
        bbox=[1141, 1187, 1552, 1742],
    )