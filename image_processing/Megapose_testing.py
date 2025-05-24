import sys

sys.path.append('../megapose6d/src/megapose/scripts')

import run_inference_on_example
from pathlib import Path
from typing import List
import argparse
import json
from megapose.datasets.scene_dataset import ObjectData
from megapose.lib3d.transform import Transform
from handlers.megapose_handler import MegaPoseHandler
import os
import torch
from handlers.rexseek_handler import RexSeekHandler
from handlers.grounding_dino_handler import GroundingDINOHandler
from handlers.lisa_handler import LISAHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

import logging
from transformers import AutoConfig

# run me with python -m image_processing.Megapose_testing

MODEL_CLASSES = {
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
    megapose_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/object_models"

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
    example_dir = Path("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/foam brick")
    image, depth, camera_data = run_inference_on_example.load_observation(example_dir, load_depth=False)
    object_data = run_inference_on_example.load_object_data(example_dir / "inputs" / "object_data.json")



    output = run_inference_on_example.my_inference(image, depth, camera_data, object_data, model_name, example_dir)


    out_path = example_dir
    run_inference_on_example.save_predictions(out_path, output)


    labels = output.infos["label"]
    poses = output.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])



    run_inference_on_example.my_visualization(example_dir, out_path / "visualizations")

def run_main():
    # pos = main(
    #     image_file="/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/foam brick/image_rgb.png",
    #     prompt="foam brick",
    #     bbox=[996, 934, 1348, 1446],
    #     DoVis=True)
    
    # print("Pose Result:\n", pos)


    pos = main(
        image_file="/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/cracker box/image_rgb.png",
        prompt="cracker box",
        bbox=[894, 1406, 1078, 1655],
        DoVis=True)
    
    print("Pose Result:\n", pos)

if __name__ == "__main__":
    # test_megapose()
    run_main()
    


    # # Set up command-line argument parser
    # parser = argparse.ArgumentParser(description='Run object pose estimation')
    # parser.add_argument('--image_file', type=str, default='images/JustPickIt/img.png', help='Path to the image file')
    # parser.add_argument('--prompt', type=str, default='foam brick', help='Text prompt for detection')
    # parser.add_argument('--bbox', type=str, help='Bounding box coordinates in format "x1,y1,x2,y2"')
    # parser.add_argument('--DoVis', default=False, type=bool, help='Enable visualization')

    # args = parser.parse_args()

    # args.bbox = args.bbox.replace("[", "").replace("]", "")
    # args.bbox = [int(coord) for coord in args.bbox.split(",")]

    # print("DoVis main Megapose_testing.py:", args.DoVis)
    # main(
    #     image_file=args.image_file,
    #     prompt=args.prompt,
    #     bbox=args.bbox,
    #     DoVis=args.DoVis
    # )