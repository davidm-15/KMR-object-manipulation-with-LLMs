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

# run me with python -m image_processing.Megapose_testing

def main():
    model_handler, pose_handler, depth_handler = initialize_handlers("grounding_dino")
    image_file = "images/JustPickIt/img.png"
    prompt = "foam brick"

    detection_result = process_image(model_handler, image_file, prompt)
    print("Result:", detection_result)

    image = Image.open(image_file).convert("RGB")
    depth_estimation_result = depth_handler.estimate_depth(image)
    estimate_pose_result = estimate_pose(pose_handler, image_file, prompt, detection_result["bounding_boxes"][0], DoVis=True, Depth=depth_estimation_result)


    print("Pose Result:", estimate_pose_result)


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