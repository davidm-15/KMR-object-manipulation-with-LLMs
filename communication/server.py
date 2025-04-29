import os
import sys
import argparse
import logging
import torch
import json
from flask import Flask, request, jsonify
from handlers.grounding_dino_handler import GroundingDINOHandler
from handlers.lisa_handler import LISAHandler
from handlers.midas_handler import MiDaSHandler
from handlers.yolo_handler import YOLOHandler
from PIL import Image
import subprocess
import re
import ast
import numpy as np
import pickle
from datetime import datetime
import pathlib


logging.basicConfig(level=logging.INFO)

# Run me with python -m communication.server glamm
# Run me with python -m communication.server rexseek 
# Run me with python -m communication.server yolo

MODEL_CLASSES = {
    "grounding_dino": GroundingDINOHandler,
    "lisa": LISAHandler,
    "yolo": YOLOHandler,
}

def parse_megapose_output(output_str):
    # Look for the poses section which contains the important data
    poses_match = re.search(r"'poses': \[(.*?)\]\}", output_str, re.DOTALL)
    
    if not poses_match:
        return None, "Could not find pose data in output"
    
    # Extract just the poses data
    poses_data = poses_match.group(1)
    
    # Parse the pose entry
    label_match = re.search(r"'label': '(.*?)'", poses_data)
    pose_matrix_match = re.search(r"'pose': (\[\[.*?\]\])", poses_data, re.DOTALL)
    
    if not label_match or not pose_matrix_match:
        return None, "Could not extract label or pose matrix"
    
    label = label_match.group(1)
    
    # Clean up and parse the pose matrix
    pose_matrix_str = pose_matrix_match.group(1)
    # Replace any numpy-specific notation
    pose_matrix_str = pose_matrix_str.replace(' ', '')
    
    try:
        # Manually parse the matrix
        rows = []
        for row_str in re.findall(r'\[(.*?)\]', pose_matrix_str):
            if row_str.startswith('['):
                row_str = row_str[1:]
            if row_str.endswith(']'):
                row_str = row_str[:-1]
            row = [float(x) for x in row_str.split(',')]
            rows.append(row)
        
        result = {
            "poses": [
                {
                    "label": label,
                    "pose": rows
                }
            ]
        }
        
        return json.dumps(result), None
    except Exception as e:
        return None, f"Error parsing pose matrix: {str(e)}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(model_handler, image_file, prompt):
    """Process an image using the given model handler."""
    try:
        logging.info(f"Processing image with prompt: {prompt}")
        results = model_handler.infer(image_file, prompt)
        return results
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {"error": str(e)}



def estimate_pose(pose_handler, image_file, object_name, bbox, **kwargs):
    """Perform 6D pose estimation."""
    print("Received request for pose estimation")
    DoVis = kwargs.get("DoVis", False)

    try:
        logging.info(f"Received image for {object_name} with bbox {bbox}")
        
        # Convert image file to PIL format
        image = Image.open(image_file).convert("RGB")

        # Run 6D pose estimation
        results = pose_handler.estimate_pose(image, object_name, bbox, DoVis=DoVis)
        return results
    except Exception as e:
        logging.error(f"Error in pose estimation: {e}")
        return {"error": str(e)}

def create_app(model_name):
    """Create Flask app with given model."""
    app = Flask(__name__)
    model_handler = MODEL_CLASSES[model_name](device)

    @app.route("/process", methods=["POST"])
    def process_image_route():
        """Object detection request via Flask."""
        if "image" not in request.files or "prompt" not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400
        image_file = request.files["image"]
        prompt = request.form["prompt"]
        return jsonify(process_image(model_handler, image_file, prompt))

    @app.route("/estimate_pose", methods=["POST"])
    def estimate_pose_route():
        megapose_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects"
        """6D pose estimation via Flask."""
        if "image" not in request.files or "object_name" not in request.form or "bbox" not in request.form:
            return jsonify({"error": "Missing image, object_name, or bbox"}), 400
        
        print("Received request for pose estimation")
        image_file = request.files["image"]
        object_name = request.form["object_name"]
        do_vis = request.form.get("DoVis", "False").lower() == "true"

        bbox = json.loads(request.form["bbox"])
        print("Received bbox:", bbox)

        # Save the image file to the megapose_path + object_name directory
        save_path = os.path.join(megapose_path, object_name)
        os.makedirs(save_path, exist_ok=True)
        image_save_path = os.path.join(save_path, "image_rgb.png")

        # Save the input image
        image_file.seek(0)  # Reset file pointer to beginning
        with open(image_save_path, "wb") as f:
            f.write(image_file.read())
        image_file.seek(0)  # Reset file pointer again for later use

        print(f"Image saved to {image_save_path}")
        
        # pose = megapose_main()
        # Call the Megapose testing module
        result = subprocess.run(
            ["python", "-m", "image_processing.Megapose_testing", "--bbox", json.dumps(bbox), "--prompt", object_name, "--image_file", image_save_path, "--DoVis", str(do_vis)],
            capture_output=True,
            text=True
        )


        json_path = os.path.join(megapose_path, object_name)
        os.makedirs(json_path, exist_ok=True)
        json_path = os.path.join(json_path, "pose_result.json")
        try:
            with open(json_path, "r") as f:
                json_str = json.load(f)
                print("Output from Megapose:", json_str)
        except Exception as e:
            logging.error(f"Error reading pose result file: {e}")
            return jsonify({"error": f"Failed to read pose data: {str(e)}"}), 500
        
        print("Pose matrix: \n", json_str["poses"][0]["pose"])



        pose = json_str["poses"][0]["pose"]

        return jsonify({
            "pose": pose,
            "label": json_str["poses"][0]["label"]
        })
    

    return app

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Start the inference server with a specified model.")
    args.add_argument("model", choices=MODEL_CLASSES.keys(), help="The inference model to use.")
    args = args.parse_args()

    app = create_app(args.model)
    app.run(host="0.0.0.0", port=5000)