import os
import sys
import argparse
import logging
import torch
import json
from flask import Flask, request, jsonify
from image_processing.glamm_handler import GLAMMHandler
from image_processing.rexseek_handler import RexSeekHandler
from image_processing.grounding_dino_handler import GroundingDINOHandler
from image_processing.megapose_handler import MegaPoseHandler
from image_processing.lisa_handler import LISAHandler
from image_processing.midas_handler import MiDaSHandler
from PIL import Image

logging.basicConfig(level=logging.INFO)

# Run me with python -m communication.server glamm
# Run me with python -m communication.server rexseek 

MODEL_CLASSES = {
    "glamm": GLAMMHandler,
    "rexseek": RexSeekHandler,
    "grounding_dino": GroundingDINOHandler,
    "lisa": LISAHandler,
}

def initialize_handlers(model_name):
    """Initialize model and pose handlers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_handler = MODEL_CLASSES[model_name](device)
    pose_handler = MegaPoseHandler(device, "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects")
    depth_handler = MiDaSHandler()
    return model_handler, pose_handler, depth_handler

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
    model_handler, pose_handler, depth_handler = initialize_handlers(model_name)

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
        """6D pose estimation via Flask."""
        if "image" not in request.files or "object_name" not in request.form or "bbox" not in request.form:
            return jsonify({"error": "Missing image, object_name, or bbox"}), 400
        
        print("Received request for pose estimation")
        image_file = request.files["image"]
        object_name = request.form["object_name"]
        bbox = json.loads(request.form["bbox"])
        print("Received bbox:", bbox)
        
        return jsonify(estimate_pose(pose_handler, image_file, object_name, bbox))

    return app

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Start the inference server with a specified model.")
    args.add_argument("model", choices=MODEL_CLASSES.keys(), help="The inference model to use.")
    args = args.parse_args()

    app = create_app(args.model)
    app.run(host="0.0.0.0", port=5000)
