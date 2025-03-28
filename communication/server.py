import os
import sys
import argparse
import logging
import torch
import json
from flask import Flask, request, jsonify
from image_processing.glamm_handler import GLAMMHandler
from image_processing.rexseek_handler import RexSeekHandler
from image_processing.megapose_handler import MegaPoseHandler
from PIL import Image

logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "glamm": GLAMMHandler,
    "rexseek": RexSeekHandler
}

# Run me with python -m communication.server glamm
# Run me with python -m communication.server rexseek 

def parse_args():
    parser = argparse.ArgumentParser(description="Start the inference server with a specified model.")
    parser.add_argument("model", choices=MODEL_CLASSES.keys(), help="The inference model to use.")
    return parser.parse_args()

def create_app(model_name):
    app = Flask(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_handler = MODEL_CLASSES[model_name](device)
    pose_handler = MegaPoseHandler(device)

    @app.route("/process", methods=["POST"])
    def process_image():
        """ Object detection request """
        try:
            if "image" not in request.files or "prompt" not in request.form:
                return jsonify({"error": "Missing image or prompt"}), 400

            image_file = request.files["image"]
            prompt = request.form["prompt"]

            logging.info(f"Processing image with prompt: {prompt}")

            results = model_handler.infer(image_file, prompt)
            return jsonify(results)

        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/estimate_pose", methods=["POST"])
    def estimate_pose():
        """ Endpoint for 6D pose estimation """
        try:
            if "image" not in request.files or "object_name" not in request.form or "bbox" not in request.form:
                return jsonify({"error": "Missing image, object_name, or bbox"}), 400

            print("Received image for 6D pose estimation")
            image_file = request.files["image"]
            print("image_file: ", image_file)
            object_name = request.form["object_name"]
            print("object_name: ", object_name)
            bbox = json.loads(request.form["bbox"])  # Expecting JSON array [x1, y1, x2, y2]
            print("bbox: ", bbox)

            logging.info(f"Received image for {object_name} with bbox {bbox}")

            # Convert image file to PIL format
            image = Image.open(image_file).convert("RGB")

            # Run 6D pose estimation
            print("Running 6D pose estimation")
            results = pose_handler.estimate_pose(image, object_name, bbox)
            return jsonify(results)

        except Exception as e:
            logging.error(f"Error in pose estimation: {e}")
            return jsonify({"error": str(e)}), 500




    return app

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.model)
    app.run(host="0.0.0.0", port=5000)
