import os
import sys
import json
import base64
import cv2
import torch
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection, 
    DPTImageProcessor, DPTForDepthEstimation, pipeline
)

sys.path.append('../../megapose6d/src/megapose/scripts')
sys.path.append('../megapose6d/src/megapose/scripts')

from megapose.scripts import run_inference_on_example
from megapose.datasets.scene_dataset import CameraData
from megapose.lib3d.transform import Transform

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_TYPE = os.getenv("INFERENCE_TYPE", "GLAMM")  # Default to GLAMM

if INFERENCE_TYPE == "GLAMM":
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    logging.info(f"Loaded GLAMM model: {model_id}")

depth_model_id = "Intel/dpt-large"
depth_processor = DPTImageProcessor.from_pretrained(depth_model_id)
depth_model = DPTForDepthEstimation.from_pretrained(depth_model_id).to(device)
logging.info(f"Loaded Depth Estimation model: {depth_model_id}")


def encode_image(image):
    _, img_encoded = cv2.imencode(".jpg", image)
    return base64.b64encode(img_encoded.tobytes()).decode("utf-8")


def depth_estimation(image):
    """ Perform depth estimation on the input image """
    inputs = depth_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
        depth_output = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
    return depth_output


def inference_glamm(image, prompt):
    """ Perform GLAMM object detection """
    inputs = processor(images=image, text=[prompt], return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )

    return results[0]["boxes"], results[0]["scores"], results[0]["labels"]



@app.route("/process", methods=["POST"])
def process_image():
    """ Endpoint for processing an image """
    try:
        if "image" not in request.files or "prompt" not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400

        # Read & Convert Image
        image_file = request.files["image"]
        image = Image.open(image_file).convert("RGB")
        prompt = request.form["prompt"]

        logging.info(f"Processing image with prompt: {prompt}")

        if INFERENCE_TYPE == "GLAMM":
            boxes, scores, labels = inference_glamm(image, prompt)
            bounding_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]

            if not bounding_boxes:
                return jsonify({"bounding_boxes": bounding_boxes})
            else:
                depth_image = depth_estimation(image)
                depth_image_encoded = encode_image(depth_image)
                return jsonify({
                    "bounding_boxes": bounding_boxes,
                    "depth_image": depth_image_encoded,
                })


    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
