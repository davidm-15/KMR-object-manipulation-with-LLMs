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
from handlers.qwen_handler import QwenHandler
from handlers.QwenDino_handler import QwenDino, QwenDino72
from PIL import Image
import subprocess
import re
import ast
import numpy as np
import pickle
from datetime import datetime
import pathlib
from accelerate import Accelerator
import requests
import json # Already imported for jsonify, but good to have explicitly for loads/dumps
import logging # For better error messages


logging.basicConfig(level=logging.INFO)

# Run me with python -m communication.server glamm
# Run me with python -m communication.server rexseek 
# Run me with python -m communication.server yolo


#python -m image_processing.Megapose_testing --image_file '/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/cracker box/image_rgb.png' --prompt 'cracker box' --bbox '[894, 1406, 1078, 1655]' --DoVis True

MODEL_CLASSES = {
    "dino": GroundingDINOHandler,
    "lisa": LISAHandler,
    "yolo": YOLOHandler,
    "qwendino": QwenDino72
}

MEGAPOSE_API_SERVICE_URL = "http://acn11:5001/estimate_pose_subprocess"
MEGAPOSE_REQUEST_TIMEOUT_SECONDS = 360

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if model_name == "qwendino":
        qwen = model_handler
    else:
        qwen = QwenHandler(device, model_name="Qwen/Qwen2.5-VL-7B-Instruct")

    @app.route("/text_inference", methods=["POST"])
    def text_inference_route():
        """Text inference request via Flask."""
        if "text_input" not in request.form:
            return jsonify({"error": "Missing text input"}), 400
        text_input = request.form["text_input"]
        max_new_tokens = int(request.form.get("max_new_tokens", 128))
        return jsonify(qwen.text_inference(text_input, max_new_tokens))
    
    @app.route("/image_inference", methods=["POST"])
    def image_inference_route():
        """Image inference request via Flask."""
        if "image" not in request.files or "prompt" not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400
        image_file = request.files["image"]
        prompt = request.form["prompt"]
        return jsonify(qwen.image_text_inference(image_file, prompt))
    

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
        """
        Receives object detection info and an image, then calls the remote Megapose API service
        to perform 6D pose estimation.
        """
        logging.info("Received request for /estimate_pose (will call remote Megapose API)")

        if "image" not in request.files or "object_name" not in request.form or "bbox" not in request.form:
            logging.warning("Missing image, object_name, or bbox in /estimate_pose request")
            return jsonify({"error": "Missing image, object_name, or bbox"}), 400
        
        image_file_storage = request.files["image"]  # This is a FileStorage object
        object_name = request.form["object_name"]
        # Bbox comes in as a JSON string representing a list, e.g., "[x,y,w,h]"
        # We will pass this string directly to the Megapose API server.
        bbox_json_str = request.form["bbox"]
        do_vis_str = request.form.get("DoVis", "False") # Default to "False" if not provided

        # Validate incoming bbox_json_str briefly before sending
        try:
            # Test if it's valid JSON locally, though the remote server will also check
            json.loads(bbox_json_str)
        except json.JSONDecodeError:
            logging.error(f"Invalid bbox JSON received: {bbox_json_str}")
            return jsonify({"error": f"Invalid bbox format. Expected a JSON string list, got: {bbox_json_str}"}), 400

        logging.info(f"Preparing to call Megapose API for object: '{object_name}', bbox_str: {bbox_json_str}, DoVis: {do_vis_str}")

        # Prepare data for the POST request to the Megapose API server
        # The key for the file ('image') must match what megapose_api_server.py expects in request.files
        files_to_send = {
            'image': (image_file_storage.filename or "uploaded_image.png", image_file_storage.stream, image_file_storage.mimetype)
        }
        form_data_payload = {
            'object_name': object_name,
            'bbox': bbox_json_str,  # Send the JSON string as is
            'DoVis': do_vis_str     # Send as string "True" or "False"
        }

        try:
            logging.info(f"Sending request to Megapose API service at: {MEGAPOSE_API_SERVICE_URL}")
            response = requests.post(
                MEGAPOSE_API_SERVICE_URL,
                files=files_to_send,
                data=form_data_payload,
                timeout=MEGAPOSE_REQUEST_TIMEOUT_SECONDS
            )
            # This will raise an HTTPError if the HTTP request returned an unsuccessful status code (4xx or 5xx)
            response.raise_for_status()

            # If we reach here, the HTTP request was successful (2xx status code)
            megapose_api_result = response.json() # Parse the JSON response from Megapose API
            logging.info(f"Successfully received response from Megapose API service.")
            # logging.debug(f"Megapose API Result: {megapose_api_result}") # Log full result if needed for debugging

            # The Megapose API server should return the contents of pose_result.json,
            # which is expected to be like: {"poses": [{"label": "...", "pose": [...]}]}
            # Or it might return {"error": "..."} if something went wrong on its end.

            if "error" in megapose_api_result:
                logging.error(f"Megapose API service reported an error: {megapose_api_result['error']}")
                return jsonify({"error": f"Megapose service error: {megapose_api_result['error']}"}), 500 # Or another appropriate status

            # Validate the structure of the successful response
            if "poses" not in megapose_api_result or not isinstance(megapose_api_result["poses"], list) or not megapose_api_result["poses"]:
                logging.error(f"Invalid 'poses' structure in response from Megapose API: {megapose_api_result}")
                return jsonify({"error": "Invalid pose data structure from Megapose service"}), 500
            
            first_pose_info = megapose_api_result["poses"][0]
            if "pose" not in first_pose_info or "label" not in first_pose_info:
                logging.error(f"Missing 'pose' or 'label' in first pose entry from Megapose API: {first_pose_info}")
                return jsonify({"error": "Incomplete pose data from Megapose service"}), 500

            extracted_pose = first_pose_info["pose"]
            extracted_label = first_pose_info["label"]
            
            logging.info(f"Pose estimation successful. Label: '{extracted_label}'")
            # logging.debug(f"Pose matrix: \n{extracted_pose}")

            return jsonify({
                "pose": extracted_pose,
                "label": extracted_label
                # You can also forward the full megapose_api_result if your client expects more
                # "full_megapose_output": megapose_api_result
            })

        except requests.exceptions.Timeout:
            logging.error(f"Request to Megapose API service ({MEGAPOSE_API_SERVICE_URL}) timed out after {MEGAPOSE_REQUEST_TIMEOUT_SECONDS}s.")
            return jsonify({"error": "Megapose service request timed out"}), 504 # 504 Gateway Timeout
        except requests.exceptions.ConnectionError:
            logging.error(f"Could not connect to Megapose API service at {MEGAPOSE_API_SERVICE_URL}.")
            return jsonify({"error": "Could not connect to Megapose service"}), 503 # 503 Service Unavailable
        except requests.exceptions.HTTPError as http_err:
            # HTTPError was raised by response.raise_for_status() for 4xx/5xx responses
            error_content = http_err.response.text # Get raw text content of error
            try:
                # Attempt to parse as JSON if the remote server sent a JSON error
                error_json = http_err.response.json()
                logging.error(f"Megapose API service returned HTTP {http_err.response.status_code}: {error_json}")
                # Forward the error from the remote service if available
                remote_error_message = error_json.get("error", "Unknown error from Megapose service")
                remote_error_details = error_json.get("details", "")
                return jsonify({"error": remote_error_message, "details": remote_error_details, "status_code": http_err.response.status_code}), http_err.response.status_code
            except json.JSONDecodeError:
                # If the error response wasn't JSON
                logging.error(f"Megapose API service returned HTTP {http_err.response.status_code} with non-JSON body: {error_content}")
                return jsonify({"error": f"Megapose service error (HTTP {http_err.response.status_code})", "details": error_content}), http_err.response.status_code
        except json.JSONDecodeError:
            # This means response.json() failed, indicating the 2xx response wasn't valid JSON
            logging.error(f"Megapose API service returned a 2xx response but it was not valid JSON. Response text: {response.text}")
            return jsonify({"error": "Megapose service returned invalid JSON success response"}), 500
        except Exception as e:
            # Catch-all for any other unexpected errors during the request or processing
            logging.error(f"An unexpected error occurred while calling Megapose API service: {e}", exc_info=True)
            return jsonify({"error": f"An unexpected error occurred with the Megapose service: {str(e)}"}), 500



    return app

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Start the inference server with a specified model.")
    args.add_argument("model", choices=MODEL_CLASSES.keys(), help="The inference model to use.")
    args = args.parse_args()

    app = create_app(args.model)
    app.run(host="0.0.0.0", port=5000)

    # accelerator = Accelerator()

    # if accelerator.is_main_process:
    #     app.run(host="0.0.0.0", port=5000)
    