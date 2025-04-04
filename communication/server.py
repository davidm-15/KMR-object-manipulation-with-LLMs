# communication/server.py
import os
import sys
import argparse
import logging
import torch
import json
from flask import Flask, request, jsonify
from image_processing.glamm_handler import GLAMMHandler
# ... other imports ...
from image_processing.megapose_handler import MegaPoseHandler
from image_processing.midas_handler import MiDaSHandler
from image_processing.rexseek_handler import RexSeekHandler
from image_processing.grounding_dino_handler import GroundingDINOHandler
from image_processing.lisa_handler import LISAHandler
from PIL import Image
import tempfile # For potential temp file usage later
import uuid     # For potential temp file usage later
import numpy as np
from pathlib import Path

import subprocess
import time # For timing subprocess

WORKER_SCRIPT_PATH = Path("communication/megapose_inference_worker.py") # Adjust if needed
MEGAPOSE_OBJECTS_PATH = Path("/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects") # Keep this absolute path or make configurable

# Run me with python -m communication.server glamm

logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    "glamm": GLAMMHandler,
    "rexseek": RexSeekHandler,
    "grounding_dino": GroundingDINOHandler,
    "lisa": LISAHandler,
}

def initialize_handlers(model_name):
    """Initialize ONLY the object detection model handler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initializing object detector '{model_name}' on device {device}")
    if model_name not in MODEL_CLASSES:
        logging.error(f"Invalid model name: {model_name}")
        raise ValueError(f"Invalid model name: {model_name}")
    model_handler = MODEL_CLASSES[model_name](device)
    # Remove MegaPoseHandler and MiDaSHandler initialization from here
    # pose_handler = MegaPoseHandler(device, MEGAPOSE_OBJECTS_PATH) # NO LONGER NEEDED HERE
    # depth_handler = MiDaSHandler() # NO LONGER NEEDED HERE (unless used elsewhere)
    logging.info("Object detector initialized successfully.")
    # Return only what's needed globally (maybe just model_handler)
    return model_handler #, depth_handler if needed elsewhere


def process_image(model_handler, image_file, prompt):
    """Process an image using the given model handler."""
    try:
        logging.info(f"Processing image with prompt: {prompt}")
        # Consider saving to temp file if direct stream causes issues
        results = model_handler.infer(image_file, prompt)
        return results
    except Exception as e:
        logging.exception("Error processing image") # Use logging.exception to include traceback
        return {"error": str(e)}

def estimate_pose_via_subprocess(image_input, object_name, bbox, visualize=False, **kwargs):
    """
    Performs 6D pose estimation by launching a separate worker script.
    Accepts Flask FileStorage object or a file path string/Path object as image_input.
    Optionally triggers visualization generation in the worker.
    """
    logging.info(f"Estimate Pose Subprocess: Request for {object_name} with bbox {bbox}, Visualize: {visualize}")
    start_time = time.time()

    temp_image_path = None # Keep track if we create a temp image file
    temp_output_path = None # Keep track of temp output file

    try:
        # --- Determine input type and setup paths ---
        if isinstance(image_input, str) or isinstance(image_input, Path):
            # Input is a path, use it directly as the source image for the worker
            image_path_for_worker = Path(image_input)
            if not image_path_for_worker.is_file():
                raise FileNotFoundError(f"Input image path not found: {image_path_for_worker}")
            logging.info(f"Using provided image path directly: {image_path_for_worker}")
            # Create only the temporary output file
            temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            temp_output_path = Path(temp_output_file.name)
            temp_output_file.close()

        elif hasattr(image_input, 'save') and hasattr(image_input, 'filename'):
            # Input looks like Flask FileStorage, save it to a temp file
            logging.info("Input is FileStorage-like, saving to temporary file.")
            _, image_suffix = os.path.splitext(image_input.filename)
            if not image_suffix: image_suffix = '.png' # Default suffix
            temp_image_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix)
            temp_output_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            temp_image_path = Path(temp_image_file_obj.name) # Path where uploaded file is saved
            temp_output_path = Path(temp_output_file_obj.name)
            temp_image_file_obj.close()
            temp_output_file_obj.close()
            image_input.save(temp_image_path) # Save the stream content
            image_path_for_worker = temp_image_path # Worker uses the saved temp file
            logging.info(f"Saved uploaded image to temporary path: {temp_image_path}")
        else:
            # Unsupported input type
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        # --- End input type handling ---


        # 2. Prepare arguments for the worker script
        python_executable = sys.executable # Use the same python env
        bbox_json_string = json.dumps(bbox)
        # Ensure all paths are strings for subprocess
        cmd = [
            str(python_executable),
            str(WORKER_SCRIPT_PATH),
            "--image-path", str(image_path_for_worker), # Use the determined path
            "--object-name", object_name,
            "--bbox-json", bbox_json_string,
            "--output-file", str(temp_output_path),
            "--object-folder-root", str(MEGAPOSE_OBJECTS_PATH),
        ]
        # Add visualization flag if requested
        if visualize:
            cmd.append("--visualize")

        logging.info(f"Executing worker command: {' '.join(cmd)}")

        # 3. Run the worker script as a subprocess
        timeout_seconds = 180 # Adjust as needed
        process = subprocess.run(
            cmd,
            capture_output=True, text=True, check=False, timeout=timeout_seconds
        )

        # 4. Process the result (logging stdout/stderr)
        elapsed_time = time.time() - start_time
        logging.info(f"Subprocess finished in {elapsed_time:.2f} seconds. Return code: {process.returncode}")
        if process.stdout:
             logging.info(f"Worker stdout:\n{process.stdout.strip()}")
        if process.stderr:
             logging.warning(f"Worker stderr:\n{process.stderr.strip()}")

        # ... (rest of result processing logic: check return code, read output file) ...
        if process.returncode == 0:
             # ... read json ...
             # ... check for internal error ...
             # Successfully read data
             if 'result_data' in locals():
                 return result_data # Return the content read from the JSON file
             else: # Should not happen if file is read correctly
                 logging.error(f"Worker success but failed to read/parse output file: {temp_output_path}")
                 return {"error": "Pose estimation worker finished but failed to parse output file."}

        else:
             # Failure
             # ... (construct error message from stderr/output file) ...
             # Return error dictionary
             if 'error_message' not in locals():
                  error_message = f"Pose estimation worker failed with exit code {process.returncode}."
             logging.error(error_message)
             return {"error": error_message}

    except FileNotFoundError as fnf_error:
         logging.error(f"File not found error: {fnf_error}")
         return {"error": str(fnf_error)}
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logging.error(f"Pose estimation worker timed out after {elapsed_time:.2f} seconds (limit: {timeout_seconds}s).")
        return {"error": f"Pose estimation worker timed out after {timeout_seconds} seconds."}
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.exception(f"An unexpected error occurred in estimate_pose_via_subprocess after {elapsed_time:.2f} seconds.")
        return {"error": f"Server error during subprocess execution: {str(e)}"}
    finally:
        # 5. Cleanup temporary files CREATED by this function
        # Only delete temp_image_path if it was created by us (from FileStorage)
        if temp_image_path and temp_image_path.exists():
            logging.debug(f"Cleaning up temporary image file: {temp_image_path}")
            try:
                os.remove(temp_image_path)
            except OSError as e:
                logging.warning(f"Could not remove temp image file {temp_image_path}: {e}")
        # Always delete the temporary output file if it exists
        if temp_output_path and temp_output_path.exists():
            logging.debug(f"Cleaning up temporary output file: {temp_output_path}")
            try:
                os.remove(temp_output_path)
            except OSError as e:
                logging.warning(f"Could not remove temp output file {temp_output_path}: {e}")



def estimate_pose(pose_handler, image_input, object_name, bbox, **kwargs):
    """Perform 6D pose estimation."""
    logging.info(f"Received request for pose estimation for {object_name} with bbox {bbox}")
    DoVis = kwargs.get("DoVis", False)
    # Depth = kwargs.get("Depth", None) # Pass depth if needed

    try:
        # Handle both file paths and PIL Images (or file streams)
        if isinstance(image_input, str): # If it's a path
             image = Image.open(image_input).convert("RGB")
        elif hasattr(image_input, 'read'): # If it's a file-like object (from request.files)
             image = Image.open(image_input).convert("RGB")
        else: # Assume it's already a PIL Image
             image = image_input # Use directly if passed as PIL

        logging.info("Running MegaPose pose estimation...")
        print("D"*100) # Keep your debug prints for now
        # Pass depth estimation result if available and needed by MegaPoseHandler
        # results = pose_handler.estimate_pose(image, object_name, bbox, DoVis=DoVis, Depth=Depth)
        results = pose_handler.estimate_pose(image, object_name, bbox, DoVis=DoVis)
        print("Pose Estimation Result Structure:", type(results), results.keys() if isinstance(results, dict) else results) # Add introspection
        return results
    except Exception as e:
        logging.exception("Error in pose estimation") # Use logging.exception
        return {"error": str(e)}


def create_app(model_name):
    """Create Flask app, initializing only the detector."""
    app_instance = Flask(__name__)
    try:
        # Only initialize the object detector globally
        model_handler = initialize_handlers(model_name)
    except Exception as e:
         logging.error(f"Failed to initialize object detector for model {model_name}: {e}", exc_info=True)
         @app_instance.route("/process", methods=["POST"])
         @app_instance.route("/estimate_pose", methods=["POST"])
         def init_error_route():
             return jsonify({"error": f"Server initialization failed for model {model_name}: {e}"}), 503
         return app_instance

    @app_instance.route("/process", methods=["POST"])
    def process_image_route():
        # ... (keep this route as is, using model_handler) ...
        if "image" not in request.files or "prompt" not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400
        image_file = request.files["image"]
        prompt = request.form["prompt"]
        return jsonify(process_image(model_handler, image_file, prompt))


    @app_instance.route("/estimate_pose", methods=["POST"])
    def estimate_pose_route():
        """6D pose estimation via Flask (using subprocess), with optional visualization."""
        required_fields = ["image", "object_name", "bbox"]
        if not all(field in request.files or field in request.form for field in required_fields):
            missing = [field for field in required_fields if field not in request.files and field not in request.form]
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        logging.info("Received request for pose estimation route (subprocess)")
        image_file = request.files["image"]
        object_name = request.form["object_name"]
        try:
            bbox = json.loads(request.form["bbox"])
            logging.info(f"Received bbox: {bbox}")
        except json.JSONDecodeError:
            logging.error(f"Invalid bbox format received: {request.form['bbox']}")
            return jsonify({"error": "Invalid bbox format, must be JSON list"}), 400

        # --- Check for visualization flag in form data ---
        # Accepts 'true', 'True', '1', 'yes' (case-insensitive)
        visualize_req = request.form.get("visualize", "false").lower() in ['true', '1', 'yes']
        logging.info(f"Visualization requested: {visualize_req}")

        # Call the function that uses the subprocess, passing the flag
        estimated_pose_result = estimate_pose_via_subprocess(
            image_file,
            object_name,
            bbox,
            visualize=visualize_req # Pass the boolean flag
        )

        # Return the result
        status_code = 500 if "error" in estimated_pose_result else 200
        return jsonify(estimated_pose_result), status_code


    return app_instance

# --- Gunicorn Entry Point ---
# Read model name from environment variable. Fallback to a default if not set.
# Make sure this default is valid on your system.
selected_model_name = os.environ.get("MEGAPOSE_SERVER_MODEL", "grounding_dino")

if selected_model_name not in MODEL_CLASSES:
    logging.error(f"Invalid model specified via environment variable MEGAPOSE_SERVER_MODEL: '{selected_model_name}'")
    logging.error(f"Valid models are: {list(MODEL_CLASSES.keys())}")
    # Exit or raise an error if the model is critical and invalid
    sys.exit(f"Exiting due to invalid model configuration: {selected_model_name}")


# Create the app instance *globally* using the selected model name. Gunicorn will look for this 'app'.
logging.info(f"Creating Flask app for Gunicorn with model: {selected_model_name}")
app = create_app(selected_model_name)
# --- End Gunicorn Entry Point ---


if __name__ == "__main__":
    # This block is now only for running with the Flask development server directly
    # (e.g., python -m communication.server glamm)
    # It's good practice to keep it for easier local testing.
    parser = argparse.ArgumentParser(description="Start the inference server (Flask dev mode).")
    # Make the model argument optional for the dev server, using the env var as default
    parser.add_argument("model", choices=MODEL_CLASSES.keys(), nargs='?', default=selected_model_name,
                        help=f"The inference model to use (default: {selected_model_name} from env var or hardcoded default).")
    args = parser.parse_args()

    # Create a specific app instance for the dev server using the argument passed
    print(f"Starting Flask development server with model: {args.model}")
    # Note: The global 'app' is for Gunicorn. We create a new one here based on command-line args.
    dev_app = create_app(args.model)
    # Run the Flask development server (not recommended for production)
    # Set threaded=False if you suspect threading issues, but start with default
    dev_app.run(host="0.0.0.0", port=5000, debug=False) # debug=False is usually safer for ML models