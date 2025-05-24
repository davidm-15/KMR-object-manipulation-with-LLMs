# megapose_server.py
# megapose_api_server.py
import flask
from flask import Flask, request, jsonify
import os
import subprocess
import json
import logging
import uuid # For unique temporary filenames/directories

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Path where Megapose_testing.py can find object models and write its output JSON
# This path MUST be accessible by the subprocess.
BASE_OBJECT_MODELS_PATH = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/object_models"
# Path to the Python interpreter in the conda environment Megapose should use
# Ensure this environment has all dependencies for Megapose_testing.py
# Root of your project where 'image_processing' module is located
PROJECT_ROOT = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs"
MEGAPOSE_TIMEOUT_SECONDS = 300 # 5 minutes

# GPU for Megapose subprocess on this node.
# This server itself might not use GPU, but the subprocess will.

@app.route('/estimate_pose_subprocess', methods=['POST'])
def estimate_pose_subprocess_route():
    logging.info("Received request for /estimate_pose_subprocess")
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    if not all(k in request.form for k in ['object_name', 'bbox']):
        return jsonify({"error": "Missing object_name or bbox from form data"}), 400

    image_file_storage = request.files['image']
    object_name = request.form['object_name']
    bbox_str = request.form['bbox']
    do_vis_str = request.form.get('DoVis', 'False')

    # Validate and parse bbox
    try:
        bbox_list = json.loads(bbox_str)
        if not (isinstance(bbox_list, list) and len(bbox_list) == 4 and all(isinstance(x, int) for x in bbox_list)):
            raise ValueError("Bbox must be a list of 4 integers.")
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Invalid bbox format: {bbox_str}. Error: {e}")
        return jsonify({"error": f"Invalid bbox format: {bbox_str}. Error: {e}"}), 400

    # --- File Handling on Shared Filesystem ---
    # Create a unique directory for this request to avoid conflicts if needed,
    # or save directly into the object_name folder if Megapose_testing.py handles it.
    # For simplicity, assuming Megapose_testing.py uses object_name for its output dir.
    # The image needs to be saved where Megapose_testing.py can access it.
    
    # Define where the input image for Megapose_testing.py will be saved
    # This path must be on the shared filesystem.
    # Megapose_testing.py expects the image in <megapose_data_path>/<prompt>/image_rgb.png
    # Let's ensure this structure or adapt. Here, we'll save it uniquely and pass the path.
    
    unique_id = str(uuid.uuid4())
    temp_image_dir = os.path.join(BASE_OBJECT_MODELS_PATH, "_temp_inputs", unique_id)
    os.makedirs(temp_image_dir, exist_ok=True)
    
    # Use a consistent name that Megapose_testing.py might expect, or just a unique one.
    # If Megapose_testing.py always looks for "image_rgb.png" in a specific relative path, adjust.
    # Here, we just save it and pass the absolute path.
    image_filename = image_file_storage.filename or f"{unique_id}_image.png"
    image_save_path = os.path.join(temp_image_dir, image_filename)

    try:
        image_file_storage.save(image_save_path)
        logging.info(f"Image saved to shared path: {image_save_path}")
    except Exception as e:
        logging.error(f"Failed to save image to {image_save_path}: {e}")
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

    # --- Subprocess Call to Megapose_testing.py ---
    megapose_cmd = [
        "python", "-m", "image_processing.Megapose_testing",
        "--image_file", image_save_path,
        "--prompt", object_name,
        "--bbox", json.dumps(bbox_list), # Pass as JSON string
        "--DoVis", do_vis_str,
    ]
    
    logging.info(f"Running Megapose command: {' '.join(megapose_cmd)}")
    
    sub_process_env = os.environ.copy()
    sub_process_env["CUDA_VISIBLE_DEVICES"] = MEGAPOSE_SUBPROCESS_CUDA_DEVICE
    sub_process_env["EGL_VISIBLE_DEVICES"] = MEGAPOSE_SUBPROCESS_CUDA_DEVICE # If EGL is used by Megapose
    # Important: Ensure PYTHONPATH allows finding 'image_processing' module
    # If PROJECT_ROOT is not already on python path for the MEGAPOSE_PYTHON_ENV
    sub_process_env["PYTHONPATH"] = f"{PROJECT_ROOT}:{sub_process_env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            megapose_cmd,
            capture_output=True,
            text=True,
            env=sub_process_env,
            timeout=MEGAPOSE_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT # Run from project root so relative paths in script work
        )
        logging.info(f"Megapose subprocess stdout:\n{result.stdout}")
        if result.stderr:
            logging.error(f"Megapose subprocess stderr:\n{result.stderr}")

        if result.returncode != 0:
            logging.error(f"Megapose script failed with return code {result.returncode}")
            # Clean up temporary image
            if os.path.exists(image_save_path): os.remove(image_save_path)
            if os.path.exists(temp_image_dir): os.rmdir(temp_image_dir) # Only if empty
            return jsonify({"error": "Megapose script failed", "details": result.stderr, "stdout": result.stdout}), 500

    except subprocess.TimeoutExpired:
        logging.error(f"Megapose script timed out after {MEGAPOSE_TIMEOUT_SECONDS} seconds.")
        if os.path.exists(image_save_path): os.remove(image_save_path)
        if os.path.exists(temp_image_dir): os.rmdir(temp_image_dir)
        return jsonify({"error": "Megapose script timed out"}), 500
    except Exception as e:
        logging.error(f"Error running Megapose subprocess: {e}", exc_info=True)
        if os.path.exists(image_save_path): os.remove(image_save_path)
        if os.path.exists(temp_image_dir): os.rmdir(temp_image_dir)
        return jsonify({"error": f"Failed to run Megapose: {str(e)}"}), 500

    # --- Read Result JSON from Shared Filesystem ---
    # Megapose_testing.py should save its output to:
    # <megapose_data_path>/<prompt>/pose_result.json
    # Ensure your Megapose_testing.py's main function uses args.megapose_data_path for this.
    # The prompt needs to be sanitized for directory names if it's arbitrary user input.
    safe_prompt_dirname = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in object_name).rstrip()
    json_output_path = os.path.join(BASE_OBJECT_MODELS_PATH, safe_prompt_dirname, "pose_result.json")
    
    logging.info(f"Attempting to read Megapose output from: {json_output_path}")
    if not os.path.exists(json_output_path):
        logging.error(f"Megapose output JSON not found at {json_output_path}.")
        return jsonify({"error": "Megapose output JSON not found by API server", "stdout": result.stdout, "stderr": result.stderr}), 500

    try:
        with open(json_output_path, "r") as f:
            pose_data = json.load(f)
        logging.info(f"Successfully read pose data from {json_output_path}")
        
        # Clean up the temporary input image after successful processing
        if os.path.exists(image_save_path): os.remove(image_save_path)
        if os.path.exists(temp_image_dir): os.rmdir(temp_image_dir) # Only if empty

        return jsonify(pose_data) # Return the full JSON content from Megapose_testing.py
    except Exception as e:
        logging.error(f"Error reading or parsing pose result file {json_output_path}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to read or parse pose data: {str(e)}"}), 500

if __name__ == '__main__':
    # This server runs on the Megapose node
    # Ensure its CUDA_VISIBLE_DEVICES for the *subprocess* is set correctly
    app.run(host='0.0.0.0', port=5001) # Use a different port than Qwen server
