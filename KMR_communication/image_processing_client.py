# KMR_communication/image_processing_client.py
import requests
import numpy as np
import json
import logging

# Assuming config.py defines these base URLs correctly (e.g., http://localhost:5000)
try:
    from . import config
    from . import utils # For ndarray_to_bytes
except ImportError:
    # Allow running standalone for testing? Or enforce package structure.
    print("Warning: Running image_processing_client outside of package context. Using hardcoded URLs.")
    import config # Adjust if config.py is elsewhere relative to this file when run standalone
    import utils
    # Hardcoded defaults if config fails
    config.DETECTION_SERVER_URL = getattr(config, "DETECTION_SERVER_URL", "http://localhost:5000")
    config.POSE_ESTIMATION_SERVER_URL = getattr(config, "POSE_ESTIMATION_SERVER_URL", "http://localhost:5000")


# Configure logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLIENT - %(levelname)s - %(message)s')


def send_image_to_server(image_bytes: bytes, prompt: str) -> list | None:
    """
    Sends image bytes to the object detection server's /process endpoint.

    Args:
        image_bytes (bytes): The image data as bytes (e.g., from utils.ndarray_to_bytes).
        prompt (str): The text prompt for object detection.

    Returns:
        list | None: A list of bounding boxes [[x1, y1, x2, y2], ...] or None on error.
                     Returns an empty list if detection runs but finds nothing.
    """
    url = f"{config.DETECTION_SERVER_URL}/process"
    logging.info(f"Sending image for detection to {url} with prompt: '{prompt}'")
    try:
        # Use a meaningful filename, though the server might ignore it
        files = {'image': ('image.png', image_bytes, 'image/png')}
        payload = {'prompt': prompt}

        response = requests.post(url, files=files, data=payload, timeout=30) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        logging.info(f"Detection server response: {result}")

        # Expecting {"bounding_boxes": [[x1, y1, x2, y2], ...]} or {"error": ...}
        if "error" in result:
             logging.error(f"Detection server returned error: {result['error']}")
             return None # Indicate error clearly
        return result.get('bounding_boxes', []) # Return empty list if key missing or no boxes

    except requests.exceptions.Timeout:
        logging.error(f"Detection request timed out connecting to {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending image for detection to {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during detection request: {e}", exc_info=True)
        return None

def send_for_pose_estimation(image_bytes: bytes, object_name: str, bounding_box: list | tuple, visualize: bool = False) -> dict | None:
    """
    Sends image bytes, object name, and bbox to the server's /estimate_pose endpoint.

    Args:
        image_bytes (bytes): The image data as bytes.
        object_name (str): The name of the object being estimated (should match Megapose model folder).
        bounding_box (list | tuple): The bounding box [x1, y1, x2, y2].
        visualize (bool): Whether to request visualization generation from the worker.

    Returns:
        dict | None: Pose estimation result (e.g., {'poses': [{'label':..., 'score':..., 'pose': [[...],...]} ]})
                     or None on error. Returns dict with 'error' key on server/worker error.
    """
    url = f"{config.POSE_ESTIMATION_SERVER_URL}/estimate_pose"
    logging.info(f"Sending image for pose estimation to {url} for object: '{object_name}', Visualize: {visualize}")
    try:
        files = {'image': ('image.png', image_bytes, 'image/png')}
        # Bbox must be sent as a JSON string
        bbox_json_string = json.dumps(bounding_box)
        payload = {
            'object_name': object_name,
            'bbox': bbox_json_string,
            'visualize': str(visualize).lower() # Send as 'true' or 'false' string
         }

        # Use a longer timeout for pose estimation as it involves a subprocess
        response = requests.post(url, files=files, data=payload, timeout=240) # e.g., 4 minutes timeout
        response.raise_for_status() # Check for HTTP errors

        result = response.json()
        logging.info(f"Pose estimation server response: {result}")

        # The server should return the worker's JSON directly or an error dict
        if "error" in result:
             logging.error(f"Pose estimation server/worker returned error: {result['error']}")
             # Return the error dict so the caller knows what happened
             return result
        # On success, expect a dict like {"poses": [...]}
        return result

    except requests.exceptions.Timeout:
        logging.error(f"Pose estimation request timed out connecting to {url}")
        return {"error": "Request timed out"} # Return error dict
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending image for pose estimation to {url}: {e}")
        return {"error": f"Connection error: {e}"} # Return error dict
    except Exception as e:
        logging.error(f"Unexpected error during pose estimation request: {e}", exc_info=True)
        return {"error": f"Client-side error: {e}"} # Return error dict

# --- Optional: Add simple test code ---
if __name__ == '__main__':
     # This allows testing the client functions directly
     # Requires a running server and a test image
     logging.info("Running image_processing_client standalone test...")
     test_image_path = Path(config.IMAGES_DIR) / "JustPickIt" / "test_image.png" # Adjust path
     test_prompt = "foam brick" # Adjust prompt
     test_object_name = "foam_brick" # Adjust object name (must match megapose folder)
     test_bbox = [100, 100, 300, 300] # Example bbox

     if not test_image_path.is_file():
          print(f"Test image not found: {test_image_path}")
     else:
          try:
               with open(test_image_path, "rb") as f:
                    img_bytes = f.read()

               # Test Detection
               print("\n--- Testing Detection ---")
               bboxes = send_image_to_server(img_bytes, test_prompt)
               if bboxes is not None:
                    print(f"Detected BBoxes: {bboxes}")
                    # Use the first detected bbox for pose estimation test (if any)
                    if bboxes:
                         test_bbox = bboxes[0]
                    else:
                         print("No objects detected, using default bbox for pose test.")
               else:
                    print("Detection failed.")

               # Test Pose Estimation (without visualization)
               print("\n--- Testing Pose Estimation (No Visualize) ---")
               pose_result = send_for_pose_estimation(img_bytes, test_object_name, test_bbox, visualize=False)
               if pose_result:
                    print(f"Pose Estimation Result: {pose_result}")
               else:
                    print("Pose estimation failed (returned None).")

                # Test Pose Estimation (with visualization)
               print("\n--- Testing Pose Estimation (Visualize=True) ---")
               pose_result_vis = send_for_pose_estimation(img_bytes, test_object_name, test_bbox, visualize=True)
               if pose_result_vis:
                    print(f"Pose Estimation Result (Vis=True): {pose_result_vis}")
                    if "visualization_output_dir" in pose_result_vis:
                         print(f"NOTE: Visualization images saved on server at: {pose_result_vis['visualization_output_dir']}")
               else:
                    print("Pose estimation failed (returned None).")


          except Exception as e:
               print(f"Error during client test: {e}")