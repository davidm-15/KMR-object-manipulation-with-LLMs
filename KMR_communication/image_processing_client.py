# image_processing_client.py
import requests
import numpy as np
import config
import utils # For ndarray_to_bytes

# Placeholder implementations: Replace with your actual client code
# This might involve sending HTTP requests, using gRPC, ROS topics, etc.

def send_image_to_server(image_bytes: bytes, item_name: str) -> list | None:
    """
    Sends image bytes to an object detection server.

    Args:
        image_bytes (bytes): The image data as bytes.
        item_name (str): The name of the item to detect (optional filter).

    Returns:
        list | None: A list of bounding boxes [(x1, y1, x2, y2), ...] or None on error.
                     Returns an empty list if detection runs but finds nothing.
    """
    print(f"Placeholder: Sending image for detection of '{item_name}'...")
    # Example using requests (replace with your actual implementation)
    try:
        files = {'image': ('image.png', image_bytes, 'image/png')}
        payload = {'item_name': item_name}
        # Ensure DETECTION_SERVER_URL is correctly defined in config.py
        url = f"{config.DETECTION_SERVER_URL}/detect"
        response = requests.post(url, files=files, data=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        # Assuming the server returns {'detections': [[x1, y1, x2, y2], ...]}
        print("Detection Response:", result)
        return result.get('detections', [])
    except requests.exceptions.RequestException as e:
        print(f"Error sending image for detection: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during detection request: {e}")
        return None

def send_for_pose_estimation(image_bytes: bytes, bounding_box: list | tuple, item_name: str) -> dict | None:
    """
    Sends image bytes and a bounding box to a 6D pose estimation server.

    Args:
        image_bytes (bytes): The image data as bytes.
        bounding_box (list | tuple): The bounding box [x1, y1, x2, y2].
        item_name (str): The name of the item being estimated.

    Returns:
        dict | None: Pose estimation result (e.g., {'poses': [{'pose': [[...],...]} ]}) or None on error.
                     Returns an empty dict if estimation runs but finds no pose.
    """
    print(f"Placeholder: Sending image and bbox for 6D pose estimation of '{item_name}'...")
    # Example using requests (replace with your actual implementation)
    try:
        files = {'image': ('image.png', image_bytes, 'image/png')}
        payload = {
            'item_name': item_name,
            'bbox': json.dumps(bounding_box) # Send bbox as JSON string
         }
        # Ensure POSE_ESTIMATION_SERVER_URL is correctly defined in config.py
        url = f"{config.POSE_ESTIMATION_SERVER_URL}/estimate"
        response = requests.post(url, files=files, data=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        # Assuming server returns {'poses': [{'pose': np.array, ...}]}
        # Important: Need to convert list back to numpy array if needed by calling code
        print("Pose Estimation Response:", result)
        if result and 'poses' in result and result['poses']:
             # Example: Convert list back to numpy array
             # result['poses'][0]['pose'] = np.array(result['poses'][0]['pose'])
             pass # Keep as list for now, conversion happens in sequence.py
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error sending image for pose estimation: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during pose estimation request: {e}")
        return None

# You might need to add imports like json if sending bbox as json string
import json