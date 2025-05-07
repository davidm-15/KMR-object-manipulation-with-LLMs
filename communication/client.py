import requests
import os
import cv2
import numpy as np
import utils.utils as utils
import time

SERVER_URL_DETECTION = "http://localhost:5000/process"
SERVER_URL_POSE = "http://localhost:5000/estimate_pose"

INPUT_FOLDER = "images/JustPickIt/"
OUTPUT_FOLDER = "images/JustPickIt/output/"
PROMPT = "cracker box"
# PROMPT = "mustard bottle"

# Run me with python -m communication.client

def send_image_to_server(image_data, prompt):
    """ Sends image to server and returns bounding boxes """
    try:
        response = requests.post(
            SERVER_URL_DETECTION,
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"prompt": prompt}
        )
        if response.status_code == 200:
            return response.json().get("bounding_boxes", [])
        else:
            print(f"Server Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return []

def send_for_pose_estimation(image_data, bounding_boxes, object_name):
    """ Sends the full image and bounding boxes to the server for 6D pose estimation """
    try:
        response = requests.post(
            SERVER_URL_POSE,
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"object_name": object_name, "bbox": str(bounding_boxes), "DoVis": "True"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Pose Estimation Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Failed to connect to pose estimation server: {e}")
        return None
    

def send_text_inference(text_input, max_new_tokens=128):
    """ Sends text to server for inference """
    try:
        response = requests.post(
            "http://localhost:5000/text_inference",
            data={
                "text_input": text_input,
                "max_new_tokens": str(max_new_tokens)
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Text Inference Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Failed to connect to text inference server: {e}")
        return None

def send_image_inference(image_data, prompt):
    """ Sends image to server for inference with prompt """
    try:
        response = requests.post(
            "http://localhost:5000/image_inference",
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"prompt": prompt}
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Image Inference Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Failed to connect to image inference server: {e}")
        return None

def process_images():
    """ Loads images, detects objects, and estimates 6D pose """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith((".jpg", ".png"))]

    print(f"Found {len(image_files)} images")

    for filename in image_files:
        image_path = os.path.join(INPUT_FOLDER, filename)   
        image = cv2.imread(image_path)

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        bounding_boxes = send_image_to_server(image_data, PROMPT)
        if not bounding_boxes:
            print(f"{PROMPT} not found in image: {filename}")
            continue
        print(f"{PROMPT} found in image: {filename} !!")
        print("Going for 6D Pose Estimation")

        print(f"Bounding Boxes: {bounding_boxes}")

        # Send full image with bounding boxes for 6D pose estimation
        pose_result = send_for_pose_estimation(image_data, bounding_boxes[0], PROMPT)
        if pose_result:
            print(f"6D Pose: {pose_result}")

        # Draw bounding boxes
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, image)
        print(f"Processed image saved: {output_path}")



if __name__ == "__main__":
    start_time = time.time()
    process_images()
    print("--- %s seconds ---" % (time.time() - start_time))

