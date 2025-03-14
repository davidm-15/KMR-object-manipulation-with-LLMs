import requests
import os
import cv2
import io
import numpy as np
from PIL import Image

SERVER_URL = "http://localhost:5000/process"
INPUT_FOLDER = "ImageProcessing/images/MustardBottle/inputs/"
OUTPUT_FOLDER = "ImageProcessing/images/MustardBottle/outputs/"
PROMPT = "a yellow mustard bottle"

def send_image_to_server(image_data, prompt):
    """ Sends image to server and returns bounding boxes """
    try:
        response = requests.post(
            SERVER_URL,
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

def process_images():
    """ Loads images, processes them, and saves results """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith((".jpg", ".png"))]

    print(f"Found {len(image_files)} images")

    for i, filename in enumerate(image_files):
        image_path = os.path.join(INPUT_FOLDER, filename)
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        bounding_boxes = send_image_to_server(image_data, PROMPT)
        if not bounding_boxes:
            print(f"Failed to process {filename}")
            continue

        image = cv2.imread(image_path)
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_path = os.path.join(OUTPUT_FOLDER, f"bbox_{i}.jpg")
        cv2.imwrite(output_path, image)
        print(f"Processed image saved: {output_path}")


if __name__ == "__main__":
    process_images()