import requests
from PIL import Image
import io
import base64
import numpy as np
import os
import cv2

def main():
    mustard_bottle()

def sample():
    image_path = "images/image.jpg"  # Replace with actual image capture logic
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Define the prompt
    prompt = "A cat sitting on a chair"

    # Send the image and prompt to the server
    processed_image, masks = Inference(image_data, prompt)

def mustard_bottle():
    folder = "../ImageProcessing/images/MustardBottle/inputs/"
    output_folder = "../ImageProcessing/images/MustardBottle/outputs/"
    images = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                images.append(image_data)

    print(f"Found {len(images)} images")

    images.sort(key=lambda x: x[0])
    # Example usage
    for i, image_data in enumerate(images):
        bounding_boxes = Inference(image_data, "a yellow mustard bottle")
        # processed_image = processed_image.convert("RGB")
        # Convert PIL image to OpenCV format


        for j, bounding_box in enumerate(bounding_boxes):
            if bounding_box is None:
                print("Failed to get bounding box")
                continue
            # Draw bounding box on the image
            processed_image = Image.open(io.BytesIO(image_data))
            processed_image = processed_image.convert("RGB")
            processed_image = np.array(processed_image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            x1, y1, x2, y2 = bounding_box
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convert OpenCV image to PIL format
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image = Image.fromarray(processed_image)


            
            processed_image.save(f"{output_folder}/combined_with_bbox_{i}.jpg")
            print(f"Combined image with bounding box saved to {output_folder}/combined_with_bbox_{i}.jpg")



def Inference(image_data, prompt):
    """
        Function that sends the image and prompt to the server and receives the processed image and predicted masks.

        Args:
            image_data (bytes): The image data as bytes.
            prompt (str): The prompt to use for processing the image.
        
        Returns:
            processed_image (PIL.Image): The processed image.
            masks (list): A list of predicted masks as numpy arrays.
    """


    # Send the image and prompt to the server
    server_url = "http://localhost:5000/process"

    try:
        response = requests.post(
            server_url,
            files={"image": ("image.jpg", image_data, "image/jpeg")},
            data={"prompt": prompt}
        )

        if response.status_code == 200:
            # Decode the JSON response
            response_data = response.json()

            bounding_boxes = response_data["bounding_boxes"]
            return bounding_boxes

    except Exception as e:
        print(f"Failed to connect to the server: {e}")

    return None, None



if __name__ == "__main__":
    main()