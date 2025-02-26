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
        processed_image, masks = Inference(image_data, "An yellow mustard bottle, If there is not any dont segment anything.")
        # processed_image = processed_image.convert("RGB")
        # Convert PIL image to OpenCV format
        processed_image = np.array(processed_image)

        # Display the image using OpenCV
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image_pil = Image.fromarray(processed_image_rgb)
        processed_image_pil.save(f"{output_folder}/output_{i}.jpg")

        print(f"Processed image saved to {output_folder}/output_{i}.jpg")
        for j, mask in enumerate(masks):
            mask_image = Image.fromarray(mask)
            mask_image.save(f"{output_folder}/mask_{i}_{j}.jpg")

            # Combine the mask with the processed image
            combined_image = processed_image.copy()
            for mask in masks:
                resized_mask = cv2.resize(mask, (combined_image.shape[1], combined_image.shape[0]))
                if len(resized_mask.shape) == 2:  # If mask is single channel, convert to 3 channels
                    resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
                # Create a red mask
                red_mask = np.zeros_like(resized_mask)
                red_mask[:, :, 0] = resized_mask[:, :, 0]  # Red channel
                red_mask[:, :, 1] = 0  # Green channel
                red_mask[:, :, 2] = 0  # Blue channel

                combined_image = cv2.addWeighted(combined_image, 1, red_mask, 0.5, 0)

            # Convert combined image to PIL format and save
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
            combined_image_pil = Image.fromarray(combined_image)
            combined_image_pil.save(f"{output_folder}/combined_{i}.jpg")

            # Create a bounding box around the object
            for mask in masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(combined_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert combined image to PIL format and save
            combined_image_pil = Image.fromarray(combined_image)
            combined_image_pil.save(f"{output_folder}/combined_with_bbox_{i}.jpg")

            print(f"Combined image with bounding box saved to {output_folder}/combined_with_bbox_{i}.jpg")

            print(f"Combined image saved to {output_folder}/combined_{i}.jpg")


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

            # Decode and save the processed image
            processed_image_base64 = response_data["processed_image"]
            processed_image_bytes = base64.b64decode(processed_image_base64)
            processed_image = Image.open(io.BytesIO(processed_image_bytes))

            # Decode and save the predicted masks
            masks = []
            masks_base64 = response_data["pred_masks"]
            for i, mask_base64 in enumerate(masks_base64):
                mask_bytes = base64.b64decode(mask_base64)
                mask_image = Image.open(io.BytesIO(mask_bytes))
                masks.append(np.array(mask_image))

            return processed_image, masks

    except Exception as e:
        print(f"Failed to connect to the server: {e}")

    return None, None



if __name__ == "__main__":
    main()