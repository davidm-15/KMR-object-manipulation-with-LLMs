import requests
from PIL import Image
import io
import base64
import numpy as np

# Define the server URL
server_url = "http://localhost:5000/process"

# Load or capture an image
image_path = "images/image.jpg"  # Replace with actual image capture logic
with open(image_path, "rb") as image_file:
    image_data = image_file.read()

# Define the prompt
prompt = "A cat sitting on a chair"

# Send the image and prompt to the server
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
        output_image_path = "images/processed_image.jpg"
        processed_image.save(output_image_path)
        print("Processed image saved as 'images/processed_image.jpg'")

        # Decode and save the predicted masks
        masks_base64 = response_data["pred_masks"]
        for i, mask_base64 in enumerate(masks_base64):
            mask_bytes = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_bytes))
            output_mask_path = f"images/pred_mask_{i}.jpg"
            mask_image.save(output_mask_path)
            print(f"Predicted mask {i} saved as '{output_mask_path}'")
                  
except Exception as e:
    print(f"Failed to connect to the server: {e}")