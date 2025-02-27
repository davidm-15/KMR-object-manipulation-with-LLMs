import sys
import os

sys.path.append('../Inference/')
import LISA_Inference
import glamm_Inference

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import torch

app = Flask(__name__)

# Choose inference type based on environment variable (LISA or GLAMM)
INFERENCE_TYPE = os.getenv("INFERENCE_TYPE", "GLAMM")  # Default to LISA

# Load only the selected model
if INFERENCE_TYPE == "GLAMM":
    args = []
    model, tokenizer, clip_image_processor, transform, args = glamm_Inference.StartModel()
else:
    args = []
    model, tokenizer, clip_image_processor, transform, args = LISA_Inference.StartModel(args)

@app.route("/process", methods=["POST"])
def process_image():
    # Check if the request contains the required files
    if "image" not in request.files or "prompt" not in request.form:
        return jsonify({"error": "Missing image or prompt"}), 400

    # Read the image
    image_file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image_path = "images/received_image.jpg"
    cv2.imwrite(output_image_path, image)

    # Read the prompt
    prompt = request.form["prompt"]
    print(f"Received prompt: {prompt}")

    # Process using the selected model
    if INFERENCE_TYPE == "GLAMM":
        pred_masks, image_np = glamm_Inference.ProcessPromptImage(args, model, tokenizer, clip_image_processor, transform, prompt, image)
    else:
        pred_masks, image_np = LISA_Inference.ProcessPromptImage(args, model, tokenizer, clip_image_processor, transform, prompt, image)

    # Convert the processed image to a byte stream
    def encode_image(image):
        _, img_encoded = cv2.imencode(".jpg", image)
        return base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    processed_image_base64 = encode_image(image_np)

    # Convert the masks to a byte stream
    masks_base64 = []

    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        _, img_encoded = cv2.imencode(".jpg", pred_mask * 100)
        masks_base64.append(base64.b64encode(img_encoded.tobytes()).decode("utf-8"))

    # Return the processed image and masks as a JSON response
    return jsonify({
        "processed_image": processed_image_base64,
        "pred_masks": masks_base64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
