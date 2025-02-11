from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import base64
import torch
import sys
sys.path.append('../')
from LISA import ProccesMany 

app = Flask(__name__)

# Load the model and other components when the server starts
if __name__ == "__main__":
    args = []
    model, tokenizer, clip_image_processor, transform, args = ProccesMany.StartModel(args)

@app.route("/process", methods=["POST"])
def process_image():
    # Check if the request contains the required files
    if "image" not in request.files or "prompt" not in request.form:
        return jsonify({"error": "Missing image or prompt"}), 400

    # Read the image
    image_file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("received_image.jpg", image)

    # Read the prompt
    prompt = request.form["prompt"]
    print(f"Received prompt: {prompt}")

    # Process the image and prompt using your model
    pred_masks, image_np = ProccesMany.ProcessPromptImage(args, model, tokenizer, clip_image_processor, transform, prompt, image)

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