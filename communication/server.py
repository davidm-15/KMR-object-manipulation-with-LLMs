import sys
import os
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import torch
from PIL import Image


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose inference type based on environment variable (LISA or GLAMM)
INFERENCE_TYPE = os.getenv("INFERENCE_TYPE", "GLAMM")  # Default to LISA

# Load only the selected model
if INFERENCE_TYPE == "GLAMM":
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    print("Loading model...")
    print("Model ID:", model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)



def encode_image(image):
    _, img_encoded = cv2.imencode(".jpg", image)
    return base64.b64encode(img_encoded.tobytes()).decode("utf-8")


@app.route("/process", methods=["POST"])
def process_image():
    # Check if the request contains the required files
    if "image" not in request.files or "prompt" not in request.form:
        return jsonify({"error": "Missing image or prompt"}), 400

    # Read the image
    image_file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Convert the image to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image_path = "images/received_image.jpg"
    cv2.imwrite(output_image_path, image)

    # Read the prompt
    prompt = [request.form["prompt"]]
    print(f"Received prompt: {prompt}")


    # Convert the image to a PIL object
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Process using the selected model
    if INFERENCE_TYPE == "GLAMM":
        boxes, scores, labels = inference_glamm(processor, model, image, prompt)
        image_np = cv2.imread(output_image_path)
        print("Boxes:", boxes)
        print("Scores:", scores)
        print("Labels:", labels)

        bounding_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            bounding_boxes.append([int(x1), int(y1), int(x2), int(y2)])


        return jsonify({
            "bounding_boxes": bounding_boxes
        })
        



def inference_glamm(processor, model, image, texts):
    inputs = processor(images=image, text=texts, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]
    return boxes, scores, labels



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
