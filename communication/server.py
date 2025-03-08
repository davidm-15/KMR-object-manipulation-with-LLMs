import sys
import os
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import DPTImageProcessor, DPTForDepthEstimation

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import torch
from PIL import Image


sys.path.append('../../megapose6d/src/megapose/scripts')
import run_inference_on_example
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.lib3d.transform import Transform


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

        if len(boxes) == 0:
            return jsonify({"error": "No objects detected"}), 400
        image_np = cv2.imread(output_image_path)
        print("Boxes:", boxes)
        print("Scores:", scores)
        print("Labels:", labels)

        bounding_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            bounding_boxes.append([int(x1), int(y1), int(x2), int(y2)])


        # TODO: Feed the bounding boxes through the megapose





        return jsonify({
            "bounding_boxes": bounding_boxes
        })
        

def megapose_inference(image, object_data):
    depth_image = depth_estimation(image)
    model_name = "megapose-1.0-RGB-multi-hypothesis"


    output = run_inference_on_example.my_inference(image, depth_image, camera_data, object_data, model_name, example_dir)
    
    
    
    
    return output


def depth_estimation(image):
    depth_model = "Intel/dpt-large"
    processor = DPTImageProcessor.from_pretrained(depth_model)
    model = DPTForDepthEstimation.from_pretrained(depth_model)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth


    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()


    return outputs


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
