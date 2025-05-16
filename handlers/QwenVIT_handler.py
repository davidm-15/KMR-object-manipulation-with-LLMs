import logging
import torch
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Owlv2Processor, Owlv2ForObjectDetection
from qwen_vl_utils import process_vision_info

class QwenOwlVit:
    def __init__(self, device, qwen_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize both Qwen VL and OWLv2 models."""
        self.device = device

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize Qwen model
        logging.info(f"Loading Qwen model: {qwen_model_name}")
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_name, torch_dtype="auto", device_map="auto"
        )
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)
        logging.info("Qwen model loaded successfully")

        # Initialize OWLv2 model
        self.owlvit_model_id = "google/owlv2-large-patch14-ensemble"
        logging.info(f"Loading OWLv2 model: {self.owlvit_model_id}")
        self.owlvit_processor = Owlv2Processor.from_pretrained(self.owlvit_model_id)
        self.owlvit_model = Owlv2ForObjectDetection.from_pretrained(self.owlvit_model_id).to(device)
        logging.info("OWLv2 model loaded successfully")

    def infer(self, image_file, text_prompts):
        """
        Combined inference pipeline:
        1. Use Qwen to determine if the object is in the image
        2. If present, use OWLv2 to localize the object

        Args:
            image_file (str): Path to the input image
            text_prompts (str or list): The object(s) to look for in the image

        Returns:
            dict: Results containing presence flag and localization if present
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        logging.info(f"Processing image: {image_file}")
        logging.info(f"Looking for: {text_prompts}")

        # Step 1: Check if object is present using Qwen
        qwen_prompt = f"This is an image from an industrial environment, I am looking for: {', '.join(text_prompts)}. Is it present in the image? Only answer with 1 if yes or 0 if not"
        qwen_response = self.qwen_image_text_inference(image_file, qwen_prompt)

        # Extract binary response (0 or 1)
        response_text = qwen_response["response"].strip()
        object_present = "1" in response_text

        logging.info(f"Qwen detection result: {'Present' if object_present else 'Not present'}")

        result = {
            "object_present": object_present,
            "qwen_response": response_text,
            "bounding_boxes": [],
            "labels": [],
            "scores": []
        }

        # Step 2: If object is present, use OWLv2 to localize it
        if object_present:
            logging.info(f"Localizing object with OWLv2")
            owlvit_results = self.owlvit_object_detection(image_file, text_prompts)

            result["bounding_boxes"] = owlvit_results["bounding_boxes"]
            result["labels"] = owlvit_results["labels"]
            result["scores"] = owlvit_results["scores"]

            logging.info(f"Found {len(result['bounding_boxes'])} instances with confidence: {result['scores']}")

        return result

    def qwen_image_text_inference(self, image_file, text_prompt, max_new_tokens=128):
        """
        Perform Qwen VL model inference on image and text input.

        Args:
            image_file (str): Path to the input image
            text_prompt (str): The text prompt to accompany the image
            max_new_tokens (int): Maximum number of tokens to generate

        Returns:
            dict: Contains the response text
        """
        image = Image.open(image_file).convert("RGB")
        image_path = os.path.join("/tmp", os.path.basename("tmp_image.jpg"))
        image.save(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)

        return {
            "response": output_text[0]
        }

    def owlvit_object_detection(self, image_file, prompts, threshold=0.1):
        """
        Perform object detection using OWLv2.

        Args:
            image_file (str): Path to the input image
            prompts (list): List of text prompts describing the objects to detect
            threshold (float): Confidence threshold for bounding boxes

        Returns:
            dict: Contains detected bounding boxes, labels, and their confidence scores
        """
        image = Image.open(image_file).convert("RGB")
        inputs = self.owlvit_processor(text=prompts, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.owlvit_model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.owlvit_processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        bounding_boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in boxes]
        detected_labels = [prompts[label] for label in labels]
        confidences = [float(score) for score in scores]

        # Sort results by confidence score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        bounding_boxes = [bounding_boxes[i] for i in sorted_indices]
        detected_labels = [detected_labels[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]

        return {
            "bounding_boxes": bounding_boxes,
            "labels": detected_labels,
            "scores": confidences
        }


class QwenOwlVit32(QwenOwlVit):
    """Combined vision handler with Qwen 32B model."""

    def __init__(self, device):
        """Initialize with the 32B model variant."""
        super().__init__(device, qwen_model_name="Qwen/Qwen2.5-VL-32B-Instruct")


class QwenOwlVit72(QwenOwlVit):
    """Combined vision handler with Qwen 72B model."""

    def __init__(self, device):
        """Initialize with the 72B model variant."""
        super().__init__(device, qwen_model_name="Qwen/Qwen2.5-VL-72B-Instruct")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_image_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/image_1743171861_png.rf.683a83e10e917a24ea6d25dd507a6c93.jpg"
    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {inference_device}")

    vision_handler = QwenOwlVit(device=inference_device)
    results = vision_handler.infer(test_image_path, ["mustard bottle", "ketchup bottle"])

    if results["object_present"]:
        logging.info(f"Object detected with {len(results['bounding_boxes'])} instances")
        logging.info(f"Bounding boxes: {results['bounding_boxes']}")
        logging.info(f"Labels: {results['labels']}")
        logging.info(f"Confidence scores: {results['scores']}")
    else:
        logging.info("Object not detected in the image")