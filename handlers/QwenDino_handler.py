import logging
import torch
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForZeroShotObjectDetection
from qwen_vl_utils import process_vision_info
import time
import re


class QwenDino:
    def __init__(self, device, qwen_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize both Qwen VL and Grounding DINO models."""
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
        
        # Initialize Grounding DINO model
        self.dino_model_id = "IDEA-Research/grounding-dino-base"
        logging.info(f"Loading Grounding DINO model: {self.dino_model_id}")
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_id
        ).to(device)
        logging.info("Grounding DINO model loaded successfully")

    def parse_response(self, response):
        presence = int(re.search(r"<presence:\s*(\d)>", response).group(1))
        description = re.search(r"<description:\s*(.*?)>", response).group(1)
        return presence, description

    def infer(self, image_file, text_prompt):
        """
        Combined inference pipeline:
        1. First use Qwen to determine if the object is in the image
        2. If present (1), use Grounding DINO to localize the object
        
        Args:
            image_file (str): Path to the input image
            text_prompt (str): The object to look for in the image
            
        Returns:
            dict: Results containing presence flag and localization if present
        """
        logging.info(f"Processing image: {image_file}")
        logging.info(f"Looking for: {text_prompt}")
        
        # Step 1: Check if object is present using Qwen
        qwen_prompt = (
            f"This is an image from an industrial environment. I am looking for: {text_prompt}. "
            f"Is it present in the image? Reply in the following format:\n"
            f"<presence: 1 or 0>\n"
            f"<description: short description if present, describe only the object not the suroundings, otherwise 'none'>"
        )
        print(f"{qwen_prompt=}")

        qwen_response = self.qwen_image_text_inference(image_file, qwen_prompt)["response"]
     
        print(f"{qwen_response=}")
        
        presence, description = self.parse_response(qwen_response)
        

        logging.info(f"Qwen detection result: {'Present' if presence else 'Not present'}")
        logging.info(f"Qwen object description: {description}")

        result = {
            "object_present": presence,
            "qwen_response": description,
            "bounding_boxes": [],
            "scores": []
        }
        
        # Step 2: If object is present, use Grounding DINO to localize it
        if presence:
            logging.info(f"Localizing object with Grounding DINO")
            dino_results = self.dino_object_detection(image_file, description)
            
            result["bounding_boxes"] = dino_results["bounding_boxes"]
            result["scores"] = dino_results["scores"]
            
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
            dict: Contains the response text and processing time
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
        inputs = inputs.to(self.device)

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

    def dino_object_detection(self, image_file, prompt, box_threshold=0.6, text_threshold=0.3):
        """
        Perform object detection using Grounding DINO.
        
        Args:
            image_file (str): Path to the input image
            prompt (str): Text prompt describing the object to detect
            box_threshold (float): Confidence threshold for bounding boxes
            text_threshold (float): Text matching threshold
            
        Returns:
            dict: Contains detected bounding boxes and their confidence scores
        """
        image = Image.open(image_file).convert("RGB")
        inputs = self.dino_processor(images=image, text=[prompt], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )

        scores = results[0]["scores"].tolist()
        bounding_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in results[0]["boxes"]]

        return {"bounding_boxes": bounding_boxes, "scores": scores}

    def text_inference(self, text_input, max_new_tokens=128):
        """
        Perform inference on text-only input.
        
        Args:
            text_input (str): The text prompt
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The generated text response
        """
        start_time = time.time()
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": text_input}]}
        ]
        
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.qwen_processor(
            text=[text],
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
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            "response": output_text[0],
            "time": elapsed_time
        }


    def image_text_inference(self, image_file, text_prompt, max_new_tokens=128):
        """
        Perform inference on image and text input.
        
        Args:
            image_path (str): Path to the input image
            text_prompt (str): The text prompt to accompany the image
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            dict: Contains the response text and processing time
        """
        start_time = time.time()

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
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        else:
            print(f"The file {image_path} does not exist")

        
        return {
            "response": output_text[0],
            "time": elapsed_time
        }


class QwenDino32(QwenDino):
    """Combined vision handler with Qwen 32B model."""
    
    def __init__(self, device):
        """Initialize with the 32B model variant."""
        super().__init__(device, qwen_model_name="Qwen/Qwen2.5-VL-32B-Instruct")


class QwenDino72(QwenDino):
    """Combined vision handler with Qwen 72B model."""
    
    def __init__(self, device):
        """Initialize with the 72B model variant."""
        super().__init__(device, qwen_model_name="Qwen/Qwen2.5-VL-72B-Instruct")


if __name__ == '__main__':
    # Usage example
    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {inference_device}")
    
    # Create the combined handler
    vision_handler = QwenDino(device=inference_device)
    
    # Test image path
    test_image_path = "path/to/your/test/image.jpg"
    
    # Run inference
    results = vision_handler.infer(test_image_path, "mustard bottle")
    
    if results["object_present"]:
        logging.info(f"Object detected with {len(results['bounding_boxes'])} instances")
        logging.info(f"Bounding boxes: {results['bounding_boxes']}")
        logging.info(f"Confidence scores: {results['scores']}")
    else:
        logging.info("Object not detected in the image")