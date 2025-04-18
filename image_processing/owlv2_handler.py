# image_processing/owlv2_handler.py
import logging
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class Owlv2Handler:
    def __init__(self, device):
        self.device = device
        self.model_id = "google/owlv2-large-patch14-ensemble"
        self.processor = Owlv2Processor.from_pretrained(self.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(device)
        logging.info(f"Loaded OWLv2 model: {self.model_id}")

    def infer(self, image_file, prompts):
        image = Image.open(image_file).convert("RGB")
        texts = [prompts]  # List of prompts
        
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            threshold=0.1, 
            target_sizes=target_sizes
        )
        
        # Process the first image results
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        # Convert to list of integer coordinates
        bounding_boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in boxes]
        detected_labels = [prompts[label] for label in labels]
        confidences = [float(score) for score in scores]
        
        return {
            "bounding_boxes": bounding_boxes,
            "labels": detected_labels,
            "scores": confidences
        }


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dummy_image_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/image_1743171861_png.rf.683a83e10e917a24ea6d25dd507a6c93.jpg"

    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {inference_device}")
    
    owlv2_detector = Owlv2Handler(device=inference_device)
    detection_results = owlv2_detector.infer(dummy_image_path, ["mustard bottle", "ketchup bottle"])

    logging.info("Inference Results:")
    logging.info(f"  Bounding Boxes: {detection_results['bounding_boxes']}")
    logging.info(f"  Labels: {detection_results['labels']}")
    logging.info(f"  Scores: {detection_results['scores']}")
