# image_processing/grounding_dino_handler.py
import logging
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINOHandler:
    def __init__(self, device):
        self.device = device
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device)
        logging.info(f"Loaded Grounding DINO model: {self.model_id}")

    def infer(self, image_file, prompt):
        image = Image.open(image_file).convert("RGB")
        inputs = self.processor(images=image, text=[prompt], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.6,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        print("Results:", results)
        bounding_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in results[0]["boxes"]]

        return {"bounding_boxes": bounding_boxes}


if __name__ == '__main__':

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




    dummy_image_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/image_1743174099_png.rf.b5f0de01f0a11e61cebca801a66a1755.jpg"



    yolo_weights = 'YOLO/outputs/experiment_1/weights/best.pt'

    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {inference_device}")
    
    yolo_detector = GroundingDINOHandler(device=inference_device) 

    detection_results = yolo_detector.infer(dummy_image_path, "mustard bottle")

    logging.info("Inference Results:")
    logging.info(f"  Bounding Boxes: {detection_results['bounding_boxes']}")
