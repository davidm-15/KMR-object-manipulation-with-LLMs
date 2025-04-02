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
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        bounding_boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in results[0]["boxes"]]

        return {"bounding_boxes": bounding_boxes}
