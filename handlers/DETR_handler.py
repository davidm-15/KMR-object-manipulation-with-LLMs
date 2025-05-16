import logging
from PIL import Image
import torch
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection

class DetrHandler:
    def __init__(self, device):
        self.device = device
        self.model_id = "facebook/deformable-detr-detic"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = DeformableDetrForObjectDetection.from_pretrained(self.model_id).to(device)
        logging.info(f"Loaded Deformable DETR Detic model: {self.model_id}")

    def infer(self, image_file, prompt=None):
        """
        Run inference on an image. For compatibility with other handlers,
        this accepts a prompt parameter but doesn't use it as Deformable DETR
        has a fixed set of classes.
        
        Args:
            image_file: Path to the image file
            prompt: Not used in this handler, kept for API compatibility
            
        Returns:
            Dictionary with detection results
        """
        image = Image.open(image_file).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        # Use a fixed threshold of 0.5 (float)
        threshold = 0.5
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes,
            threshold=threshold
        )
        
        # Process the first image results
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        # Convert to list of integer coordinates
        bounding_boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in boxes]
        detected_labels = [self.model.config.id2label[label.item()] for label in labels]
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


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example image path
    dummy_image_path = "path/to/your/image.jpg"

    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {inference_device}")
    
    detr_detector = DeformableDetrHandler(device=inference_device)
    detection_results = detr_detector.infer(dummy_image_path, threshold=0.7)

    logging.info("Inference Results:")
    logging.info(f"  Bounding Boxes: {detection_results['bounding_boxes']}")
    logging.info(f"  Labels: {detection_results['labels']}")
    logging.info(f"  Scores: {detection_results['scores']}")