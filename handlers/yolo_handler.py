# image_processing/yolo_handler.py
import logging
import torch
from PIL import Image
from ultralytics import YOLO
import os
import tempfile
import cv2

# PIL is often useful for image manipulation, though YOLO().predict can handle paths directly
# from PIL import Image



class YOLOHandler:
    """
    Handler class for performing inference with YOLO models
    using the ultralytics library.
    """
    def __init__(self, device):
        self.device = device
        self.weights_path = 'YOLO/outputs/experiment_13/weights/best.pt'

        self.model = YOLO(self.weights_path)
        logging.info(f"Loaded YOLO model from: {self.weights_path} onto device: {self.device}")
        self.class_names = self.model.names
        logging.info(f"Model classes: {self.class_names}")


    def infer(self, image_file: str, prompt, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        image = Image.open(image_file).convert("RGB")

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        if not results or len(results) == 0:  
            logging.warning(f"No results returned by YOLO model for image: {image_file}")
            return {
                "bounding_boxes": [],
                "scores": [],
                "class_ids": [],
                "class_names": []
            }

        result = results[0] # Get the Results object for the first image

        boxes_tensor = result.boxes.xyxy
        conf_tensor = result.boxes.conf
        cls_tensor = result.boxes.cls

        # Move tensors to CPU and convert to lists
        bounding_boxes = boxes_tensor.cpu().numpy().astype(int).tolist()
        confidences = conf_tensor.cpu().numpy().tolist()
        class_ids = cls_tensor.cpu().numpy().astype(int).tolist()

        # Map class IDs to class names
        class_names = [self.class_names[cls_id] for cls_id in class_ids]

        if prompt in class_names:
            # Filter results based on the prompt
            filtered_indices = [i for i, name in enumerate(class_names) if name == prompt]
            bounding_boxes = [bounding_boxes[i] for i in filtered_indices]
            confidences = [confidences[i] for i in filtered_indices]
            class_ids = [class_ids[i] for i in filtered_indices]
            class_names = [class_names[i] for i in filtered_indices]
        else:
            # If the prompt is not in class names, return all results
            logging.info(f"Prompt '{prompt}' not found in class names. Returning empty box.")
            # Note: This may not be the desired behavior; adjust as needed.
            bounding_boxes = []
            confidences = []
            class_ids = []
            class_names = []


        return {
            "bounding_boxes": bounding_boxes,
            "scores": confidences,
            "class_ids": class_ids,
            "class_names": class_names
        }
# Example Usage (assuming you run this script directly or import it)
if __name__ == '__main__':

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




    dummy_image_path = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/megapose_objects/foam brick/img.png"



    yolo_weights = 'YOLO/outputs/experiment_1/weights/best.pt'

    inference_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {inference_device}")
    
    yolo_detector = YOLOHandler(device=inference_device) 

    detection_results = yolo_detector.infer(dummy_image_path, "gray box", conf_threshold=0.25)

    logging.info("Inference Results:")
    logging.info(f"  Bounding Boxes: {detection_results['bounding_boxes']}")
    logging.info(f"  Confidences: {detection_results['confidences']}")
    logging.info(f"  Class IDs: {detection_results['class_ids']}")
    logging.info(f"  Class Names: {detection_results['class_names']}")

    # Save the image with bounding boxes

    def save_image_with_bbox(image_path, bboxes, class_names, output_path=None):
        """
        Save the image with bounding boxes drawn on it
        
        Args:
            image_path: Path to the original image
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            class_names: List of class names corresponding to the bounding boxes
            output_path: Path to save the output image (if None, saves in the same folder with '_bbox' suffix)
        """
        # Read the image
        img = cv2.imread(image_path)
        
        # Draw bounding boxes
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_names[i]}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Generate output path if not provided
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_bbox{ext}"
        
        # Save the image
        cv2.imwrite(output_path, img)
        logging.info(f"Image with bounding boxes saved to: {output_path}")
        return output_path

    # Save the image with bounding boxes if any were detected
    if detection_results['bounding_boxes']:
        saved_image = save_image_with_bbox(
            dummy_image_path, 
            detection_results['bounding_boxes'], 
            detection_results['class_names']
        )
        logging.info(f"Saved annotated image to: {saved_image}")
    else:
        logging.info("No bounding boxes to draw.")


