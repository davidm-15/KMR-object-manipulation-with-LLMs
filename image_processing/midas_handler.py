from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np




class MiDaSHandler:
    def __init__(self):
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    def estimate_depth(self, image):
        # Prepare image for the model
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy array
        output = prediction.squeeze().cpu().numpy()
        # formatted = (output * 255 / np.max(output)).astype("uint8")
        return output