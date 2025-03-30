import logging
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class LISAHandler:
    def __init__(self, device):
        self.device = device
        self.model_id = "xinlai/LISA-13B-llama2-v1"
        
        # Load processor and model (assuming LLaMA-based multimodal support)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        ).to(device)

        logging.info(f"Loaded LISA model: {self.model_id}")

    def infer(self, image_file, prompt):
        image = Image.open(image_file).convert("RGB")
        
        # Process input (assumes multimodal compatibility)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)

        # Decode the generated response
        response_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response_text}
