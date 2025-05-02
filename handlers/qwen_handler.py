from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import os

class QwenHandler:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize the Qwen model and processor."""
        print(f"Loading Qwen model: {model_name}")
        start_time = time.time()
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
    
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
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            "response": output_text[0],
            "time": elapsed_time
        }
    
    def image_text_inference(self, image_path, text_prompt, max_new_tokens=128):
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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            "response": output_text[0],
            "time": elapsed_time
        }
    

