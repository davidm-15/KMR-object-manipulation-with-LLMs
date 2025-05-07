from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import os
from PIL import Image
import glob
import json
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

class QwenHandler:
    def __init__(self, device, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize the Qwen model and processor."""
        print(f"Loading Qwen model: {model_name}")
        start_time = time.time()
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
        self.device = device
    
    def infer(self, image_file, text_prompt):
        text_prompt = f"This is an image from an industrial enviroment, I am looking for: {image_file, text_prompt}. Is it present in the image? Only answer with 1 if yes or 0 if not"
        response = self.image_text_inference(image_file, text_prompt)
        return {"object_in_scene": response["response"]}

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

        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        else:
            print(f"The file {image_path} does not exist")

        
        return {
            "response": output_text[0],
            "time": elapsed_time
        }

class QwenHandler32(QwenHandler):
    """Qwen handler specifically for the 32B model version."""
    
    def __init__(self, device):
        """Initialize with the 32B model variant."""
        super().__init__(device, model_name="Qwen/Qwen2.5-VL-32B-Instruct")

class QwenHandler72(QwenHandler):
    """Qwen handler specifically for the 32B model version."""
    
    def __init__(self, device):
        """Initialize with the 32B model variant."""
        super().__init__(device, model_name="Qwen/Qwen2.5-VL-72B-Instruct")



def postprocess_responses():
    """
    Processes LLM responses to determine if objects were found (1) or not (0).
    Reads responses from json file, sends each to the LLM for binary classification,
    and saves the results.
    """
    name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("Postprocessing responses...")
    qwen_handler = QwenHandler(model_name=name)
    
    # Load the results file
    try:
        with open("vision_models_comparison/qwen/qwen_results25-32B.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Results file not found.")
        return
    
    # Process each response
    binary_results = {}
    
    for item in results:
        image = item.get("image")
        prompt = item.get("prompt")
        response = item.get("response")
        
        if not all([image, prompt, response]):
            continue
            
        # Create a metaprompt for binary classification
        metaprompt = f'This was the user prompt: "{prompt}" and this was the answer: "{response}". I want you to answer 1 if the object was found in the image or 0 if not. Answer with this number only.'
        
        # Get binary classification from LLM
        result = qwen_handler.text_inference(metaprompt, max_new_tokens=16)
        binary_answer = result["response"].strip()
        
        # Validate the answer is only 0 or 1
        if binary_answer not in ["0", "1"]:
            print(f"Invalid response for {image}, prompt: {prompt[:30]}... - Got: {binary_answer}")
            binary_answer = "0"  # Default to not found if invalid
        
        # Store results
        if image not in binary_results:
            binary_results[image] = []
            
        binary_results[image].append({
            "prompt": prompt,
            "original_response": response,
            "binary_result": binary_answer
        })
    
    # Save binary results
    output_file = "vision_models_comparison/qwen/qwen_binary_results25-32B.json"
    with open(output_file, "w") as f:
        json.dump(binary_results, f, indent=2)
    
    print(f"Binary results saved to {output_file}")

    
def test_qwen():
    prompts = [
        "I am looking for a red box of cheezit, is it in the image?",
        "I am looking for a yellow mustard bottle, is it in the image?",
        "I am looking for a brown foam brick, is it in the image?",
        "I am looking for a gray junction box, is it in the image?",
        "I am looking for a tuna fish can, is it in the image?",
        "I am looking for a plug-in outlet expander, is it in the image?",
        "I am looking for a 6 by 2 lego brick, is it in the image?"
        "I am looking for a box of jello, is it in the image?",
        ]

    # Initialize model
    qwen_handler = QwenHandler(model_name="Qwen/Qwen2.5-VL-32B-Instruct")

    # Get all image files
    image_files = glob.glob("images/TestingImages/*.jpg")

    # Create results structure
    results = []

    for image_file in image_files:
        image_name = os.path.basename(image_file)
        print(f"Processing image: {image_name}")
        
        for prompt in prompts:
            print(f"  Testing prompt: {prompt[:30]}...")
            
            # Run inference
            try:
                response = qwen_handler.image_text_inference(image_file, prompt)
                
                # Store result
                results.append({
                    "image": image_name,
                    "prompt": prompt,
                    "response": response["response"],
                    "time": response["time"]
                })
                
                print(f"    Response time: {response['time']:.2f}s")
            except Exception as e:
                print(f"    Error processing {image_name} with prompt '{prompt}': {e}")
                results.append({
                    "image": image_name,
                    "prompt": prompt,
                    "error": str(e)
                })

    # Save results to JSON
    with open("qwen_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to qwen_results.json")


def visualise_results():
    """
    Visualize and evaluate the results of the object detection model.
    Calculates TP, TN, FP, FN and creates visualization plots.
    """

    # Load binary results
    with open("vision_models_comparison/qwen/qwen_binary_results25-32B.json", "r") as f:
        binary_results = json.load(f)

    # Load ground truth annotations
    with open("images/TestingImages/_annotations.coco.json", "r") as f:
        annotations = json.load(f)

    # Create mapping from filename to image_id
    filename_to_id = {img["file_name"]: img["id"] for img in annotations["images"]}
    
    # Create mapping from image_id to objects present in the image
    image_objects = {}
    for ann in annotations["annotations"]:
        img_id = ann["image_id"]
        category_id = ann["category_id"]
        if img_id not in image_objects:
            image_objects[img_id] = []
        image_objects[img_id].append(category_id)
    
    # Create mapping from category name to category_id
    category_map = {cat["name"]: cat["id"] for cat in annotations["categories"]}
    reverse_category_map = {cat["id"]: cat["name"] for cat in annotations["categories"]}
    
    # Create data structure for evaluation
    true_labels = []
    pred_labels = []
    
    # Keywords to map prompts to categories
    keyword_to_category = {
        "red box of cheezit": "cracker box",
        "yellow mustard bottle": "mustard bottle",
        "brown foam brick": "foam brick",
        "gray junction box": "gray box",
        "tuna fish can": "tuna fish can",
        "plug-in outlet expander": "plug-in outlet expander",
        "6 by 2 lego brick": "lego brick",
        "box of jello": "box of jello"
    }
    
    # Process each image and its predictions
    for image_file, results in binary_results.items():
        # Get image_id if available
        image_id = None
        for img in annotations["images"]:
            if img["file_name"] == image_file:
                image_id = img["id"]
                break
        
        if image_id is None:
            print(f"Warning: No ground truth found for {image_file}")
            continue
            
        # Get objects present in the image
        present_objects = image_objects.get(image_id, [])
        present_object_names = [reverse_category_map[obj_id] for obj_id in present_objects]
        
        # Evaluate each prompt result
        for result in results:
            prompt = result["prompt"]
            prediction = result["binary_result"]
            
            # Determine which category is being asked for
            category = None
            for keyword, cat_name in keyword_to_category.items():
                if keyword in prompt.lower():
                    category = cat_name
                    break
            
            if category is None:
                print(f"Warning: Could not map prompt to category: {prompt}")
                continue
                
            # Ground truth: is the object in the image?
            ground_truth = "1" if category in present_object_names else "0"
            
            # Add to evaluation data
            true_labels.append(int(ground_truth))
            pred_labels.append(int(prediction))
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Extract TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Qwen 2.5-VL-32B Object Detection')
    plt.savefig('vision_models_comparison/qwen/qwen32_confusion_matrix.png', bbox_inches='tight')
    
    # Create metrics bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
    plt.title('Performance Metrics for Qwen 2.5-VL-32B Object Detection')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics_df['Value']):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.savefig('vision_models_comparison/qwen/qwen32_metrics.png', bbox_inches='tight')
    
    # Print results
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    print("\nResults visualization saved to vision_models_comparison/qwen/")

if __name__ == "__main__":
    # Uncomment to run the test
    # test_qwen()
    
    # Uncomment to postprocess responses
    # postprocess_responses()
    
    visualise_results()
    
    pass