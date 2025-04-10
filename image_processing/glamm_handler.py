# image_processing/glamm_handler_direct.py

# 1) Imports - Exactly as you provided
import os
import sys
# This assumes you run the script from a directory where '../groundingLMM' points correctly
sys.path.append('../groundingLMM')
import app

import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch # Needed for mask conversion .cpu()

# No json import needed for the handler class itself


class GlammHandler:
    """
    Direct translation of the provided Glamm inference code into a handler class,
    outputting bounding boxes. Minimal modifications.
    """
    def __init__(self):
        """
        Initializes the model using the logic from the original StartModel function.
        """
        print("Initializing GlammHandler...")
        # 2) Init - Directly from your StartModel function
        arg_list = [] # Use default arguments as in the original StartModel call
        self.args = app.parse_args(arg_list)
        self.tokenizer = app.setup_tokenizer_and_special_tokens(self.args)
        model_unprepared = app.initialize_model(self.args, self.tokenizer)
        self.model = app.prepare_model_for_inference(model_unprepared, self.args)
        # Assuming CLIPImageProcessor exists in app - adjust class name if needed based on your app.py
        self.clip_image_processor = app.CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = app.ResizeLongestSide(self.args.image_size)
        self.model.eval()
        print("GlammHandler initialization complete.")
        # No return statement, variables are stored in self

    def _masks_to_boxes(self, masks: list) -> list:
        """
        Converts segmentation masks (list of tensors) to bounding boxes [x_min, y_min, x_max, y_max].
        Uses np.where, min, max.
        """
        bounding_boxes = []
        if not masks or masks[0] is None: # Check if inference returned masks
            print("Warning: No masks received from inference.")
            return []

        for i, mask_tensor in enumerate(masks):
            if mask_tensor is None:
                print(f"Warning: Mask {i} is None.")
                continue

            # Ensure tensor is on CPU and convert to numpy
            try:
                # Squeeze removes single dimensions, cpu moves to CPU, numpy converts
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
            except Exception as e:
                print(f"Error converting mask {i} to numpy: {e}")
                continue

            # Find coordinates of non-zero pixels (mask region)
            indices = np.where(mask_np > 0)
            # Check if we have the expected number of arrays (should be 2 for 2D masks)
            if len(indices) == 2:
                y_indices, x_indices = indices
            else:
                print(f"Warning: Unexpected mask format for mask {i}. Skipping.")
                continue

            if y_indices.size == 0 or x_indices.size == 0:
                # print(f"Mask {i} is empty after thresholding.")
                continue # Skip empty masks

            # Calculate bounding box coordinates using min/max
            x_min = int(np.min(x_indices))
            y_min = int(np.min(y_indices))
            x_max = int(np.max(x_indices))
            y_max = int(np.max(y_indices))

            # Append bounding box in [x_min, y_min, x_max, y_max] format
            bounding_boxes.append([x_min, y_min, x_max, y_max])

        return bounding_boxes

    def infer(self, image_path: str, prompt: str) -> dict:
        """
        Performs inference on a single image file with a text prompt and returns bounding boxes.

        Args:
            image_path (str): Path to the input image file.
            prompt (str): The text prompt describing the object(s) to detect.

        Returns:
            dict: A dictionary containing the key "bounding_boxes" with a list of
                  detected bounding boxes in [x_min, y_min, x_max, y_max] format.
                  Returns {"bounding_boxes": []} if no objects are detected.
        """
        # 3) Infer - Logic adapted from ProcessPromptImage and main loop

        # Load image using PIL
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
             print(f"ERROR: Image file not found at '{image_path}'")
             return {"bounding_boxes": []} # Return empty if image not found

        # Prepare inputs for app.inference - similar to ProcessPromptImage
        all_inputs = {'image': image, 'boxes': []}
        follow_up = False  # As per original script
        generate = False   # As per original script

        # Call the core inference function using the initialized components (self.*)
        # We only need pred_masks from the output tuple for bounding boxes
        # The other outputs (_output_image, _markdown_out, _output_ids) are ignored for now
        _output_image, _markdown_out, _output_ids, pred_masks = app.inference(
            prompt,
            all_inputs=all_inputs,
            follow_up=follow_up,
            generate=generate,
            args=self.args,
            global_enc_processor=self.clip_image_processor, # Use self.clip_image_processor
            transform=self.transform,                       # Use self.transform
            tokenizer=self.tokenizer,                       # Use self.tokenizer
            model=self.model                                # Use self.model
        )

        # Convert the predicted segmentation masks to bounding boxes
        bounding_boxes = self._masks_to_boxes(pred_masks)

        # Return the bounding boxes in the specified dictionary format
        return {"bounding_boxes": bounding_boxes}


# --- Main execution block for testing ---
if __name__ == '__main__':
    print("--- Running GlammHandler Direct Test ---")

    # --- Define Test Parameters ---
    # IMPORTANT: Replace with an actual image path and a relevant prompt for that image
    # Using the path from your original main() for consistency, but make sure it exists
    # test_image_file = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/LISA/images/img_0.jpg"
    # test_prompt = "person"

    test_image_file = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/image_1743172190_png.rf.4d1791406f3ad11e540332b31dd976a0.jpg"
    test_prompt = "Please segment yellow mustard bottle"

    if not os.path.exists(test_image_file):
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: Test image file does NOT exist: '{test_image_file}'")
        print(f"Please update the 'test_image_file' variable in the script.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        # Initialize the handler
        handler = GlammHandler()

        # Run inference
        print(f"\nRunning inference on '{test_image_file}' with prompt: '{test_prompt}'...")
        results = handler.infer(test_image_file, test_prompt)

        # Print results
        print(f"\n--- Inference Results ---")
        if results:
             print(f"  Detected Bounding Boxes: {results['bounding_boxes']}")
        else:
             print("  Inference returned None or an error occurred.")
        print(f"-------------------------\n")


    # Draw bounding boxes on image and save it
    if results["bounding_boxes"]:
        
        # Create a copy of the image to draw on
        img = Image.open(test_image_file).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Draw each bounding box
        for box in results["bounding_boxes"]:
            x_min, y_min, x_max, y_max = box
            # Draw rectangle with red outline (width=2)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        
            # Create output directory if it doesn't exist
            output_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/glamm"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output filename based on input filename
            base_filename = os.path.basename(test_image_file)
            output_filename = os.path.join(output_dir, f"bbox_{base_filename}")
            
            # Save the image
            img.save(output_filename)
            print(f"Image with bounding boxes saved to: {output_filename}")
    else:
        print("No bounding boxes detected, image not saved.")
    

    print("--- GlammHandler Direct Test Finished ---")