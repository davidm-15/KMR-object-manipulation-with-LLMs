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
from skimage import measure
import cv2
import os
import random
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
                mask_np = mask_tensor
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

    def infer(self, image_path: str, prompt: str, **kwargs) -> dict:
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

        numm = kwargs.get('image_id', 0)  # Default to 0 if not provided

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
        # Assuming pred_masks is a list of torch tensors
        # Save the predicted masks as .npy files for debugging
        output_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/glamm"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'pred_masks_{numm}.npy'), pred_masks[0].cpu().numpy())

        pred_masks = pred_masks[0].squeeze().cpu().numpy()
        print(f"Predicted masks: {pred_masks}")
        original_values = np.unique(pred_masks[0])
        print(f"Original values in pred_masks: {original_values}")

        original_pred_masks = [pred_masks.copy()]  # Keep original for debugging

        # Convert the predicted segmentation masks to bounding boxes
        # Assuming pred_masks is a list of torch tensors
        mask = pred_masks
        mask = np.where(mask > 1, 255, 0).astype(np.uint8)  # Thresholding to create a binary mask
        print(f"{np.unique(mask)}")

        if len(np.unique(mask)) <= 1:
            print("No objects detected in the image.")
            return {"bounding_boxes": []}
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        biggest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        out = np.where(labels == biggest, 255, 0).astype(np.uint8)

        pred_masks = [out]




        bounding_boxes = self._masks_to_boxes(pred_masks)


        # Return the bounding boxes in the specified dictionary format
        return {"bounding_boxes": bounding_boxes, "pred_masks": pred_masks, "original_masks": [original_pred_masks]}


# --- Main execution block for testing ---
if __name__ == '__main__':
    print("--- Running GlammHandler Direct Test ---")

    # --- Define Test Parameters ---
    # IMPORTANT: Replace with an actual image path and a relevant prompt for that image
    # Using the path from your original main() for consistency, but make sure it exists
    # test_image_file = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/LISA/images/img_0.jpg"
    # test_prompt = "person"

    test_image_file = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/image_1743172190_png.rf.4d1791406f3ad11e540332b31dd976a0.jpg"
    # Directory containing the images
    image_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/"

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Set the seed for reproducibility
    random.seed(42)
    # Randomly select 20 image files
    num_images = min(20, len(image_files))  # Ensure we don't try to select more images than available
    test_image_files = random.sample(image_files, num_images)

    test_prompt = "If there is an yellow mustard bottle, segment it."
    
    handler = GlammHandler()

    for image_id, test_image_file in enumerate(test_image_files):
        # Construct the full path to the image file
        test_image_file = os.path.join(image_dir, test_image_file)

        if not os.path.exists(test_image_file):
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"ERROR: Test image file does NOT exist: '{test_image_file}'")
            print(f"Please update the 'test_image_file' variable in the script.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            # Initialize the handler
            
            # Run inference
            print(f"\nRunning inference on '{test_image_file}' with prompt: '{test_prompt}'...")
            results = handler.infer(test_image_file, test_prompt, image_id=image_id)

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

            # Save the predicted masks as well
            for i, mask_tensor in enumerate(results["pred_masks"]):
                # Ensure tensor is on CPU and convert to numpy
                mask_np = mask_tensor

                # Create output directory if it doesn't exist
                output_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/glamm"
                os.makedirs(output_dir, exist_ok=True)

                # Create output filename based on input filename
                base_filename = os.path.basename(test_image_file)
                mask_filename = os.path.join(output_dir, f"mask_{base_filename.split('.')[0]}_{i}.png")

                # Convert the numpy mask to a PIL image and save it
                mask_image = Image.fromarray(mask_np)  # Scale to 0-255
                mask_image.save(mask_filename)
                print(f"Mask saved to: {mask_filename}")

            # # Save the original predicted masks as well
            # for i, mask_tensor in enumerate(results["original_masks"]):
            #     # Ensure tensor is on CPU and convert to numpy
            #     mask_np = mask_tensor

            #     # Create output directory if it doesn't exist
            #     output_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/YOLO/train/glamm"
            #     os.makedirs(output_dir, exist_ok=True)

            #     # Create output filename based on input filename
            #     base_filename = os.path.basename(test_image_file)
            #     mask_filename = os.path.join(output_dir, f"original_mask_{base_filename.split('.')[0]}_{i}.png")

            #     # Convert the numpy mask to a PIL image and save it
            #     mask_image = Image.fromarray(mask_np * 255)  # Scale to 0-255
            #     mask_image.save(mask_filename)
            #     print(f"Original mask saved to: {mask_filename}")
        else:
            print("No bounding boxes detected, image not saved.")
        

        print("--- GlammHandler Direct Test Finished ---")