import os
import sys
sys.path.append('../../groundingLMM')
import app

import numpy as np
from PIL import Image
import json


def StartModel():
    args = app.parse_args(sys.argv[1:])
    tokenizer = app.setup_tokenizer_and_special_tokens(args)
    model = app.initialize_model(args, tokenizer)
    model = app.prepare_model_for_inference(model, args)
    global_enc_processor = app.CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = app.ResizeLongestSide(args.image_size)
    model.eval()
    return model, tokenizer, global_enc_processor, transform, args


def ProcessPromptImage(args, model, tokenizer, clip_image_processor, transform, prompt, image):
    all_inputs = {'image': image, 'boxes': []}
    follow_up = False
    generate = False
    output_image, markdown_out, output_ids, pred_masks = app.inference(prompt, all_inputs, follow_up, generate, args, clip_image_processor, transform, tokenizer, model)
    return pred_masks, output_image


def main():
    model, tokenizer, global_enc_processor, transform, args = StartModel()

    # Load the JSON file
    with open('/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/LISA/PromptsAndImages.json', 'r') as file:
        data = json.load(file)

    # Extract the necessary information from the JSON data

    for item in data:
        input_str = item.get('prompt')
        all_inputs = {'image': item.get('image'), 'boxes': []}
        output = item.get('output')

        input_str = input_str
        follow_up = False
        generate = False
        output_image, markdown_out, output_ids, pred_masks = app.inference(input_str, all_inputs, follow_up, generate, args, global_enc_processor, transform, tokenizer, model)

        # Convert the output image array to a PIL Image
        output_image_array = np.array(output_image, dtype=np.uint8)
        output_pil_image = Image.fromarray(output_image_array)

        # Save the PIL Image to a file
        output_pil_image.save(output)

        # Save the predicted masks to a file
        for i, mask in enumerate(pred_masks):
            pred_masks_array = np.array(mask.squeeze().cpu(), dtype=np.uint8)
            pred_masks_pil_image = Image.fromarray(pred_masks_array)
            pred_masks_pil_image.save(output.replace('.png', f'_mask_{i}.png'))

        print("Output image saved to: ", output_image)
        print("Markdown output: ", markdown_out)
        print("Done!")


if __name__ == '__main__':
    main()

