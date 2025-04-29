from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import os
import glob

# Start timing for entire process
start_time_total = time.time()
print("Starting processing at:", time.strftime("%H:%M:%S", time.localtime(start_time_total)))

# Load the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

# Get all images from the directory
image_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/images/LLMPreformance/"
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

results = {}

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    print(f"\nProcessing image: {image_name}")
    
    # Start timing for this image
    start_time_image = time.time()
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": " I am analyzing an image taken in an industrial environment. Can you detect any of the following items in the image: yellow mustard bottle, 6x2 LEGO flat brick, outlet expander (plug adapter), brown foam brick, box of Cheez-Its, tuna fish can, box of Jello, or a gray plastic junction box?"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # End timing for this image
    end_time_image = time.time()
    elapsed_time = end_time_image - start_time_image
    
    # Store results
    results[image_name] = {
        "time": elapsed_time,
        "response": output_text[0]
    }
    
    # Print results for this image
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Response: {output_text[0]}")
    print(f"Time for {image_name}: {int(minutes)} minutes and {seconds:.2f} seconds")

# End timing for entire process
end_time_total = time.time()
total_elapsed_time = end_time_total - start_time_total
total_minutes, total_seconds = divmod(total_elapsed_time, 60)

# Print summary
print("\n=== SUMMARY ===")
for image_name, data in results.items():
    mins, secs = divmod(data["time"], 60)
    print(f"{image_name}: {int(mins)} minutes and {secs:.2f} seconds")

print("\n=== TOTAL TIME ===")
print(f"Total processing time: {int(total_minutes)} minutes and {total_seconds:.2f} seconds")