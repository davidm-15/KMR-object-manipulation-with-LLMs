from transformers import AutoProcessor, Llama4ForConditionalGeneration # Or LlavaForConditionalGeneration if appropriate
import torch
import sys # Import sys to check for image handling capabilities if needed

# --- Configuration ---
model_id = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/Llama_weights"
# Choose one of the alternative attention implementations:
# attn_choice = "sdpa"   # Preferred alternative
attn_choice = "eager"  # Fallback if sdpa also causes issues or isn't available

print(f"Attempting to load with attn_implementation='{attn_choice}'...")
# -------------------

try:
    # !! Crucially, check if your model is actually Llama or a multimodal variant like Llava !!
    # If it handles images as in your example, it's likely Llava or similar.
    # Use the correct class name. Let's assume Llava for this example based on image input.
    # If it's truly a text-only Llama, change back to LlamaForConditionalGeneration.
    from transformers import LlavaForConditionalGeneration # Example: Use Llava class

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained( # Use correct model class
        model_id,
        attn_implementation=attn_choice, # Use the chosen implementation
        device_map="auto",             # Keep auto mapping for now
        torch_dtype=torch.bfloat16,
    )
    print(f"Successfully loaded with attn_implementation='{attn_choice}'.")

except ImportError as e:
     print(f"ImportError: {e}. Make sure you have the correct model class installed/imported.")
     sys.exit(1)
except ValueError as e:
    print(f"ValueError during loading with '{attn_choice}': {e}")
    if attn_choice == "sdpa":
         print("Trying 'eager' attention implementation as fallback...")
         try:
              attn_choice = "eager"
              model = LlavaForConditionalGeneration.from_pretrained( # Use correct model class
                model_id,
                attn_implementation=attn_choice,
                device_map="auto",
                torch_dtype=torch.bfloat16,
              )
              print(f"Successfully loaded with attn_implementation='{attn_choice}'.")
         except Exception as e2:
              print(f"Failed to load with 'eager' as well: {e2}")
              sys.exit(1)
    else:
         print("Failed even with 'eager'. Check model compatibility and environment.")
         sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    sys.exit(1)


# --- Rest of your code ---
url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
        ]
    },
]

try:
    print("Processing inputs...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device) # processor usually handles device placement correctly with model.device

    print("Generating response...")
    # It's good practice to disable gradients during inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    print("Decoding response...")
    # Decode only the newly generated tokens
    response_start_index = inputs["input_ids"].shape[-1]
    response = processor.batch_decode(outputs[:, response_start_index:], skip_special_tokens=True)[0]

    print("\nResponse:")
    print(response)
    # print("\nRaw output tensor (first sequence):") # Optional: print raw output if needed
    # print(outputs[0])

except Exception as e:
    print(f"\nAn error occurred during processing or generation: {e}")
    import traceback
    traceback.print_exc()