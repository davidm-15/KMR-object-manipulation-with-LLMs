import requests
from PIL import Image
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoModel, AutoConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForZeroShotObjectDetection, LlavaForConditionalGeneration
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import OmDetTurboForObjectDetection
from PIL import Image, ImageDraw
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

def main():
    input_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/MustardBottle/inputs"
    output_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/MustardBottle/outputs"
    os.makedirs(output_dir, exist_ok=True)

    torch.cuda.empty_cache()

    texts = [["a yellow mustard bottle"]]
    # texts = "a yellow mustard bottle"
    test_models = ["IDEA"]

    for model in test_models:
        used_model = model
        if used_model == "OWLVIT2":
            processor, model = load_OWLVIT2()
            inference_function = inference_OWLVIT
        elif used_model == "OWLVIT":
            processor, model = load_OWLVIT()

            inference_function = inference_OWLVIT
        elif used_model == "glamm":
            processor, model = load_glamm()

            inference_function = inference_glamm
        
        elif used_model == "omdet":
            processor, model = load_omdet()

            inference_function = inference_omdet

        elif used_model == "OWLVIT2Xenova":
            processor, model = load_OWLVIT2_Xenova()
            inference_function = inference_OWLVIT

        elif used_model == "IDEA":
            processor, model = load_IDEA()
            inference_function = inference_IDEA


        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_dir, filename)
                image = Image.open(image_path)


                boxes, scores, labels = inference_function(processor, model, image, texts)

                print(boxes, scores, labels)

                draw = ImageDraw.Draw(image)
                for box in boxes:
                    draw.rectangle(tuple(box), outline="red", width=3)

                output_image_path = os.path.join(output_dir, f"{used_model}_{filename}")
                image.save(output_image_path)
                print(f"Image saved with bounding boxes at {output_image_path}")
                # Free up memory after processing each image
                torch.cuda.empty_cache()



def load_IDEA():
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return processor, model

def load_OWLVIT2_Xenova():
    model_id = "Xenova/owlv2-base-patch16"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

    return processor, model
    
def load_omdet():
    model_id = "omlab/omdet-turbo-swin-tiny-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(device)

    return processor, model

def inference_IDEA(processor, model, image, texts):
    inputs = processor(images=image, text=texts, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]
    return boxes, scores, labels

def inference_omdet(processor, model, image, texts):
    inputs = processor(images=image, text=texts, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        text_labels=texts,
        target_sizes=[image.size[::-1]],
        threshold=0.3,
        nms_threshold=0.3,
    )[0]


    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]

    return boxes, scores, labels

def load_OWLVIT():
    model_id = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

    return processor, model

def load_OWLVIT2():
    model_id = "google/owlv2-large-patch14-ensemble"
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)

    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Calculate model size in bytes (assuming float32 precision)
    model_size_bytes = num_params * 4  # Each float32 parameter takes 4 bytes
    model_size_mb = model_size_bytes / (1024 ** 2)  # Convert to MB
    model_size_gb = model_size_mb / 1024  # Convert to GB

    print(f"Number of parameters: {num_params}")
    print(f"Model size: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")

    return processor, model


def load_glamm():
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return processor, model

def inference_glamm(processor, model, image, texts):
    inputs = processor(images=image, text=texts, return_tensors="pt").to(device)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]
    return boxes, scores, labels


def inference_OWLVIT(processor, model, image, texts):
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]

    return boxes, scores, labels




def QWEN():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    ).to(device)

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-72B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/communication/images/received_image.jpg",
                },
                {"type": "text", "text": "Describe this image."},
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
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)



def glamm():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    input_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/MustardBottle/inputs"
    output_dir = "/mnt/proj3/open-29-7/mira_ws/Projects/Diplomka/KMR-object-manipulation-with-LLMs/ImageProcessing/MustardBottle/outputs"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)
            text = "yellow mustard bottle."

            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

            print(f"Detected {len(results[0]['boxes'])} objects in the image {filename}.")

            for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
                box = [round(i, 2) for i in box.tolist()]
                print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for box in results[0]["boxes"]:
                draw.rectangle(tuple(box), outline="red", width=3)

            # Save the image with bounding boxes
            output_image_path = os.path.join(output_dir, filename)
            image.save(output_image_path)
            print(f"Image saved with bounding boxes at {output_image_path}")




if __name__ == "__main__":
    # main()

    QWEN()
