from PIL import Image
import torch
from dotenv import load_dotenv
import os
import gradio as gr
from transformers import pipeline
from huggingface_hub import login
from diffusers import StableDiffusionPipeline


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")
# if HF_TOKEN is None:
#     raise ValueError("HF_TOKEN not found. Please set it in core/.env")


login(token=HF_TOKEN)
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-2-1",
    # "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--stabilityai--stable-diffusion-2-1",
    "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=HF_TOKEN,
    local_files_only=True,
)
sd_pipeline.enable_model_cpu_offload()

model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--Salesforce--blip-image-captioning-large/snapshots/353689b859fcf0523410b1806dace5fb46ecdf41"
image_captionizer_pipeline = pipeline("image-to-text", model=model_path)


def caption_image(img):
    semantics = image_captionizer_pipeline(img)[0]["generated_text"]
    return semantics


def image_generator(img):
    prompt_semantics = caption_image(img)

    image = sd_pipeline(
        prompt=prompt_semantics,
        negative_prompt="blurred, ugly, watermark, low resolution, blurry",
        num_inference_steps=77,
        height=1024,
        width=1024,
        guidance_scale=9.0,
    ).images[0]
    return image


def main():
    gr.close_all()

    gr.Interface(
        fn=image_generator,
        inputs=gr.Image(label="Upload an image to generate more...", type="pil"),
        outputs=gr.Image(label="Generated Image will be shown here..."),
        title="Image Generator from Image",
        description="Generates a new Image based on a provided image",
    ).launch()


if __name__ == "__main__":
    main()
