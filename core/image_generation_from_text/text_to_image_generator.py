import os
from dotenv import load_dotenv
import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login

# # Get absolute path to .env (one level up from this script)
# dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
# print("Looking for .env at:", dotenv_path)  # debug print

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
# if HF_TOKEN is None:
#     raise ValueError("HF_TOKEN not found. Please set it in core/.env")

# Authenticate once at the start
login(HF_TOKEN)
# login(token=HF_TOKEN)

# Initialize pipeline globally
device = "cuda" if torch.cuda.is_available() else "cpu"
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    # "stabilityai/stable-diffusion-2-1",
    # "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--stabilityai--stable-diffusion-2-1",
    "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=HF_TOKEN,
    local_files_only=True,
)
pipeline.enable_model_cpu_offload()


def image_generator(prompt):
    image = pipeline(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark, low resolution, blurry",
        num_inference_steps=77,
        height=1024,
        width=1024,
        guidance_scale=9.0,
    ).images[0]
    return image


def main():

    # Gradio interface
    interface = gr.Interface(
        fn=image_generator,
        inputs=gr.Textbox(
            label="Enter Prompt", placeholder="e.g., An octopus trying to cast a spell"
        ),
        outputs=gr.Image(label="Generated Image"),
        title="Image Generator from Text",
        description="Generate images from Text prompts.",
    )
    interface.launch()


if __name__ == "__main__":
    main()
