import os
import gradio as gr
from PIL import Image
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import scipy.io.wavfile as wavfile

model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--Salesforce--blip-image-captioning-large/snapshots/353689b859fcf0523410b1806dace5fb46ecdf41"
narrator_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--kakao-enterprise--vits-ljs/snapshots/3bcb8321394f671bd948ebf0d086d694dda95464"

image_captionizer_pipeline = pipeline("image-to-text", model=model_path)
narrator_pipeline = pipeline("text-to-speech", model=narrator_path)


def generate_audio(text):
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_audio_file = os.path.join(output_dir, "output.wav")

    narrated_text = narrator_pipeline(text)
    wavfile.write(
        filename=output_audio_file,
        rate=narrated_text["sampling_rate"],
        data=narrated_text["audio"][0],
    )
    # print(text)
    return output_audio_file, text


#     narrated_text = narrator(text)
#
#     wavfile.write(
#         "output.wav", narrated_text["sampling_rate"], narrated_text["audio"][0]
#     )
#     return "output.wav"


def caption_image(img):
    semantics = image_captionizer_pipeline(img)[0]["generated_text"]
    return generate_audio(semantics)


def main():

    gr.close_all()

    interface = gr.Interface(
        fn=caption_image,
        inputs=gr.Image("Select an Image", type="pil"),
        outputs=[
            gr.Audio("Image Caption", label=""),
            gr.Text("Image Caption", label="Image Caption"),
        ],
        title="Image Captionizer with audio",
        description="Provides a textual and audio caption based on provided image",
    )

    interface.launch()
    # generate_audio("hello dogs cats")


if __name__ == "__main__":
    main()
