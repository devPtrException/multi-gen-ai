import os
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import scipy.io.wavfile as wavfile


model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
narrator_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--kakao-enterprise--vits-ljs/snapshots/3bcb8321394f671bd948ebf0d086d694dda95464"


object_detector = pipeline("object-detection", model=model_path)
narrator = pipeline("text-to-speech", model=narrator_path)
#
#
# def draw_bound_boxes(img, detections, font_path=None, font_size=20):
#     draw_image = img.copy()
#     draw = ImageDraw.Draw(draw_image)
#
#     # Load font
#     if font_path:
#         try:
#             font = ImageFont.truetype(font_path, font_size)
#         except IOError:
#             print(f"Warning: Could not load font from {font_path}. Using default font.")
#             font = ImageFont.load_default()
#     else:
#         font = ImageFont.load_default()
#
#     for (
#         detection
#     ) in detections:  # 'detection' is a single dictionary from the list 'detections'
#         box = detection["box"]
#         xmin = box["xmin"]
#         ymin = box["ymin"]
#         xmax = box["xmax"]
#         ymax = box["ymax"]
#
#         label = detection["label"]
#         score = detection["score"]
#         text = f"{label} {score:.2f}"
#
#         # Draw the bounding box (e.g., in red, with a width)
#         draw.rectangle(
#             [(xmin, ymin), (xmax, ymax)], outline="red", width=3
#         )  # Added outline color and width
#
#         # --- CORRECTED PART FOR TEXT DIMENSIONS AND BACKGROUND ---
#         # Calculate the bounding box for the text
#         # textbbox returns (left, top, right, bottom) coordinates of the text bounding box
#         # We need to provide an anchor point for the text. Using (0, 0) as anchor for calculation.
#         # This gives us the width and height of the text as if drawn from (0,0)
#         # The first two elements (left, top) will usually be 0 or small negative values,
#         # and (right, bottom) will give you the width and height.
#         bbox_coords = draw.textbbox((0, 0), text, font=font)
#         text_width = bbox_coords[2] - bbox_coords[0]  # right - left
#         text_height = bbox_coords[3] - bbox_coords[1]  # bottom - top
#
#         # Position for text background - typically slightly above the top-left corner of the bbox
#         text_bg_xmin = xmin
#         text_bg_ymin = ymin - text_height - 5  # 5 pixels padding above the box
#         if text_bg_ymin < 0:  # Ensure it doesn't go off-image at the top
#             text_bg_ymin = (
#                 ymin + 5
#             )  # If it goes off, put it inside the box (or adjust as desired)
#
#         text_bg_xmax = text_bg_xmin + text_width + 5  # Add some padding to width
#         text_bg_ymax = text_bg_ymin + text_height + 5  # Add some padding to height
#
#         # Draw a filled rectangle for the text background (e.g., black)
#         draw.rectangle(
#             [(text_bg_xmin, text_bg_ymin), (text_bg_xmax, text_bg_ymax)], fill="black"
#         )
#
#         # Draw the text on top of the background (e.g., in white)
#         # Position the text within its background. The anchor for `draw.text` should be
#         # the top-left corner of where the text itself should start.
#         text_x = text_bg_xmin + (
#             bbox_coords[0] * -1
#         )  # Adjust for potential negative left offset from textbbox
#         text_y = text_bg_ymin + (
#             bbox_coords[1] * -1
#         )  # Adjust for potential negative top offset from textbbox
#
#         # A simpler approach for text positioning relative to the background box:
#         # text_x = text_bg_xmin + 2 # Small padding from left
#         # text_y = text_bg_ymin + 2 # Small padding from top
#         # Use the simpler (text_bg_xmin + 2, text_bg_ymin + 2) if bbox_coords[0/1] are consistently near 0 or you don't need pixel-perfect alignment based on font metrics.
#         # The current version with bbox_coords[0/1] * -1 is more accurate to the font's true rendering.
#
#         draw.text((text_x, text_y), text, fill="white", font=font)
#         # --- END OF CORRECTED PART ---
#
#     return draw_image
#


def draw_bound_boxes(img, detections, font_path=None, font_size=20, threshold=0.8):
    draw_image = img.copy()
    draw = ImageDraw.Draw(draw_image)

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Could not load font from {font_path}. Using default font.")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    for detection in detections:
        score = detection["score"]
        if score < threshold:
            continue  # Skip if below threshold

        box = detection["box"]
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        label = detection["label"]
        text = f"{label} {score:.2f}"

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

        bbox_coords = draw.textbbox((0, 0), text, font=font)
        text_width = bbox_coords[2] - bbox_coords[0]
        text_height = bbox_coords[3] - bbox_coords[1]

        text_bg_xmin = xmin
        text_bg_ymin = ymin - text_height - 5
        if text_bg_ymin < 0:
            text_bg_ymin = ymin + 5

        text_bg_xmax = text_bg_xmin + text_width + 5
        text_bg_ymax = text_bg_ymin + text_height + 5

        draw.rectangle(
            [(text_bg_xmin, text_bg_ymin), (text_bg_xmax, text_bg_ymax)], fill="black"
        )

        text_x = text_bg_xmin + (-bbox_coords[0])
        text_y = text_bg_ymin + (-bbox_coords[1])
        draw.text((text_x, text_y), text, fill="white", font=font)

    return draw_image


def read_objects(detection_objects):
    object_counts = {}
    for detection in detection_objects:
        label = detection["label"]
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    # Prepare parts for a natural sentence
    parts = []
    for label, count in object_counts.items():
        # Handle singular/plural
        item_text = f"{count} {label}"
        if count > 1:
            item_text += "s"
        parts.append(item_text)

    # Construct the final response based on the number of unique object types
    if not parts:
        response = "This picture contains no discernible objects."
    elif len(parts) == 1:
        response = f"This picture contains {parts[0]}."
    elif len(parts) == 2:
        response = f"This picture contains {parts[0]} and {parts[1]}."
    else:
        # For three or more items, use comma separation and 'and' before the last
        response = (
            "This picture contains " + ", ".join(parts[:-1]) + f", and {parts[-1]}."
        )

    return response


def generate_audio(text):
    if not text:
        print(
            "Warning: Received empty text for audio generation. Returning None for audio."
        )
        return None  # Or path to a silent audio file, or "" to signify no audio

    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)
    output_audio_file = os.path.join(output_dir, "output.wav")

    narrated_text = narrator(text)  # Ensure text is passed as a list

    # Check if narrated_text has 'audio' data before writing
    if narrated_text and "audio" in narrated_text and len(narrated_text["audio"]) > 0:
        wavfile.write(
            filename=output_audio_file,
            rate=narrated_text["sampling_rate"],
            data=narrated_text["audio"][0],
        )
        return output_audio_file
    else:
        print("Warning: Narrator returned no audio data. Returning None for audio.")
        return None

    return output_audio_file


# def detect_objects(img):
#     raw_img = img
#     output = object_detector(img)
#     processed_image = draw_bound_boxes(raw_img, output)
#     # processed_image = img
#
#     natural_text = read_objects(output)
#     processed_audio = generate_audio(natural_text)
#
#     # print(type(output))
#     # print(natural_text)
#
#     # if processed_audio is None:
#     #     print("No audio file generated.")
#
#     return processed_image, natural_text


def detect_objects(img):
    raw_img = img
    raw_output = object_detector(img)

    # Filter detections above 80%
    filtered_output = [d for d in raw_output if d["score"] >= 0.8]

    processed_image = draw_bound_boxes(raw_img, filtered_output)
    natural_text = read_objects(filtered_output)
    processed_audio = generate_audio(natural_text)
    print(natural_text)

    return processed_image, natural_text


def main():

    gr.close_all()

    interface = gr.Interface(
        fn=detect_objects,
        inputs=gr.Image("Select an Image", type="pil"),
        outputs=[
            gr.Image("Objects", type="pil", label="Objects"),
            gr.Text(
                "You will see objects detected here.",
                label="Detection",
            ),
        ],
        title="Object Detector",
        description="Identifies object in a provided image/url",
    )

    interface.launch()


# generate_audio("hi there")


if __name__ == "__main__":
    main()
