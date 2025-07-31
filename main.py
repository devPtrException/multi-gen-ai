import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

from core.summarization.summarizer import summarizer
from core.script_summarization.script_summarizer import script_summarizer
from core.image_captioning.image_captionizer import caption_image
from core.image_generation.image_generator import image_generator


st.set_page_config(
    page_title="MultiPurpose AI",
    page_icon="🦹🏻‍♀️",
    layout="centered",
)

with st.sidebar:
    side_menu_selector = option_menu(
        "AI Menu",
        [
            "Summarize Text",
            "",
            "Summarize YT Video",
            "Generate Image",
        ],
        default_index=0,
        menu_icon=":brain:",
        # icons=["robot", "ball", "heart"],
        icons=["chat-dots-fill", "image-fill", "textarea-t", "image-fill"],
    )

#
# def main():
#     print("Hello from multi!")
#
#
# if __name__ == "__main__":
#     main()


if side_menu_selector == "Summarize Text":
    st.title("Summarizer")

    user_text_corpus = st.text_area("Enter text to summarize...")

    if st.button("Summarize"):
        if user_text_corpus:
            output = summarizer(user_text_corpus)
            st.text_area(output)
        else:
            st.info("Please enter a text and try again")


if side_menu_selector == "Summarize YT Video":
    st.title("Summarize YT Video")

    user_url = st.text_area("Enter a YouTube URL here to summarize...")

    if st.button("Summarize Video"):
        if user_url:
            output = script_summarizer(user_url)
            st.markdown(output)
        else:
            st.info("Please upload an image and try again")

#
if side_menu_selector == "Caption Image":
    st.title("Caption Image")

    user_img = st.file_uploader("Enter an image here to generate caption...")

    if st.button("Generate Caption"):
        if user_img:
            output_audio_path, output_text = caption_image(user_img)

            st.markdown(output_text)
            st.audio(output_audio_path, format="audio/wav")

        else:
            st.info("Please upload and image and try again")


if side_menu_selector == "Generate Image":
    st.title("Generate Image:")

    user_prompt = st.text_input("Enter a prompt here to generate image...")

    if st.button("Generate Image"):
        if user_prompt:
            output = image_generator(user_prompt)
            st.image(output)
        else:
            st.info("Pleas  enter a prompt and try again")
