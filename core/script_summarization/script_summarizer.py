import yt_dlp
import whisper
import os
import gradio as gr

import torch
from PIL import Image
from transformers import pipeline, AutoTokenizer


model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff/"
summarizer_pipeline = pipeline("summarization", model=model_path)


def download_audio_from_youtube(url: str, output_dir: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",  # This is the correct key
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        video_id = result["id"]
        audio_path = os.path.join(output_dir, f"{video_id}.mp3")

    return audio_path


# Function to transcribe audio using Whisper
def transcribe_audio(audio_path: str):
    model = whisper.load_model(
        "base"
    )  # You can also use 'small', 'medium', 'large' for better accuracy but more resources
    result = model.transcribe(audio_path)
    return result["text"]


def transcriber(youtube_url):

    # youtube_url = "https://youtu.be/4qCJQJa8IWg?si=y2ch1qX7sgHv6r1d"
    output_dir = "downloads"  # You can change this to any directory

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading audio...")
    audio_path = download_audio_from_youtube(youtube_url, output_dir)
    print(f"Audio downloaded: {audio_path}")
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    print("Transcription completed:")
    print(transcript)
    # Save transcript to a text file
    transcript_file = os.path.join(output_dir, "transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(transcript)
    print(f"Transcript saved to {transcript_file}")
    # Clean up the downloaded audio file
    os.remove(audio_path)

    return transcript_file


def text_to_chunks(text, tokenizer, max_tokens=900):  # keep it below 1024 to be safe
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0

    for word in words:
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        if current_len + len(word_tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(word_tokens)
        else:
            current_chunk.append(word)
            current_len += len(word_tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_by_chunks(text, summarize_fn, max_tokens=900):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    chunks = text_to_chunks(text, tokenizer, max_tokens)
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            print(
                f"Summarizing chunk {i+1}/{len(chunks)} (approx {len(tokenizer.encode(chunk))} tokens)"
            )
            summary = summarize_fn(chunk)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
    return " ".join(summaries)


def script_summarizer(youtube_url):
    transcript_file = transcriber(youtube_url)

    with open(transcript_file, "r") as f:
        transcript_text = f.read()

    return summarize_by_chunks(transcript_text, summarizer_pipeline, 900)


def main():

    gr.close_all()
    # summary = summarize(transcript_text)

    demo = gr.Interface(
        fn=script_summarizer,
        inputs=[gr.Textbox(label="Enter URL to summarize", lines=1)],
        outputs=[
            gr.Textbox(
                label="Summary",
                lines=10,
            )
        ],
        title="YT Script Summarizer",
        description="This app summarises a given YT video from URL.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
