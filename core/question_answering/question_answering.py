import random
import json
import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer


model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"

# # a) Get predictions
question_answering = pipeline(
    "question-answering", model=model_path, tokenizer=model_path
)


def que_context(context, que):

    # context = ""
    # question = ""
    answer = question_answering(que, context)

    # return f"Answer: {answer['answer']}\nScore: {answer['score']:.4f}\nStart: {answer['start']}\nEnd: {answer['end']}"
    return json.dumps(answer, indent=2)


def main():

    gr.close_all()

    interface = gr.Interface(
        fn=que_context,
        inputs=[
            gr.Textbox(label="Enter context:", lines=10),
            gr.Textbox(label="Enter question", lines=2),
        ],
        # outputs=gr.Textbox(label="Answer", lines=5),
        outputs=gr.JSON(label="Answer Details"),
        title="QnA Wizard",
        description="Answers questions based on provided context",
    )

    interface.launch()


if __name__ == "__main__":
    main()
