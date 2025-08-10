# import gradio as gr
# from transformers import pipeline
#
# # DEBUG: Verify model path is correct and readable
# model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
#
# try:
#     qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)
#     print("✅ Model loaded successfully.")
# except Exception as e:
#     print(f"❌ Failed to load model: {e}")
#     raise e
#
#
# def answer_question(context, question):
#     print("📝 Question:", question)
#     print("📄 Context snippet:", context[:200], "..." if len(context) > 200 else "")
#
#     try:
#         answer = qa_pipeline(question=question, context=context)
#         print("✅ Model output:", answer)
#         return f"Answer: {answer['answer']}\nScore: {answer['score']:.4f}\nStart: {answer['start']}, End: {answer['end']}"
#     except Exception as e:
#         print(f"❌ Error during inference: {e}")
#         return f"Error: {e}"
#
#
# gr.Interface(
#     fn=answer_question,
#     inputs=[
#         gr.Textbox(label="Context", lines=10, placeholder="Paste context here..."),
#         gr.Textbox(label="Question", lines=2, placeholder="What do you want to know?"),
#     ],
#     outputs=gr.Textbox(label="Answer", lines=4),
#     allow_flagging="never",
#     title="🧠 Question Answering Debugger",
#     description="Provide a context and a question. The model will try to find the answer.",
# ).launch(server_port=7860)
#
# #
# # boy age 20
# # lives chicago


import gradio as gr
from transformers import pipeline


model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)


def answer_question(context, question):
    return qa_pipeline(question=question, context=context)["answer"]


# iface = gr.Interface(
#     fn=answer_question, inputs=["text", "text"], outputs="text", allow_flagging="never"
# )
#
# # ✅ Enable API mode
# iface.launch(server_port=7860, share=False, show_api=True)
# # iface.launch(server_port=7860, show_api=True)


gr.Interface(
    fn=answer_question,
    inputs=["text", "text"],
    outputs="text",
    allow_flagging="never",
    api_name="predict",  # 🔥 This allows POST at /api/predict
).launch(
    server_port=7860, show_api=True  # ← ✅ REQUIRED to expose /api/predict/
)
