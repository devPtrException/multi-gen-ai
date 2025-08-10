from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 🔧 Load model
model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)


def preprocess_context(raw_text):
    # Narrow down to the likely info block
    # if "About this item" in raw_text:
    #     raw_text = raw_text.split("About this item")[1]

    import re

    match = re.search(r"frequently", raw_text, flags=re.IGNORECASE)
    return raw_text[: match.start()].strip() if match else raw_text.strip()


def chunk_text(text, max_tokens=400):
    import textwrap

    return textwrap.wrap(text, max_tokens)


def chunk_context(context, max_chunk_length=400):
    import textwrap

    return textwrap.wrap(context, width=max_chunk_length)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        context, question = data["data"]

        result = qa_pipeline(question=question, context=context)

        start, end = result["start"], result["end"]
        answer = result["answer"]

        # Highlight the answer in the context
        highlighted = (
            context[:start]
            + "<mark style='background-color: yellow;'>"
            + context[start:end]
            + "</mark>"
            + context[end:]
        )

        return jsonify(
            {
                "data": [answer],
                "highlighted_context": highlighted,
                "score": result["score"],
                "start": start,
                "end": end,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Run Server ----------

if __name__ == "__main__":
    app.run(port=7860, debug=True)
