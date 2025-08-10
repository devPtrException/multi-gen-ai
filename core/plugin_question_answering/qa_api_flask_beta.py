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
        data = request.get_json()
        context = data["data"][0]
        question = data["data"][1]

        # Logging (optional)
        print(f"📘 Context received (len={len(context)}): {context[:100]}...")
        print(f"❓ Question: {question}")

        context = preprocess_context(context)
        # Chunk long context into small sections
        chunks = chunk_context(context)

        answers = []
        for chunk in chunks:
            try:
                result = qa_pipeline(question=question, context=chunk)
                if result.get("answer"):
                    answers.append(result["answer"])
            except:
                continue

        # Return most frequent answer (basic voting)
        if answers:
            final_answer = max(set(answers), key=answers.count)
        else:
            # Fallback for brand-specific questions
            if "brand" in question.lower():
                # final_answer = fallback_brand_extraction(context) or "Brand not found"
                pass
            else:
                final_answer = "Sorry, I couldn't find an answer."

        return jsonify({"data": [final_answer]})

    except Exception as e:
        print("❌ Exception:", e)
        return jsonify({"error": str(e)}), 500


# ---------- Run Server ----------

if __name__ == "__main__":
    app.run(port=7860, debug=True)
