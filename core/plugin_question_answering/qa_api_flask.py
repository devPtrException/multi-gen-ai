from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 🔧 Load model
model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--deepset--roberta-base-squad2/snapshots/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        context, question = data["data"]
        result = qa_pipeline(question=question, context=context)
        return jsonify({"data": [result["answer"]]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ QnA Flask API is running. POST to /api/predict."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
