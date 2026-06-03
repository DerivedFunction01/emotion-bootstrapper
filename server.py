# server_roberta.py
# Production inference server for SINGLE-LABEL RoBERTa (domain-adapted or fine-tuned)

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import multiprocessing as mp
import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
app = Flask(__name__)

# ==================== CONFIG ====================
# Change this to your model path (local or Hugging Face Hub)
MODEL_PATH = (
    "DerivedFunction/derivative-category-classifier"  # ← your trained model folder
)
# OR use HF Hub: "your-username/roberta-finance-classifier"

# Optional: override device via env var (useful for CPU-only deployments)
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu").lower()
device = torch.device(
    "cpu" if DEVICE_TYPE == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
)


# ==================== DYNAMIC BATCH SIZE ====================
def get_dynamic_batch_size():
    """Determines a safe batch size based on available GPU VRAM."""
    if not torch.cuda.is_available():
        return 32  # A safe default for CPU

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if vram_gb >= 20:  # e.g., A100, RTX 4090/3090
        return 256
    elif vram_gb >= 14:  # e.g., V100, T4
        return 128
    elif vram_gb >= 8:  # e.g., RTX 3070, 2080
        return 64
    return 32  # For GPUs with < 8GB VRAM


# ==================== LOAD MODEL & LABELS ====================
print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Load label mapping (saved during training)
label_mapping_path = Path(MODEL_PATH) / "label_mapping.json"
if label_mapping_path.exists():
    with open(label_mapping_path) as f:
        mapping = json.load(f)
    id2label = mapping.get(
        "id2label", {int(k): v for k, v in mapping.get("id2label", {}).items()}
    )
    label2id = mapping.get("label2id", {v: int(k) for k, v in id2label.items()})
else:
    # Fallback: use model.config
    id2label = model.config.id2label
    label2id = model.config.label2id

labels = [id2label[i] for i in range(len(id2label))]
print(f"Loaded {len(labels)} labels: {labels}")

model.to(device)
model.eval()

MAX_BATCH_SIZE = get_dynamic_batch_size()
print(f"Using device: {device}. Max batch size set to: {MAX_BATCH_SIZE}")


# ==================== PREDICTION FUNCTION ====================
def predict_batch(texts):

    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # print(f"Received {len(texts)} texts for prediction")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        results = []
        for probs in probabilities:
            probs = probs.cpu().numpy()
            pred = {id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}
            results.append(pred)
    except Exception as e:
        log.error(f"Error during prediction: {e}")
        return {"predictions": [{"error": str(e)} for _ in texts]}

    return {"predictions": results}


# ==================== ROUTES ====================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return (
            jsonify({"error": "Request must contain 'texts' as a list of strings"}),
            400,
        )

    texts = data["texts"]
    if not all(isinstance(t, str) for t in texts):
        return jsonify({"error": "All items in 'texts' must be strings"}), 400

    # Process the texts in batches to avoid GPU OOM errors
    all_predictions = []
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch_texts = texts[i : i + MAX_BATCH_SIZE]
        try:
            batch_result = predict_batch(batch_texts)
            all_predictions.extend(batch_result.get("predictions", []))
        except Exception as e:
            log.error(f"Error processing batch: {e}")
            # If a batch fails, return an error for the whole request
            return jsonify({"error": f"Error processing batch: {e}"}), 500

    return jsonify({"predictions": all_predictions})


@app.route("/info", methods=["GET"])
def info():
    info = {
        "model": MODEL_PATH,
        "task": "single-label-classification",
        "labels": labels,
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "max_batch_size": MAX_BATCH_SIZE,
    }
    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                ),
            }
        )
    else:
        info["cpu_cores"] = mp.cpu_count()

    return jsonify(info)


@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {"status": "ok", "message": "RoBERTa single-label server is running"}
    )


# ==================== RUN ====================
if __name__ == "__main__":
    print(f"Server starting on {device} with {len(labels)} classes")
    app.run(host="0.0.0.0", port=5000, threaded=True)
# GPU (recommended)
# DEVICE_TYPE=gpu gunicorn --workers 1 --threads 8 --timeout 120 roberta_server:app --bind 0.0.0.0:5001

# CPU-only fallback
# DEVICE_TYPE=cpu gunicorn --workers 1 --threads 12 --timeout 120 roberta_server:app --bind 0.0.0.0:5002
