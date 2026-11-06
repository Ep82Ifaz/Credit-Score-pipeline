"""Flask API for the Twitter sentiment model - test it with Postman.

Run:   python app.py          (needs best_model.pkl + tfidf_vectorizer.pkl in this same folder)
Env:   HOST (default 0.0.0.0)  PORT (default 5000)
"""
import os
import pickle
import re
import sys
from pathlib import Path

import nltk
import numpy as np
from flask import Flask, jsonify, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
MAX_BATCH = 500


def ensure_nltk():
    """Make sure the NLTK data used by the notebook's clean_text() is available."""
    for resource, path in (("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource '{resource}' ...")
            nltk.download(resource, quiet=True)


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(
            f"'{path.name}' not found in {path.parent}. Run the notebook's save-model cell first "
            f"and keep the .pkl files in the same folder as app.py.")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not load '{path.name}': {e}. Use the same scikit-learn version "
                           f"that created it.") from e


ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))

try:
    model = load_pickle(MODEL_PATH)
    vectorizer = load_pickle(VECTORIZER_PATH)
except Exception as err:
    print(f"[STARTUP ERROR] {err}", file=sys.stderr)
    sys.exit(1)


def clean_text(text):
    """Same cleaning as the notebook (lowercase, drop <unk>, URLs, symbols, stopwords)."""
    text = str(text).lower()
    text = text.replace("<unk>", "")
    text = re.sub(r"http\S+|www\S+|pic\.twitter\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = word_tokenize(text)
    return " ".join(w for w in words if w not in STOP_WORDS)


def predict_texts(texts):
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    labels = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs, kind = model.predict_proba(X), "probability"
    else:  # LinearSVC has no predict_proba -> softmax over decision scores
        d = model.decision_function(X)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        probs, kind = e / e.sum(axis=1, keepdims=True), "softmax of SVM decision scores (not a calibrated probability)"
    results = []
    for text, clean, label, p in zip(texts, cleaned, labels, probs):
        item = {
            "text": text,
            "cleaned_text": clean,
            "sentiment": str(label),
            "confidence": round(float(p.max()), 4),
            "scores": {str(c): round(float(v), 4) for c, v in zip(model.classes_, p)},
        }
        if not clean.strip():
            item["warning"] = "Nothing left after cleaning (only URLs/symbols/stopwords) - prediction is unreliable."
        results.append(item)
    return results, kind


app = Flask(__name__)


def error(message, status=400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return jsonify({
        "service": "Twitter Sentiment API (Flask)",
        "endpoints": {
            "GET /health": "service + model status",
            "POST /predict": 'body: {"text": "your tweet"}',
            "POST /predict/batch": 'body: {"texts": ["tweet 1", "tweet 2"]}',
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__,
                    "classes": [str(c) for c in model.classes_],
                    "vocabulary_size": len(vectorizer.vocabulary_)})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form.to_dict() or request.args.to_dict()      # form-data / x-www-form-urlencoded / ?text=
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return error('Send a non-empty "text" field, e.g. {"text": "I love this game"} '
                     '(Postman: Body -> raw -> JSON).')
    results, kind = predict_texts([text])
    return jsonify({**results[0], "confidence_type": kind})


@app.post("/predict/batch")
def predict_batch():
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        texts = data.get("texts")
    else:
        texts = request.form.getlist("texts") or None                # form-data: repeat the key "texts"
    if not isinstance(texts, list) or not texts:
        return error('Send a non-empty "texts" list, e.g. {"texts": ["I love it", "worst update ever"]}.')
    if len(texts) > MAX_BATCH:
        return error(f"Too many texts ({len(texts)}). Maximum per request is {MAX_BATCH}.", 413)
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return error('Every item in "texts" must be a non-empty string.')
    results, kind = predict_texts(texts)
    return jsonify({"count": len(results), "confidence_type": kind, "predictions": results})


@app.errorhandler(404)
def not_found(_):
    return error("Route not found. Try GET /, GET /health, POST /predict or POST /predict/batch.", 404)


@app.errorhandler(405)
def bad_method(_):
    return error("Method not allowed for this route (prediction routes need POST).", 405)


@app.errorhandler(500)
def server_error(_):
    return error("Internal server error - check the terminal / flask_server.log for details.", 500)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    print(f"Model: {type(model).__name__} | classes: {list(model.classes_)}")
    print(f"Open in Postman: http://127.0.0.1:{port}/health")
    app.run(host=host, port=port, debug=False)
