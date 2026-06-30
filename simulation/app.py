"""
PhishGuard-GH — Inference API
------------------------------
Loads the trained CNN-LSTM model (phishguard_gh_model.keras / .h5) and serves
real predictions to the forensic evidence simulator.

Nothing here is re-derived or approximated: the character encoding
(url_to_seq) and the rule-based rationale (build_rationale) are copied
verbatim from model_hash.py in the PhishGuard_Ghana repository, so a URL
processed through this API is encoded exactly the way it was during training
and full-dataset inference.

This file does NOT train or modify the model in any way. It only loads the
artefact you already produced with model_hash.py.
"""

import os
import re
from urllib.parse import urlparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Constants — copied from model_hash.py. Do not change unless your training
# configuration changes, or the model will see differently-shaped input.
# ---------------------------------------------------------------------------
MAX_LEN     = 200
VOCAB_SIZE  = 97
THRESHOLD   = 0.50
CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}

MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS = os.path.join(MODEL_DIR, "phishguard_gh_model.keras")
MODEL_H5    = os.path.join(MODEL_DIR, "phishguard_gh_model.h5")

# Rule-based secondary indicators — copied verbatim from model_hash.py
_BRANDS     = ["mtn", "vodafone", "airtel", "ecobank", "gcb", "absa", "bog"]
_FREE_TLDS  = [".ml", ".tk", ".cf", ".ga", ".gq", ".xyz", ".top"]
_SHORTENERS = ["bit.ly", "tinyurl.com", "ow.ly", "rb.gy", "is.gd"]

VERSION = "PhishGuard-GH vCNN-LSTM"
LEGAL   = ("Act 772 Ss.7-11 [Electronic Records]; "
           "NRCD 323 S.135 [Evidence]; Act 1038 S.4(c)")

# ---------------------------------------------------------------------------
# Load the trained model — fails loudly (and immediately, at startup) if the
# weight file isn't found. This is intentional: the server should never fall
# back to a substitute classifier.
# ---------------------------------------------------------------------------
import keras


def _load_trained_model():
    if os.path.exists(MODEL_KERAS):
        path = MODEL_KERAS
    elif os.path.exists(MODEL_H5):
        path = MODEL_H5
    else:
        raise FileNotFoundError(
            "No trained model found. Place phishguard_gh_model.keras or "
            "phishguard_gh_model.h5 (the artefact produced by model_hash.py) "
            f"in this folder: {MODEL_DIR}"
        )
    print(f"[PhishGuard-GH] Loading trained model from: {path}")
    m = keras.models.load_model(path, compile=False)
    print(f"[PhishGuard-GH] Model loaded. Input shape: {m.input_shape}")
    return m


model = _load_trained_model()


# ---------------------------------------------------------------------------
# Preprocessing — copied verbatim from model_hash.py:url_to_seq()
# ---------------------------------------------------------------------------
def url_to_seq(url: str) -> list:
    url = str(url).lower().strip()
    seq = [CHAR_TO_INT.get(c, VOCAB_SIZE - 1) for c in url[:MAX_LEN]]
    return seq + [0] * (MAX_LEN - len(seq))


# ---------------------------------------------------------------------------
# Rationale — copied verbatim from model_hash.py:build_rationale()
# ---------------------------------------------------------------------------
def build_rationale(url: str, verdict: str, prob: float) -> str:
    url_l = url.lower()
    indicators = []
    try:
        hostname = urlparse(url).netloc or ""
    except Exception:
        hostname = ""
    for tld in _FREE_TLDS:
        if hostname.endswith(tld):
            indicators.append(f"Free TLD ({tld})")
            break
    if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", hostname):
        indicators.append("IP address hostname")
    for s in _SHORTENERS:
        if s in url_l:
            indicators.append(f"Shortener ({s})")
            break
    for b in _BRANDS:
        if b in url_l and any(hostname.endswith(t) for t in _FREE_TLDS):
            indicators.append(f"Brand '{b}' on suspicious domain")
            break
    if "@" in url:
        indicators.append("@ symbol in URL")
    if not url_l.startswith("https"):
        indicators.append("Non-HTTPS")
    ind = "; ".join(indicators[:3]) if indicators else "No rule-based indicators"
    return (f"[PRIMARY] CNN-LSTM: {prob*100:.2f}% {verdict}. "
            f"[SECONDARY] {ind}.")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # allow the simulator HTML (file:// or Live Server origin) to call this


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    seq = url_to_seq(url)
    x = np.array([seq], dtype=np.float32)  # shape (1, MAX_LEN) — matches training input

    prob = float(model.predict(x, verbose=0)[0][0])
    verdict = "PHISHING" if prob >= THRESHOLD else "LEGITIMATE"
    rationale = build_rationale(url, verdict, prob)

    return jsonify({
        "verdict": verdict,
        "confidence": round(prob, 6),
        "rationale": rationale,
        "model": VERSION,
        "legal": LEGAL
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)