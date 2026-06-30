"""
PhishGuard-GH — Diagnostic script
----------------------------------
Run this directly (no Flask, no browser) to isolate where the "predicts
everything as phishing" bug is coming from:

    cd backend
    python debug_predict.py

It loads the model exactly the way app.py does, prints the model summary
(so you can confirm it's the CNN-LSTM hybrid and not a baseline file by
mistake), then runs known ground-truth URLs pulled straight from your real
predictions.csv — URLs your actual trained model scored near 0% and near
100% phishing during the experiment your thesis reports.

If these come back wrong here too, the bug is in model loading or
preprocessing (this script), not in Flask or the browser.
If they come back correct here but wrong through the API, the bug is in
app.py or the request path.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import keras

MAX_LEN     = 200
VOCAB_SIZE  = 97
CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}

MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS = os.path.join(MODEL_DIR, "phishguard_gh_model.keras")
MODEL_H5    = os.path.join(MODEL_DIR, "phishguard_gh_model.h5")


def url_to_seq(url: str) -> list:
    url = str(url).lower().strip()
    seq = [CHAR_TO_INT.get(c, VOCAB_SIZE - 1) for c in url[:MAX_LEN]]
    return seq + [0] * (MAX_LEN - len(seq))


def load():
    path = MODEL_KERAS if os.path.exists(MODEL_KERAS) else MODEL_H5
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {MODEL_KERAS} or {MODEL_H5}")
    print(f"Loading: {path}\n")
    return keras.models.load_model(path, compile=False)


# Ground truth pulled directly from predictions.csv — these are URLs your
# REAL model already scored during the actual experiment behind your thesis.
CONFIDENT_LEGITIMATE = [
    ("youtube.com/watch?v=yTKAB7X6Q3g", 0.000815),
    ("hatyaiok.com/blog/wp-content/uploads/2012/03/aol.com.htm", 0.001071),
    ("nyfa.edu/acting-school/acting-for-film/12evening.php", 0.000625),
    ("ccbusa.com/", 0.000404),
    ("capitolmuseum.ca.gov/", 0.000254),
]

CONFIDENT_PHISHING = [
    ("http://www.meridianranch.com/index.php?option=com_content&view=article&id=121:creekview-grill&catid=7:recreation&Itemid=190", 0.999998),
    ("http://www.zuzatravel.pl/polska-centralna/item/43-warszawa-2-dni.html", 0.999998),
    ("www.gamereport.com/tgr7/freighttrain.html", 0.997110),
    ("https://safirbetgiristikla.blogspot.com/", 0.999991),
]


def main():
    model = load()

    print("=" * 70)
    print("MODEL SUMMARY (confirm this is the CNN-LSTM hybrid, not a baseline)")
    print("=" * 70)
    model.summary()
    print()

    print("Input shape expected by model:", model.input_shape)
    print()

    print("=" * 70)
    print("SHOULD ALL BE NEAR 0.0  (originally scored near-0% phishing)")
    print("=" * 70)
    for url, original_prob in CONFIDENT_LEGITIMATE:
        seq = url_to_seq(url)
        x = np.array([seq], dtype=np.float32)
        new_prob = float(model.predict(x, verbose=0)[0][0])
        flag = "  <-- MISMATCH" if new_prob > 0.5 else ""
        print(f"  original={original_prob:.6f}  now={new_prob:.6f}  {url[:60]}{flag}")

    print()
    print("=" * 70)
    print("SHOULD ALL BE NEAR 1.0  (originally scored near-100% phishing)")
    print("=" * 70)
    for url, original_prob in CONFIDENT_PHISHING:
        seq = url_to_seq(url)
        x = np.array([seq], dtype=np.float32)
        new_prob = float(model.predict(x, verbose=0)[0][0])
        flag = "  <-- MISMATCH" if new_prob < 0.5 else ""
        print(f"  original={original_prob:.6f}  now={new_prob:.6f}  {url[:60]}{flag}")

    print()
    print("=" * 70)
    print("RAW OUTPUT STATS across all 9 test URLs (sanity check)")
    print("=" * 70)
    all_urls = [u for u, _ in CONFIDENT_LEGITIMATE + CONFIDENT_PHISHING]
    seqs = np.array([url_to_seq(u) for u in all_urls], dtype=np.float32)
    probs = model.predict(seqs, verbose=0).flatten()
    print("  min:", probs.min(), " max:", probs.max(), " std:", probs.std())
    if probs.std() < 0.01:
        print("  --> All outputs nearly identical regardless of input.")
        print("      This points to: weights not actually loaded (random/")
        print("      default init), wrong file loaded, or a custom layer")
        print("      mismatch that Keras silently skipped during load.")


if __name__ == "__main__":
    main()