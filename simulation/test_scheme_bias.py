"""
PhishGuard-GH — Scheme-sensitivity probe
------------------------------------------
Run this against your CURRENT model to see, directly, how much the verdict
depends on the http:// vs https:// prefix alone. This does NOT fix anything
and does NOT require retraining — it's purely diagnostic, run before you
decide whether the scheme-stripping retrain (Step 2/3) is worth doing.

    cd backend
    python test_scheme_bias.py
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


def strip_scheme(url: str) -> str:
    u = url.strip()
    for prefix in ("https://", "http://"):
        if u.lower().startswith(prefix):
            return u[len(prefix):]
    return u


def load():
    path = MODEL_KERAS if os.path.exists(MODEL_KERAS) else MODEL_H5
    return keras.models.load_model(path, compile=False)


# Pick your own real-world legitimate URLs to test here — domains the model
# almost certainly never saw during training.
TEST_URLS = [
    "https://www.bbc.com/news",
    "https://www.who.int/health-topics",
    "https://www.gov.uk/government/policies/water-and-sanitation-in-developing-countries",
    "https://www.un.org/en/about-us",
    "https://www.worldbank.org/en/home",
]


def predict(model, url):
    seq = url_to_seq(url)
    x = np.array([seq], dtype=np.float32)
    return float(model.predict(x, verbose=0)[0][0])


def main():
    model = load()
    print(f"{'AS-IS (https kept)':>22}  {'SCHEME STRIPPED':>18}   URL")
    print("-" * 90)
    for url in TEST_URLS:
        p_with = predict(model, url)
        p_without = predict(model, strip_scheme(url))
        shift = "  <-- big shift" if abs(p_with - p_without) > 0.3 else ""
        print(f"{p_with:>20.4f}  {p_without:>18.4f}   {url}{shift}")

    print()
    print("If stripping the scheme swings these from PHISHING-leaning toward")
    print("LEGITIMATE-leaning, that confirms the model is keying heavily on the")
    print("scheme rather than the host/path content — proceed to the retrain fix.")


if __name__ == "__main__":
    main()