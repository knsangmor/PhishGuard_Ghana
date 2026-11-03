"""Compare three saved models on identical development validation sets."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import keras

import train_coverage_pilot as pilot

ROOT = Path(__file__).resolve().parent


def original_encode(url):
    text = str(url).strip()
    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    seq = [
        ord(c) - 31 if 32 <= ord(c) <= 126 else 96
        for c in text.lower().strip()[:200]
    ]
    return seq + [0] * (200 - len(seq))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    folder = ROOT / "coverage_development_v1"

    datasets = {
        "validation_v3": folder / "validation_v3.csv",
        "validation_phiusiil": (
            folder / "validation_phiusiil_development.csv"
        ),
    }

    models = {
        "original": (
            ROOT / "corrected_results_v3" / "seed_42"
            / "phishguard_gh_model.keras",
            original_encode,
        ),
        "root_only": (
            ROOT / "root_normalized_pilot_v3" / "seed_42"
            / "model.keras",
            pilot.encode,
        ),
        "broader_data": (
            ROOT / "coverage_pilot_v1" / "seed_42"
            / "model.keras",
            pilot.encode,
        ),
    }

    frames = {}
    for name, path in datasets.items():
        expected = pilot.FILES[path.name]
        if pilot.sha(path) != expected:
            raise ValueError(f"Validation fingerprint mismatch: {name}")

        frame = pd.read_csv(path, keep_default_na=False)
        if not frame["label"].isin([0, 1]).all():
            raise ValueError(f"Invalid labels: {name}")
        frames[name] = frame
        print(f"{name}: {len(frame):,} verified rows")

    for name, (path, _) in models.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} model missing: {path}")
        print(f"{name}: model found")

    print("CHECK PASSED: identical validation inputs available.")
    if args.check_only:
        print("No inference, training or output folders created.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "coverage_validation_comparison" / stamp
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "seed": 42,
        "threshold": 0.5,
        "purpose": "Development comparison; no model training",
        "reserved_holdout_scored": False,
        "internal_test_scored": False,
        "validation_sha256": {
            name: pilot.sha(path) for name, path in datasets.items()
        },
        "model_sha256": {},
        "results": {},
        "limitations": [
            "Validation data have already informed development.",
            "PhiUSIIL is development data, not independent test evidence.",
            "Models differ in preprocessing and training data.",
            "Single-seed comparison; no statistical superiority claim.",
        ],
    }
    table = []

    for model_name, (model_path, encoder) in models.items():
        keras.backend.clear_session()
        print(f"\nLoading {model_name}...", flush=True)
        model = keras.models.load_model(model_path, compile=False)
        report["model_sha256"][model_name] = pilot.sha(model_path)
        report["results"][model_name] = {}

        for dataset_name, frame in frames.items():
            print(
                f"Predicting {model_name}: {dataset_name}...",
                flush=True,
            )
            x = np.asarray(
                [encoder(url) for url in frame["url"]],
                dtype=np.float32,
            )
            y = frame["label"].to_numpy(dtype=np.int32)
            p = model.predict(x, batch_size=256, verbose=1).reshape(-1)

            if (
                len(p) != len(y)
                or not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()
            ):
                raise ValueError("Invalid model probabilities")

            scores = pilot.metrics(y, p)
            report["results"][model_name][dataset_name] = scores
            table.append({
                "model": model_name,
                "validation_set": dataset_name,
                **scores,
            })

            pd.DataFrame({
                "url": frame["url"],
                "true_label": y,
                "probability": p,
                "pred_label": (p >= 0.5).astype(int),
            }).to_csv(
                output / f"{model_name}_{dataset_name}_predictions.csv",
                index=False,
            )
            del x

        del model

    pilot.write_json(output / "report.json", report)
    summary = pd.DataFrame(table)
    summary.to_csv(output / "comparison.csv", index=False)

    columns = [
        "model", "validation_set", "accuracy", "recall",
        "f1", "auc_roc", "false_positive_rate",
    ]
    print("\nVALIDATION COMPARISON")
    print(summary[columns].to_string(index=False, float_format="%.4f"))
    print("\nSaved to:", output)
    print("No training, internal-test or reserved-holdout scoring.")


if __name__ == "__main__":
    main()