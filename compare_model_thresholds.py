import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from check_coverage_thresholds import threshold_for_cap, evaluate

ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT / "coverage_validation_comparison"
    / "20261020T182329_135060Z"
)

MODELS = ["original", "root_only", "broader_data"]
SOURCES = {
    "validation_v3": 7668,
    "validation_phiusiil": 4000,
}


def main():
    rows = []
    report = {
        "policy": (
            "For each model and target, select one shared threshold "
            "that meets the observed FPR cap separately on both "
            "development validation sources."
        ),
        "limitations": [
            "Thresholds are selected and measured on development data.",
            "Observed validation FPR caps do not guarantee future FPR.",
            "Single-seed comparison.",
            "No test or reserved-holdout scoring.",
        ],
        "results": {},
        "prediction_sha256": {},
    }

    import hashlib

    for model_name in MODELS:
        frames = {}

        for source_name, expected_rows in SOURCES.items():
            path = SOURCE / f"{model_name}_{source_name}_predictions.csv"
            report["prediction_sha256"][path.name] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
            frame = pd.read_csv(path, keep_default_na=False)

            if len(frame) != expected_rows:
                raise ValueError(f"Unexpected row count: {path.name}")
            if not frame.true_label.isin([0, 1]).all():
                raise ValueError(f"Invalid labels: {path.name}")

            p = frame.probability.to_numpy(dtype=float)
            if (
                not np.isfinite(p).all()
                or ((p < 0) | (p > 1)).any()
            ):
                raise ValueError(f"Invalid probabilities: {path.name}")

            # Confirm every model is evaluated on identical rows.
            reference = pd.read_csv(
                SOURCE / f"original_{source_name}_predictions.csv",
                keep_default_na=False,
            )
            if not frame[["url", "true_label"]].equals(
                reference[["url", "true_label"]]
            ):
                raise ValueError(f"Row alignment failed: {path.name}")

            frames[source_name] = frame

        report["results"][model_name] = {}

        for cap in (0.01, 0.005):
            threshold = max(
                threshold_for_cap(frame, cap)
                for frame in frames.values()
            )
            target_name = f"fpr_cap_{cap}"
            report["results"][model_name][target_name] = {}

            for source_name, frame in frames.items():
                result = evaluate(frame, threshold)

                if result["false_positive_rate"] > cap + 1e-12:
                    raise ValueError("Selected threshold exceeds cap")

                report["results"][model_name][target_name][source_name] = result
                rows.append({
                    "model": model_name,
                    "target_fpr_percent": cap * 100,
                    "source": source_name,
                    "threshold": threshold,
                    "recall_percent": result["recall"] * 100,
                    "actual_fpr_percent": (
                        result["false_positive_rate"] * 100
                    ),
                    "false_positives": result["fp"],
                    "true_positives": result["tp"],
                })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "model_threshold_comparison" / stamp
    output.mkdir(parents=True, exist_ok=False)

    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON AT SHARED VALIDATION FPR TARGETS")
    print(table.to_string(index=False, float_format="%.6f"))
    print("\nSaved to:", output)
    print("No training, inference or reserved-holdout scoring.")


if __name__ == "__main__":
    main()