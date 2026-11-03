import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent


def threshold_for_cap(frame, cap):
    negatives = np.sort(
        frame.loc[frame.true_label == 0, "probability"]
        .to_numpy(dtype=np.float64)
    )[::-1]

    # Maximum number of false positives permitted.
    allowed = int(np.floor(cap * len(negatives)))

    # Predictions use probability >= threshold.
    # Move just above the boundary to handle tied scores safely.
    boundary = negatives[allowed]
    return float(np.nextafter(boundary, np.inf))


def evaluate(frame, threshold):
    y = frame.true_label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    prediction = (p >= threshold).astype(int)

    tn, fp, fn, tp = map(
        int, confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    )
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if tp + fp else None

    # Hypothetical prevalence adjustment; assumes these TPR/FPR
    # carry over to the deployment population.
    adjusted = {}
    for prevalence in (0.001, 0.01):
        denominator = (
            prevalence * recall + (1 - prevalence) * fpr
        )
        adjusted[str(prevalence)] = (
            prevalence * recall / denominator
            if denominator else None
        )

    return {
        "threshold": threshold,
        "recall": recall,
        "false_positive_rate": fpr,
        "precision": precision,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "hypothetical_precision_by_prevalence": adjusted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-folder",
        default=(
            "coverage_validation_comparison/"
            "20261020T182329_135060Z"
        ),
    )
    args = parser.parse_args()
    source = ROOT / args.comparison_folder

    frames = {}
    for name, expected_size in [
        ("validation_v3", 7668),
        ("validation_phiusiil", 4000),
    ]:
        path = source / f"broader_data_{name}_predictions.csv"
        frame = pd.read_csv(path, keep_default_na=False)

        if len(frame) != expected_size:
            raise ValueError(f"Unexpected row count: {name}")
        if not frame.true_label.isin([0, 1]).all():
            raise ValueError(f"Invalid labels: {name}")

        probabilities = frame.probability.to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError(f"Invalid probabilities: {name}")

        frames[name] = frame

    report = {
        "selection": (
            "One common threshold per target, chosen using both "
            "development validation sources; observed FPR capped "
            "separately on each source."
        ),
        "limitations": [
            "Threshold-selection results are development estimates.",
            "Validation FPR caps do not guarantee future FPR.",
            "Prevalence-adjusted precision is hypothetical.",
            "No test or reserved-holdout evaluation performed.",
        ],
        "operating_points": {},
    }
    rows = []

    for target in (None, 0.01, 0.005):
        name = "default_0.5" if target is None else f"fpr_cap_{target}"
        threshold = (
            0.5 if target is None
            else max(
                threshold_for_cap(frame, target)
                for frame in frames.values()
            )
        )

        report["operating_points"][name] = {
            "target_fpr": target,
            "threshold": threshold,
            "sources": {},
        }

        for source_name, frame in frames.items():
            result = evaluate(frame, threshold)

            if (
                target is not None
                and result["false_positive_rate"] > target + 1e-12
            ):
                raise ValueError("Selected threshold exceeds FPR cap")

            report["operating_points"][name]["sources"][source_name] = result
            rows.append({
                "operating_point": name,
                "source": source_name,
                "threshold": threshold,
                "recall_percent": 100 * result["recall"],
                "fpr_percent": 100 * result["false_positive_rate"],
                "false_positives": result["fp"],
                "true_positives": result["tp"],
            })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "coverage_threshold_analysis" / stamp
    output.mkdir(parents=True, exist_ok=False)

    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    table = pd.DataFrame(rows)
    table.to_csv(output / "threshold_comparison.csv", index=False)

    print("\nDEVELOPMENT VALIDATION THRESHOLDS")
    print(table.to_string(index=False, float_format="%.6f"))
    print("\nSaved to:", output)
    print("No training, inference or reserved-holdout scoring.")


if __name__ == "__main__":
    main()
    