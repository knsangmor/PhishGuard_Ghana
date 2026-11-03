import csv
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

folder = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def key(url):
    return re.sub(r"^https?://", "", url.strip()).lower().strip()


generated_keys = {
    key(row["url"])
    for row in read_rows(folder / "modern_legit_augmentation.csv")
}

test_rows = read_rows(folder / "dataset_fix_outputs" / "test.csv")
labels = np.array([int(row["label"]) for row in test_rows])

generated = np.array([
    key(row["url"]) in generated_keys for row in test_rows
])
other = ~generated

assert generated.sum() == 279
assert np.all(labels[generated] == 0)

print("\nTEST PERFORMANCE BY PROVENANCE")
print("Generated benign URLs:", int(generated.sum()))
print("Other test URLs:", int(other.sum()))

for seed in [42, 43, 44, 45, 46]:
    run = folder / "corrected_results" / f"seed_{seed}"

    saved_labels = np.load(run / "y_test.npy").reshape(-1)
    probabilities = np.load(run / "y_prob.npy").reshape(-1)

    if not np.array_equal(saved_labels, labels):
        raise ValueError(f"Test label alignment failed for seed {seed}")

    # Check exact test URL order against saved full-dataset predictions.
    prediction_rows = read_rows(run / "predictions.csv")
    saved_test_rows = prediction_rows[-len(test_rows):]

    if [row["url"] for row in saved_test_rows] != [
        row["url"] for row in test_rows
    ]:
        raise ValueError(f"Test URL alignment failed for seed {seed}")

    if len(probabilities) != len(labels):
        raise ValueError(f"Probability count mismatch for seed {seed}")

    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities for seed {seed}")

    predictions = (probabilities >= 0.5).astype(int)

    false_positives = int(predictions[generated].sum())
    generated_fpr = false_positives / int(generated.sum())

    original_benign = other & (labels == 0)
    original_fp = int(predictions[original_benign].sum())
    original_fpr = original_fp / int(original_benign.sum())

    print(f"\nSEED {seed}")
    print(
        f"Generated benign false positives: "
        f"{false_positives}/279"
    )
    print(f"Generated benign FPR: {generated_fpr * 100:.2f}%")
    print(
        f"Generated benign specificity: "
        f"{(1 - generated_fpr) * 100:.2f}%"
    )
    print(f"Other benign FPR: {original_fpr * 100:.2f}%")
    print(
        "Other test accuracy:",
        f"{accuracy_score(labels[other], predictions[other]) * 100:.2f}%",
    )
    print(
        "Other test F1:",
        f"{f1_score(labels[other], predictions[other], zero_division=0) * 100:.2f}%",
    )
    print(
        "Other test AUC:",
        f"{roc_auc_score(labels[other], probabilities[other]):.4f}",
    )

print("\nNo training started and no files changed.")