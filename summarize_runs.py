import json
from pathlib import Path
from statistics import mean, stdev

folder = Path(__file__).resolve().parent / "corrected_results"
seeds = [42, 43, 44, 45, 46]

metrics = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "auc_roc": "AUC-ROC",
}

runs = []

for seed in seeds:
    path = folder / f"seed_{seed}" / "metrics.json"

    with path.open(encoding="utf-8") as file:
        result = json.load(file)

    result["seed"] = seed
    runs.append(result)

# Confirm that the runs used matching evaluation settings.
for field in ["dataset_sha256", "test_size", "threshold"]:
    if len({run[field] for run in runs}) != 1:
        raise ValueError(f"Runs have different values for {field}")

for run in runs:
    for key in metrics:
        if not 0 <= run[key] <= 1:
            raise ValueError(
                f"Invalid {key} value for seed {run['seed']}"
            )

print("\nFIVE-RUN TEST RESULTS")
print("Seeds:", seeds)
print("Test records:", runs[0]["test_size"])
print("Threshold:", runs[0]["threshold"])
print("Dataset fingerprints match.\n")

for key, label in metrics.items():
    values = [run[key] for run in runs]

    if key == "auc_roc":
        print(
            f"{label}: {mean(values):.4f} "
            f"± {stdev(values):.4f}"
        )
    else:
        percentages = [value * 100 for value in values]
        print(
            f"{label}: {mean(percentages):.2f}% "
            f"± {stdev(percentages):.2f} percentage points"
        )

print("\n± means sample standard deviation across five seeds.")
print("No training was started and no files were changed.")