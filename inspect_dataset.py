import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

# Find the CSV beside this Python script.
folder = Path(__file__).resolve().parent
dataset = folder / "phishguard_gh_dataset_augmented.csv"

if not dataset.exists():
    raise SystemExit(
        "CSV not found. Put this script in the same folder as "
        "phishguard_gh_dataset_augmented.csv"
    )

# Read the dataset without changing it.
with dataset.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    if not {"url", "label"}.issubset(reader.fieldnames or []):
        raise SystemExit("The CSV must contain url and label columns.")

    rows = list(reader)

# Reproduce the ORIGINAL project's duplicate-identification rule.
# This is for checking the earlier audit, not our final cleaning rule.
def original_key(url):
    return re.sub(
        r"^https?://", "", url.strip()
    ).lower().strip()

labels_by_url = defaultdict(set)

for row in rows:
    key = original_key(row["url"])
    labels_by_url[key].add(row["label"].strip())

conflicting_keys = {
    key
    for key, labels in labels_by_url.items()
    if len(labels) > 1
}

counts = Counter(row["label"].strip() for row in rows)

print("\nDATASET CHECK")
print("Total rows:", len(rows))
print("Benign rows (0):", counts["0"])
print("Malicious rows (1):", counts["1"])
print("Unique canonical URLs:", len(labels_by_url))
print("Canonical URLs with conflicting labels:", len(conflicting_keys))
print("\nYour original CSV has NOT been changed.")