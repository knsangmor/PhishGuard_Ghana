import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

folder = Path(__file__).resolve().parent
source = folder / "phishguard_gh_dataset_augmented.csv"
output_folder = folder / "dataset_fix_outputs"
output_folder.mkdir(exist_ok=True)

# Read the original file.
with source.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    fields = reader.fieldnames

    if not {"url", "label"}.issubset(fields or []):
        raise SystemExit("The CSV must contain url and label columns.")

    rows = list(reader)

# Reproduce the existing project's canonical-key rule.
# The original URL text will remain unchanged in the output.
def canonical_key(url):
    return re.sub(
        r"^https?://", "", url.strip()
    ).lower().strip()

labels_by_key = defaultdict(set)

for row in rows:
    label = row["label"].strip()

    if label not in {"0", "1"}:
        raise SystemExit(f"Unexpected label: {label!r}")

    row["label"] = label
    labels_by_key[canonical_key(row["url"])].add(label)

# Find keys that have both benign and malicious labels.
conflicting_keys = {
    key
    for key, labels in labels_by_key.items()
    if len(labels) > 1
}

conflicts = []
duplicates = []
cleaned = []
seen = set()

# Sort first so duplicate selection does not depend on input row order.
ordered_rows = sorted(
    rows,
    key=lambda row: (
        canonical_key(row["url"]),
        row["url"],
        row["label"],
    ),
)

for row in ordered_rows:
    key = canonical_key(row["url"])

    if key in conflicting_keys:
        conflicts.append(row)
    elif key in seen:
        duplicates.append(row)
    else:
        seen.add(key)
        cleaned.append(row)

def save_csv(filename, records):
    path = output_folder / filename

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    return path

cleaned_path = save_csv("dataset_cleaned.csv", cleaned)
save_csv("conflicting_records.csv", conflicts)
save_csv("removed_duplicates.csv", duplicates)

# Check that every original row is accounted for.
assert len(rows) == len(cleaned) + len(conflicts) + len(duplicates)
assert len(cleaned) == len({canonical_key(r["url"]) for r in cleaned})

counts = Counter(row["label"] for row in cleaned)

print("\nCLEANING COMPLETE")
print("Original rows:", len(rows))
print("Conflicting URL groups:", len(conflicting_keys))
print("Rows quarantined for conflicting labels:", len(conflicts))
print("Repeated rows removed:", len(duplicates))
print("Cleaned rows:", len(cleaned))
print("Cleaned benign rows:", counts["0"])
print("Cleaned malicious rows:", counts["1"])
print("\nSaved cleaned dataset to:", cleaned_path)
print("Your original CSV has NOT been changed.")