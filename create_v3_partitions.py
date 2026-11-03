import csv
import hashlib
from pathlib import Path

project = Path(__file__).resolve().parent
source = project / "dataset_fix_outputs_v2"
output = project / "dataset_fix_outputs_v3"

CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}


def encoded_key(url):
    text = url.strip()

    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    text = text.lower().strip()[:200]
    tokens = [CHAR_TO_INT.get(ch, 96) for ch in text]
    tokens += [0] * (200 - len(tokens))
    return hashlib.sha256(bytes(tokens)).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


parts = {}
fields = {}

for name in ["train", "validation", "test"]:
    fields[name], parts[name] = read_csv(source / f"{name}.csv")

train_keys = {encoded_key(row["url"]) for row in parts["train"]}

removed = [
    row for row in parts["validation"]
    if encoded_key(row["url"]) in train_keys
]
kept = [
    row for row in parts["validation"]
    if encoded_key(row["url"]) not in train_keys
]

if len(removed) != 1:
    raise ValueError(f"Expected one validation removal; got {len(removed)}")

parts["validation"] = kept

expected = {"train": 35664, "validation": 7668, "test": 7653}
encodings = {}

for name in parts:
    assert len(parts[name]) == expected[name]
    encodings[name] = {
        encoded_key(row["url"]) for row in parts[name]
    }

assert not encodings["train"] & encodings["validation"]
assert not encodings["train"] & encodings["test"]
assert not encodings["validation"] & encodings["test"]

output.mkdir(exist_ok=False)

for name, rows in parts.items():
    with (output / f"{name}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields[name])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{name}: {len(rows):,} rows")

with (output / "validation_encoding_overlap_removed.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=fields["validation"])
    writer.writeheader()
    writer.writerows(removed)

print("\nAll cross-partition encoded-input overlap checks passed.")
print("Saved to:", output)
print("Version 2 files preserved.")