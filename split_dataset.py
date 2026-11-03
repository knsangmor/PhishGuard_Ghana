import csv
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

from inspect_dataset import original_key

folder = Path(__file__).resolve().parent
output_folder = folder / "dataset_fix_outputs"
source = output_folder / "dataset_cleaned.csv"

extractor = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)

def get_domain(url):
    text = url.strip()

    if text.startswith("//"):
        text = "https:" + text
    elif "://" not in text:
        text = "https://" + text

    host = urlsplit(text).hostname

    if not host:
        raise ValueError("No hostname found")

    host = host.lower().rstrip(".")

    # Keep IP addresses as their own groups.
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass

    result = extractor(host)
    return result.top_domain_under_public_suffix or host

with source.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    fields = reader.fieldnames
    rows = list(reader)

valid_rows = []
invalid_rows = []
groups = []

for row in rows:
    try:
        domain = get_domain(row["url"])
        valid_rows.append(row)
        groups.append(domain)
    except ValueError:
        invalid_rows.append(row)

def save_csv(filename, records):
    with (output_folder / filename).open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

save_csv("invalid_hostname_records.csv", invalid_rows)

labels = np.array([int(row["label"]) for row in valid_rows])
groups = np.array(groups)
inputs = np.zeros((len(valid_rows), 1))

# Make 20 folds while keeping domains together
# and attempting to balance the classes.
splitter = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=42,
)

folds = [
    held_out
    for _, held_out in splitter.split(inputs, labels, groups)
]

# Fix this allocation before training.
train_indices = np.concatenate(folds[:14])
validation_indices = np.concatenate(folds[14:17])
test_indices = np.concatenate(folds[17:])

partitions = {
    "train": train_indices,
    "validation": validation_indices,
    "test": test_indices,
}

# Check that every accepted record appears exactly once.
all_indices = np.concatenate(list(partitions.values()))
assert len(all_indices) == len(valid_rows)
assert len(set(all_indices.tolist())) == len(valid_rows)

# Check that domain groups and canonical URLs do not overlap.
names = list(partitions)

for i, first in enumerate(names):
    for second in names[i + 1:]:
        first_indices = partitions[first]
        second_indices = partitions[second]

        first_domains = set(groups[first_indices])
        second_domains = set(groups[second_indices])

        assert not first_domains & second_domains, (
            f"Domain overlap: {first} and {second}"
        )

        first_urls = {
            original_key(valid_rows[j]["url"])
            for j in first_indices
        }
        second_urls = {
            original_key(valid_rows[j]["url"])
            for j in second_indices
        }

        assert not first_urls & second_urls, (
            f"Canonical URL overlap: {first} and {second}"
        )

print("\nSPLIT RESULTS")
print("Cleaned input rows:", len(rows))
print("Rows excluded because hostname parsing failed:", len(invalid_rows))
print("Rows included in partitions:", len(valid_rows))

for name, indices in partitions.items():
    records = [valid_rows[j] for j in indices]
    save_csv(f"{name}.csv", records)

    malicious = sum(int(row["label"]) for row in records)
    benign = len(records) - malicious

    print(f"\n{name.upper()}")
    print("Rows:", len(records))
    print("Benign:", benign)
    print("Malicious:", malicious)
    print("Malicious percentage:", round(100 * malicious / len(records), 2))

print("\nCHECKS PASSED")
print("No overlapping domain groups.")
print("No overlapping canonical URLs.")
print("Every accepted record belongs to exactly one partition.")
print("Original dataset files have NOT been changed.")