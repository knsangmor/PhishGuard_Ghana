import csv
import re
from pathlib import Path

folder = Path(__file__).resolve().parent
splits = folder / "dataset_fix_outputs"


# Match the rule used by your cleaning script.
def canonical_key(url):
    return re.sub(r"^https?://", "", url.strip()).lower().strip()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


augmentation = read_rows(folder / "modern_legit_augmentation.csv")
augmented = read_rows(folder / "phishguard_gh_dataset_augmented.csv")

generated_keys = {
    canonical_key(row["url"]) for row in augmentation
}
input_keys = {
    canonical_key(row["url"]) for row in augmented
}

print("\nAUGMENTATION PROVENANCE CHECK")
print("Saved generated rows:", len(augmentation))
print("Unique generated keys:", len(generated_keys))
print(
    "Generated rows labelled benign:",
    sum(row["label"].strip() == "0" for row in augmentation),
)
print(
    "Generated keys absent from augmented input:",
    len(generated_keys - input_keys),
)

retained_keys = set()

for name in ["train", "validation", "test"]:
    rows = read_rows(splits / f"{name}.csv")
    matched = [
        row for row in rows
        if canonical_key(row["url"]) in generated_keys
    ]

    retained_keys.update(canonical_key(row["url"]) for row in matched)

    print(f"\n{name.upper()}")
    print("Total rows:", len(rows))
    print("Rows matching generated URLs:", len(matched))
    print("Other rows:", len(rows) - len(matched))
    print(
        "Generated matches with malicious labels:",
        sum(row["label"].strip() == "1" for row in matched),
    )

for filename in [
    "conflicting_records.csv",
    "removed_duplicates.csv",
    "invalid_hostname_records.csv",
]:
    rows = read_rows(splits / filename)
    matches = sum(
        canonical_key(row["url"]) in generated_keys
        for row in rows
    )
    print(f"\n{filename}: {matches} matching rows")

print("\nTOTAL")
print("Generated keys retained in partitions:", len(retained_keys))
print(
    "Generated keys absent from final partitions:",
    len(generated_keys - retained_keys),
)
print("\nNo files were changed.")

print("\nORIGINAL-SOURCE OVERLAP CHECK")

original_rows = read_rows(folder / "malicious_phish.csv")

original_keys = {
    canonical_key(row["url"])
    for row in original_rows
    if row.get("url", "").strip()
}

overlap = generated_keys & original_keys

print("Original source rows:", len(original_rows))
print("Generated keys also in original source:", len(overlap))

for key in sorted(overlap)[:10]:
    print("Example overlap:", key)

print("\nREMAINING-RECORD SOURCE CHECK")

for name in ["train", "validation", "test"]:
    rows = read_rows(splits / f"{name}.csv")

    other_rows = [
        row for row in rows
        if canonical_key(row["url"]) not in generated_keys
    ]

    unmatched = [
        row for row in other_rows
        if canonical_key(row["url"]) not in original_keys
    ]

    print(
        f"{name}: {len(other_rows):,} non-generated rows; "
        f"{len(unmatched):,} absent from original source"
    )

    for row in unmatched[:3]:
        print("Example unmatched URL:", repr(row["url"]))

from collections import defaultdict, Counter

print("\nSOURCE LABEL CHECK")

malicious_types = {"phishing", "malicious", "defacement", "malware", "1"}
benign_types = {"benign", "0"}

source_labels = defaultdict(set)
source_types = defaultdict(set)
unknown_types = Counter()

for row in original_rows:
    category = row["type"].strip().lower()
    url_key = canonical_key(row["url"])

    if category in malicious_types:
        label = 1
    elif category in benign_types:
        label = 0
    else:
        unknown_types[category] += 1
        continue

    source_labels[url_key].add(label)
    source_types[url_key].add(category)

print("Unrecognised source categories:", dict(unknown_types))

for name in ["train", "validation", "test"]:
    mismatches = 0
    ambiguous = 0
    categories = Counter()

    for row in read_rows(splits / f"{name}.csv"):
        url_key = canonical_key(row["url"])

        if url_key in generated_keys:
            continue

        expected_labels = source_labels.get(url_key, set())

        if len(expected_labels) > 1:
            ambiguous += 1
        elif expected_labels != {int(row["label"])}:
            mismatches += 1

        types = source_types.get(url_key, set())
        category = next(iter(types)) if len(types) == 1 else "multiple/unknown"
        categories[category] += 1

    print(f"\n{name.upper()}")
    print("Label mismatches or missing mappings:", mismatches)
    print("Keys with conflicting source labels:", ambiguous)
    print("Source categories:", dict(categories))

print("\nEXAMPLES OF SOURCE LABEL CONFLICTS")

source_examples = defaultdict(list)

for row in original_rows:
    url_key = canonical_key(row["url"])

    if len(source_labels.get(url_key, set())) > 1:
        example = (row["url"], row["type"])
        if example not in source_examples[url_key]:
            source_examples[url_key].append(example)

shown = set()

for name in ["train", "validation", "test"]:
    for row in read_rows(splits / f"{name}.csv"):
        url_key = canonical_key(row["url"])

        if (
            len(source_labels.get(url_key, set())) > 1
            and url_key not in shown
        ):
            shown.add(url_key)
            print(f"\nPartition: {name}")
            print("Retained URL:", repr(row["url"]))
            print("Retained label:", row["label"])
            print("Source versions:")

            for url, category in source_examples[url_key][:6]:
                print(" ", repr(url), "->", category)

            if len(shown) == 10:
                break

    if len(shown) == 10:
        break

print("\nMODEL-ENCODING CONFLICT CHECK")

def model_text(url):
    text = str(url).strip()

    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    return text.lower().strip()[:200]


for name in ["train", "validation", "test"]:
    ambiguous = 0
    encoding_conflicts = 0

    for row in read_rows(splits / f"{name}.csv"):
        url_key = canonical_key(row["url"])

        if len(source_labels.get(url_key, set())) <= 1:
            continue

        ambiguous += 1
        labels_by_text = defaultdict(set)

        for source_url, category in source_examples[url_key]:
            if category in malicious_types:
                label = 1
            elif category in benign_types:
                label = 0
            else:
                continue

            labels_by_text[model_text(source_url)].add(label)

        if any(len(labels) > 1 for labels in labels_by_text.values()):
            encoding_conflicts += 1

    print(
        f"{name}: {ambiguous} source-ambiguous URLs; "
        f"{encoding_conflicts} with opposite labels "
        "for the same normalized first 200 characters"
    )

import hashlib

print("\nFULL-SOURCE ENCODED-INPUT AUDIT")

CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}


def encoded_key(url):
    text = model_text(url)
    sequence = [CHAR_TO_INT.get(ch, 96) for ch in text]
    sequence += [0] * (200 - len(sequence))
    return hashlib.sha256(bytes(sequence)).digest()


labels_by_encoding = defaultdict(set)

for row in original_rows:
    category = row["type"].strip().lower()

    if category in malicious_types:
        label = 1
    elif category in benign_types:
        label = 0
    else:
        raise ValueError(f"Unknown source category: {category!r}")

    labels_by_encoding[encoded_key(row["url"])].add(label)

# Include the saved synthetic additions in the audit.
for row in augmentation:
    labels_by_encoding[encoded_key(row["url"])].add(
        int(row["label"])
    )

ambiguous_encodings = {
    code for code, labels in labels_by_encoding.items()
    if len(labels) > 1
}

total_affected = 0

for name in ["train", "validation", "test"]:
    rows = read_rows(splits / f"{name}.csv")

    affected = sum(
        encoded_key(row["url"]) in ambiguous_encodings
        for row in rows
    )
    total_affected += affected

    print(f"{name}: {affected:,} records with ambiguous encoded inputs")

print("Total affected partition records:", total_affected)
print("No files were changed.")

print("\nSAVING SOURCE-AUDITED PARTITIONS")

revised_folder = folder / "dataset_fix_outputs_v2"
revised_folder.mkdir(exist_ok=False)

expected_removed = {
    "train": 480,
    "validation": 68,
    "test": 83,
}

for name in ["train", "validation", "test"]:
    source_path = splits / f"{name}.csv"

    with source_path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)

    kept = []
    quarantined = []

    for row in rows:
        if encoded_key(row["url"]) in ambiguous_encodings:
            quarantined.append(row)
        else:
            kept.append(row)

    assert len(quarantined) == expected_removed[name]
    assert len(kept) + len(quarantined) == len(rows)

    for filename, records in [
        (f"{name}.csv", kept),
        (f"{name}_quarantined.csv", quarantined),
    ]:
        with (revised_folder / filename).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    counts = Counter(row["label"].strip() for row in kept)

    print(f"\n{name.upper()}")
    print("Quarantined:", len(quarantined))
    print("Retained:", len(kept))
    print("Benign:", counts["0"])
    print("Malicious:", counts["1"])

print("\nSaved to:", revised_folder)
print("Original partitions and previous results preserved.")