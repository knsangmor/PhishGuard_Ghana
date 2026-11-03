import csv
import hashlib
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import tldextract

folder = Path(__file__).resolve().parent / "dataset_fix_outputs_v2"

extractor = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)

CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}


def normalized(url):
    text = url.strip()
    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    return text.lower().strip()


def encoded_key(url):
    text = normalized(url)[:200]
    tokens = [CHAR_TO_INT.get(ch, 96) for ch in text]
    tokens += [0] * (200 - len(tokens))
    return hashlib.sha256(bytes(tokens)).hexdigest()


def domain_group(url):
    text = url.strip()
    if text.startswith("//"):
        text = "https:" + text
    elif "://" not in text:
        text = "https://" + text

    host = (urlsplit(text).hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError(f"No hostname: {url!r}")

    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        result = extractor(host)
        return result.top_domain_under_public_suffix or host


expected = {"train": 35664, "validation": 7669, "test": 7653}
partitions = {}

for name, size in expected.items():
    path = folder / f"{name}.csv"

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == size, f"Unexpected size: {name}"
    assert all(row["label"] in {"0", "1"} for row in rows)

    partitions[name] = {
        "domains": {domain_group(row["url"]) for row in rows},
        "urls": {
            re.sub(r"^https?://", "", row["url"].strip()).lower().strip()
            for row in rows
        },
        "encodings": {encoded_key(row["url"]) for row in rows},
    }

    print(
        f"{name}: {len(rows):,} rows; "
        f"{len(partitions[name]['domains']):,} domain groups"
    )
    print("SHA256:", hashlib.sha256(path.read_bytes()).hexdigest())

names = list(partitions)
problems = []

for i, first in enumerate(names):
    for second in names[i + 1:]:
        print(f"\n{first} versus {second}")

        for field in ["domains", "urls", "encodings"]:
            overlap = partitions[first][field] & partitions[second][field]
            print(f"Overlapping {field}: {len(overlap)}")

            if overlap:
                problems.append(f"{first}/{second}: {field}")

if problems:
    print("\nREVIEW NEEDED:", "; ".join(problems))
else:
    print("\nALL CROSS-PARTITION OVERLAP CHECKS PASSED")

print("No files changed and no training started.")

print("\nTRAIN–VALIDATION ENCODING OVERLAP DETAILS")

shared = (
    partitions["train"]["encodings"]
    & partitions["validation"]["encodings"]
)

for name in ["train", "validation"]:
    with (folder / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if encoded_key(row["url"]) in shared:
            print(f"\nPartition: {name}")
            print("URL:", repr(row["url"]))
            print("Label:", row["label"])
            print("Domain group:", domain_group(row["url"]))
            print("Normalized length:", len(normalized(row["url"])))
            print("Encoded text:", repr(normalized(row["url"])[:200]))