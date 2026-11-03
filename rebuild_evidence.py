import csv
import json
from pathlib import Path

import model_hash as pipeline

# Importing model_hash does not run its training section.
project = Path(__file__).resolve().parent
source = project / "corrected_results" / "seed_46"
output = source / "evidence_confidence_corrected"


def verify_saved_file(filename, fingerprint_filename):
    path = source / filename

    with (source / fingerprint_filename).open(
        encoding="utf-8"
    ) as file:
        fingerprint = json.load(file)

    actual = pipeline.sha256_file(str(path))

    if actual != fingerprint["sha256"]:
        raise RuntimeError(f"Fingerprint mismatch: {filename}")

    print(f"Verified: {filename}")
    return actual


# Verify the saved source files before rebuilding.
pred_hash = verify_saved_file(
    "predictions.csv", "predictions_hash.txt"
)
model_hash = verify_saved_file(
    "phishguard_gh_model.h5", "model_fingerprint.txt"
)

with (source / "dataset_hash.txt").open(encoding="utf-8") as file:
    dataset_record = json.load(file)

# Read and validate predictions before creating any chain.
with (source / "predictions.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

if len(rows) != 51617:
    raise ValueError(f"Expected 51,617 records; found {len(rows):,}")

for row in rows:
    probability = float(row["probability"])
    label = int(row["pred_label"])

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Invalid probability in predictions.csv")

    if label not in (0, 1):
        raise ValueError("Invalid predicted label")

# Stop if this output folder already exists.
output.mkdir(exist_ok=False)

# Redirect the chain's output files before initialising it.
pipeline.CHAIN_JSONL = str(
    output / "phishguard_evidence_chain.jsonl"
)
pipeline.HASH_REGISTRY_CSV = str(
    output / "phishguard_hash_registry.csv"
)
pipeline.VERIFY_TXT = str(
    output / "phishguard_chain_verification.txt"
)

chain = pipeline.PhishGuardHashChain()

for number, row in enumerate(rows, start=1):
    url = row["url"]
    probability = float(row["probability"])
    verdict = (
        "PHISHING" if int(row["pred_label"]) == 1
        else "LEGITIMATE"
    )

    confidence = (
        probability if verdict == "PHISHING"
        else 1.0 - probability
    )

    rationale = pipeline.build_rationale(
        url, verdict, probability
    )
    chain.log(url, verdict, confidence, rationale)

    if number % 1000 == 0:
        print(f"Logged {number:,}/{len(rows):,}")

verified = chain.save_report(
    dataset_record["sha256"], model_hash, pred_hash
)

if not verified:
    raise RuntimeError("Rebuilt evidence chain failed verification")

certificate = chain.forensic_cert(1)
(output / "certificate_record_1.txt").write_text(
    certificate, encoding="utf-8"
)

print("\n" + certificate)
print("\nREBUILD COMPLETE")
print("Records:", len(rows))
print("Saved to:", output)
print("Original training results and evidence files preserved.")