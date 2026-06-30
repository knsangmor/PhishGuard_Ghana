"""
PhishGuard-GH — Phishing recall recovery via counterfactual scheme augmentation
---------------------------------------------------------------------------------
WHY THIS EXISTS
After scheme-stripping was added to url_to_seq(), the model stopped using the
http(s):// prefix as a phishing shortcut. That fixed the false positives on
modern legitimate HTTPS sites, but it cost some recall: phishing URLs whose
main distinguishing signal was the scheme started slipping through.

This script recovers that recall WITHOUT reintroducing the bias. It takes a
sample of phishing URLs and creates counterfactual duplicates with the scheme
flipped (bare <-> https://), keeping the SAME phishing label. The point:

  * The model sees HTTPS attached to phishing too, so "https = legitimate"
    can't form either.
  * Because url_to_seq() strips the scheme anyway during training, these
    duplicates force the model to rely on host/path CONTENT to separate
    phishing from legitimate, instead of leaning on the prefix.

It deliberately does NOT add legitimate rows — that would tilt the class
balance further toward legit, which is what reduced recall in the first place.

USAGE
    python recover_phishing_recall.py \
        --dataset phishguard_gh_dataset.csv \
        --n_augment 4000 \
        --output phishguard_gh_dataset_recall.csv

Then (same as before): swap the output into phishguard_gh_dataset.csv,
regenerate features, retrain, and re-test.
"""

import argparse
import csv
import random
import re


SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def has_scheme(url: str) -> bool:
    return bool(SCHEME_RE.match(url.strip()))


def strip_scheme(url: str) -> str:
    u = url.strip()
    for prefix in ("https://", "http://"):
        if u.lower().startswith(prefix):
            return u[len(prefix):]
    return u


def normalize_for_dedup(url: str) -> str:
    u = strip_scheme(url).lower()
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Current training dataset (url,label)")
    ap.add_argument("--n_augment", type=int, default=4000,
                    help="How many counterfactual phishing rows to add")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="phishguard_gh_dataset_recall.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = []
    with open(args.dataset, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if "url" not in fieldnames or "label" not in fieldnames:
        raise SystemExit(f"Expected url,label columns; got {fieldnames}")

    existing_keys = {normalize_for_dedup(r["url"]) for r in rows}

    phishing = [r for r in rows if str(r["label"]) == "1"]
    print(f"Loaded {len(rows)} rows | {len(phishing)} phishing | "
          f"{len(rows) - len(phishing)} legitimate")

    # Build counterfactuals: for each sampled phishing URL, flip its scheme.
    # bare -> add https://   |   has-scheme -> strip to bare
    rng.shuffle(phishing)
    new_rows = []
    for r in phishing:
        if len(new_rows) >= args.n_augment:
            break
        url = r["url"].strip()
        if has_scheme(url):
            flipped = strip_scheme(url)            # https://x -> x
        else:
            flipped = "https://" + url             # x -> https://x
        # Skip if the flipped form already exists anywhere (avoid exact dupes;
        # note dedup is scheme-insensitive, so we only add a counterfactual when
        # the OPPOSITE scheme form isn't already literally present).
        if flipped.strip().lower() in {x["url"].strip().lower() for x in rows}:
            continue
        new_rows.append({"url": flipped, "label": "1"})

    merged = rows + new_rows
    rng.shuffle(merged)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    # Stats
    https_added = sum(1 for r in new_rows if r["url"].lower().startswith("https://"))
    bare_added = len(new_rows) - https_added
    print(f"Added {len(new_rows)} counterfactual phishing rows:")
    print(f"  bare  -> https:// : {https_added}")
    print(f"  https -> bare     : {bare_added}")
    print(f"New total rows      : {len(merged)}")
    new_phish = sum(1 for r in merged if str(r['label']) == '1')
    print(f"New class balance   : phishing={new_phish}  "
          f"legitimate={len(merged) - new_phish}")
    print(f"Wrote {args.output}")
    print()
    print("Next: copy this over phishguard_gh_dataset.csv, run feature_engineering.py,")
    print("then retrain with model_hash.py, then re-run debug_predict.py + test_scheme_bias.py.")


if __name__ == "__main__":
    main()