# """
# Generates realistic, modern-style legitimate URLs from a Tranco top-domains
# list.

# USAGE
# -----
# 1. Download https://tranco-list.eu/top-1m.csv.zip and unzip it next to
#    this script (or pass --tranco_csv with the path).
# """

import argparse
import csv
import random
import sys
from pathlib import Path
from urllib.parse import urlparse


# Modern, short, navigational path templates  the style your training data

PATH_TEMPLATES = [
    "", "", "", "",                      # bare root, weighted heavily
    "/about", "/about-us", "/contact", "/contact-us",
    "/news", "/blog", "/press", "/media",
    "/products", "/services", "/solutions", "/pricing",
    "/login", "/signin", "/signup", "/register", "/account",
    "/support", "/help", "/faq", "/docs", "/documentation",
    "/careers", "/jobs", "/team", "/leadership",
    "/privacy", "/terms", "/legal", "/cookies",
    "/en/home", "/en/about", "/en/about-us", "/en/news",
    "/store", "/shop", "/cart", "/checkout",
    "/dashboard", "/profile", "/settings",
    "/download", "/downloads", "/api/docs",
    "/events", "/resources", "/insights", "/research",
]

# Rank bands to stratify sampling across, so we don't just re-learn 50 new
# mega-brands instead of one. (start_rank, end_rank, weight)
RANK_BANDS = [
    (1, 1000, 0.15),
    (1000, 10000, 0.20),
    (10000, 50000, 0.25),
    (50000, 200000, 0.25),
    (200000, 1000000, 0.15),
]


def normalize_for_dedup(url: str) -> str:
    """Loose normalization so dedup catches scheme/www/trailing-slash variants."""
    u = url.strip().lower()
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
            break
    if u.startswith("www."):
        u = u[4:]
    return u.rstrip("/")


def load_tranco(path: str):
    """Tranco's permanent list ships as headerless rank,domain CSV."""
    domains = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rank = int(row[0])
            except ValueError:
                continue  # skip a stray header row if present
            domains.append((rank, row[1].strip().lower()))
    if not domains:
        raise ValueError(f"No usable rows found in {path} — check the file format.")
    return domains


def stratified_sample(domains, n_total, rng):
    """Sample domains across rank bands so the augmentation isn't just
    the top-50 mega-brands repeated with different paths."""
    by_rank = sorted(domains, key=lambda d: d[0])
    max_rank = by_rank[-1][0]
    picked = []
    for start, end, weight in RANK_BANDS:
        band = [d for d in by_rank if start <= d[0] <= min(end, max_rank)]
        if not band:
            continue
        k = max(1, int(n_total * weight))
        picked.extend(rng.sample(band, min(k, len(band))))
    rng.shuffle(picked)
    return picked[:n_total]


def build_augmentation(domains, paths_per_domain, rng):
    rows = []
    for rank, domain in domains:
        k = rng.randint(1, paths_per_domain)
        chosen_paths = rng.sample(PATH_TEMPLATES, min(k, len(PATH_TEMPLATES)))
        use_www = rng.random() < 0.5
        host = f"www.{domain}" if use_www else domain
        for path in set(chosen_paths):
            url = f"https://{host}{path}"
            rows.append(url)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranco_csv", required=True, help="Path to Tranco's top-1m.csv")
    ap.add_argument("--existing_dataset", required=True, help="Path to phishguard_gh_dataset.csv")
    ap.add_argument("--n_domains", type=int, default=1500, help="How many domains to sample")
    ap.add_argument("--paths_per_domain", type=int, default=2, help="Max path variants per domain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmentation_output", default="modern_legit_augmentation.csv",
                     help="Output file containing ONLY the new rows")
    ap.add_argument("--merged_output", default=None,
                     help="If given, also write a full merged dataset (original + new, shuffled)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading Tranco list from {args.tranco_csv} ...")
    domains = load_tranco(args.tranco_csv)
    print(f"  {len(domains)} domains loaded.")

    print(f"Loading existing dataset from {args.existing_dataset} ...")
    existing_urls = set()
    existing_rows = []
    with open(args.existing_dataset, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            existing_rows.append(row)
            existing_urls.add(normalize_for_dedup(row["url"]))
    print(f"  {len(existing_rows)} existing rows loaded.")

    print(f"Sampling {args.n_domains} domains across rank bands ...")
    sampled = stratified_sample(domains, args.n_domains, rng)
    print(f"  {len(sampled)} domains sampled "
          f"(rank range {min(r for r,_ in sampled)}-{max(r for r,_ in sampled)}).")

    print("Generating modern-style legitimate URLs ...")
    candidate_urls = build_augmentation(sampled, args.paths_per_domain, rng)

    seen_in_batch = set()
    new_rows = []
    for url in candidate_urls:
        key = normalize_for_dedup(url)
        if key in existing_urls or key in seen_in_batch:
            continue
        seen_in_batch.add(key)
        new_rows.append({"url": url, "label": 0})

    print(f"  {len(candidate_urls)} candidate URLs generated, "
          f"{len(new_rows)} kept after de-duplication "
          f"({len(candidate_urls) - len(new_rows)} duplicates dropped).")

    with open(args.augmentation_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        writer.writerows(new_rows)
    print(f"Wrote {len(new_rows)} new rows to {args.augmentation_output}")

    # ---- sanity stats, so you can sanity-check this before retraining ----
    https_count = sum(1 for r in new_rows if r["url"].lower().startswith("https://"))
    path_lengths = [len(urlparse(r["url"]).path) for r in new_rows]
    unique_domains = len({urlparse(r["url"]).netloc.replace("www.", "") for r in new_rows})
    print()
    print("Sanity check on the new rows:")
    print(f"  HTTPS fraction        : {https_count/len(new_rows)*100:.1f}%  "
          f"(training set overall was 0.42% for the legitimate class)")
    print(f"  Mean path length      : {sum(path_lengths)/len(path_lengths):.1f}  "
          f"(original legit mean was 47.96; phishing mean was 27.87)")
    print(f"  Unique domains added  : {unique_domains}")

    if args.merged_output:
        merged = existing_rows + new_rows
        rng.shuffle(merged)
        with open(args.merged_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
        print(f"\nWrote merged dataset ({len(merged)} rows) to {args.merged_output}")
       


if __name__ == "__main__":
    main()