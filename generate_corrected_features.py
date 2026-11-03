import ipaddress
import math
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract

FREE_TLDS = {
    ".ml", ".tk", ".cf", ".ga", ".gq", ".pw",
    ".xyz", ".top", ".click", ".online", ".site", ".buzz"
}
MFS_BRANDS = {
    "mtn", "vodafone", "airtel", "airteltigo", "momo",
    "gcb", "ecobank", "absa", "calbank", "stanbic", "bog"
}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "ow.ly", "rb.gy", "cutt.ly", "is.gd"
}
TRUSTED_GH = {
    ".com.gh", ".gov.gh", ".edu.gh", ".org.gh", ".net.gh"
}

extract_domain = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def extract(url):
    url = str(url).strip()
    url_l = url.lower()

    # Add a scheme only for parsing; preserve the original URL features.
    parse_text = url
    if parse_text.startswith("//"):
        parse_text = "https:" + parse_text
    elif "://" not in parse_text:
        parse_text = "https://" + parse_text

    parsed = urlsplit(parse_text)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""

    if not hostname:
        raise ValueError(f"No hostname found: {url!r}")

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    domain = extract_domain(hostname)
    subdomain_depth = (
        len(domain.subdomain.split("."))
        if domain.subdomain and not is_ip
        else 0
    )

    freq = {}
    for ch in url_l:
        freq[ch] = freq.get(ch, 0) + 1

    n = len(url_l) or 1
    entropy = -sum(
        (count / n) * math.log2(count / n)
        for count in freq.values()
    )

    free_tld = any(hostname.endswith(t) for t in FREE_TLDS)
    brand_present = any(b in url_l for b in MFS_BRANDS)

    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "digit_ratio": round(
            sum(c.isdigit() for c in url) / max(len(url), 1), 4
        ),
        "alpha_ratio": round(
            sum(c.isalpha() for c in url) / max(len(url), 1), 4
        ),
        "url_entropy": round(entropy, 4),
        "is_ip_hostname": is_ip,
        "subdomain_depth": subdomain_depth,
        "is_free_tld": int(free_tld),
        "is_trusted_gh_tld": int(
            any(hostname.endswith(t) for t in TRUSTED_GH)
        ),
        "brand_injected": int(brand_present),
        "brand_on_free_tld": int(brand_present and free_tld),
        "uses_shortener": int(
            any(
                hostname == s or hostname.endswith("." + s)
                for s in SHORTENERS
            )
        ),
        "has_at_symbol": int("@" in url),
        "is_https": int(url_l.startswith("https://")),
        "path_length": len(path),
    }


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "dataset_fix_outputs_v3"

    expected_sizes = {
        "train": 35664,
        "validation": 7668,
        "test": 7653,
    }

    # Stop before writing if any output already exists.
    for name in expected_sizes:
        target = folder / f"{name}_features.csv"
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")

    for name, expected in expected_sizes.items():
        source = folder / f"{name}.csv"
        df = pd.read_csv(source, keep_default_na=False)

        if not {"url", "label"}.issubset(df.columns):
            raise ValueError(f"{source.name} needs url and label columns.")

        if len(df) != expected:
            raise ValueError(
                f"{name}: expected {expected:,} rows, got {len(df):,}"
            )

        if not df["label"].isin([0, 1]).all():
            raise ValueError(f"Invalid labels in {source.name}")

        print(f"\nProcessing {name}: {len(df):,} URLs...")
        records = []

        for number, row in enumerate(
            df.itertuples(index=False), start=1
        ):
            features = extract(row.url)

            if len(features) != 23:
                raise ValueError("Expected exactly 23 numeric features.")

            records.append({
                "url": row.url,
                **features,
                "label": int(row.label),
            })

            if number % 5000 == 0:
                print(f"    {number:,}/{len(df):,}")

        result = pd.DataFrame(records)
        target = folder / f"{name}_features.csv"
        result.to_csv(target, index=False, encoding="utf-8")

        # Check the saved URLs and labels against the source partition.
        saved = pd.read_csv(target, keep_default_na=False)
        assert saved["url"].tolist() == df["url"].tolist()
        assert saved["label"].tolist() == df["label"].tolist()

        print(f"Saved: {target.name}")
        print("23 numeric features; URL and label alignment passed.")

    print("\nFEATURE EXTRACTION COMPLETE")
    print("Original datasets and partitions unchanged.")