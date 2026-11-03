from urllib.parse import urlsplit
import tldextract

# Use the package's bundled Public Suffix List.
# Disable live downloads so this check works offline.
extractor = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)

def get_domain(url):
    text = url.strip()

    # Add a temporary scheme so Python can identify the hostname.
    if text.startswith("//"):
        text = "https:" + text
    elif "://" not in text:
        text = "https://" + text

    host = urlsplit(text).hostname

    if not host:
        raise ValueError(f"Cannot identify hostname: {url}")

    result = extractor(host)

    return result.top_domain_under_public_suffix or host.lower()

examples = [
    ("https://shop.example.co.uk/login", "example.co.uk"),
    ("https://news.other.co.uk/story", "other.co.uk"),
    ("https://www.example.com/about", "example.com"),
    ("https://account.example.com/login", "example.com"),
]

print("\nDOMAIN EXTRACTION CHECK")

for url, expected in examples:
    domain = get_domain(url)
    assert domain == expected, f"Expected {expected}, got {domain}"
    print(f"{url} → {domain}")

print("\nAll example checks passed.")
print("Your dataset has NOT been changed.")