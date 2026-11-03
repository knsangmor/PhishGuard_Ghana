from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent / "dataset_fix_outputs"
MAX_LEN = 200


def normalized_url(url):
    text = str(url).strip()

    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    return text.lower().strip()


print("\nURL LENGTH AND TRUNCATION CHECK")

total_rows = 0
total_truncated = 0

for name in ["train", "validation", "test"]:
    df = pd.read_csv(
        folder / f"{name}.csv",
        keep_default_na=False,
    )

    lengths = df["url"].map(normalized_url).str.len()
    truncated = int((lengths > MAX_LEN).sum())

    print(f"\n{name.upper()}")
    print("URLs:", len(df))
    print("Median encoded length:", lengths.median())
    print("Maximum encoded length:", lengths.max())
    print("URLs exceeding 200 characters:", truncated)
    print(
        "Truncated percentage:",
        f"{100 * truncated / len(df):.2f}%",
    )

    for label, description in [(0, "Benign"), (1, "Malicious")]:
        selected = lengths[df["label"] == label]
        count = int((selected > MAX_LEN).sum())
        print(
            f"{description}: {count:,}/{len(selected):,} truncated "
            f"({100 * count / len(selected):.2f}%)"
        )

    total_rows += len(df)
    total_truncated += truncated

print("\nTOTAL")
print("URLs:", total_rows)
print("Truncated URLs:", total_truncated)
print(
    "Truncated percentage:",
    f"{100 * total_truncated / total_rows:.2f}%",
)
print("\nNo files were changed and no training was started.")