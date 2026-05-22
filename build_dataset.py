import pandas as pd , numpy as np, os


print("PhishGuard-GH Dataset")


CSV_FILE = "malicious_phish.csv"
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"ERROR: {CSV_FILE} not found. Download from Kaggle.")

print(f"\nLoading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"  Loaded {len(df):,} rows | Columns: {list(df.columns)}")
df.columns = [c.lower().strip() for c in df.columns]
if 'url' not in df.columns:
    df = df.rename(columns={df.columns[0]: 'url', df.columns[1]: 'type'})
label_col = 'type' if 'type' in df.columns else 'label'
phishing_labels = ['phishing','malicious','defacement','malware','1',1]
df['label'] = df[label_col].apply(
    lambda x: 1 if str(x).lower().strip() in [str(l).lower() for l in phishing_labels] else 0
)
N = 25000
phish = df[df['label']==1].sample(min(N,(df['label']==1).sum()), random_state=42)
legit = df[df['label']==0].sample(min(N,(df['label']==0).sum()), random_state=42)
balanced = pd.concat([phish,legit],ignore_index=True)[['url','label']].dropna()
balanced = balanced.sample(frac=1,random_state=42).reset_index(drop=True)
balanced.to_csv("phishguard_gh_dataset.csv", index=False)

print("\n DATASET READY")

print(f"  Total: {len(balanced):,}  |  Phishing: {(balanced['label']==1).sum():,}  |  Legitimate: {(balanced['label']==0).sum():,}")
print("  Saved: phishguard_gh_dataset.csv")

