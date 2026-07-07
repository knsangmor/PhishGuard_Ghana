"""Drop this file into your project folder. It gives you the corrected
split + safety checks. You do NOT need to edit it."""
import re, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def canonical(u): return re.sub(r'^https?://','',str(u).strip()).lower().strip()
def regdom(u):
    host = canonical(u).split('/')[0].split('?')[0]
    p = host.split('.'); return '.'.join(p[-2:]) if len(p)>=2 else host

def group_split(df, seed=42):
    g = df['url'].map(regdom).values
    tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(df, df.label, g))
    vr, ter = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(df.iloc[tmp], df.label.iloc[tmp], g[tmp]))
    va, te = tmp[vr], tmp[ter]
    for a,b,n in [(tr,va,'train/val'),(tr,te,'train/test'),(va,te,'val/test')]:
        if set(g[a])&set(g[b]): raise RuntimeError("DOMAIN LEAKAGE "+n)
        ca=set(df.url.iloc[a].map(canonical)); cb=set(df.url.iloc[b].map(canonical))
        if ca&cb: raise RuntimeError("URL LEAKAGE "+n)
    print("INTEGRITY PASSED: no leakage between any partitions")
    return tr, va, te

if __name__ == "__main__":
    df = pd.read_csv("phishguard_gh_dataset.csv")
    df['c'] = df.url.map(canonical)
    df = df.drop_duplicates('c').drop(columns='c').reset_index(drop=True)
    df.to_csv("phishguard_gh_dataset.csv", index=False)
    print("dataset deduplicated:", len(df), "rows")
    group_split(df)