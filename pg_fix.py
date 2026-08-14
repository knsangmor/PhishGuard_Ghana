
#pg_fix.py

import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def canonical(u):
    return re.sub(r'^https?://', '', str(u).strip()).lower().strip()


def regdom(u):
    host = canonical(u).split('/')[0].split('?')[0]
    p = host.split('.')
    return '.'.join(p[-2:]) if len(p) >= 2 else host


def group_split(df, seed=42):
   

    sort_key = df['url'].map(canonical)
    order = sort_key.sort_values(kind="stable").index
    df_sorted = df.loc[order].reset_index(drop=True)

    g = df_sorted['url'].map(regdom).values
    tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed)
                   .split(df_sorted, df_sorted.label, g))
    vr, ter = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed)
                   .split(df_sorted.iloc[tmp], df_sorted.label.iloc[tmp], g[tmp]))
    va, te = tmp[vr], tmp[ter]

    for a, b, n in [(tr, va, 'train/val'), (tr, te, 'train/test'), (va, te, 'val/test')]:
        if set(g[a]) & set(g[b]):
            raise RuntimeError("DOMAIN LEAKAGE " + n)
        ca = set(df_sorted.url.iloc[a].map(canonical))
        cb = set(df_sorted.url.iloc[b].map(canonical))
        if ca & cb:
            raise RuntimeError("URL LEAKAGE " + n)
    print("INTEGRITY PASSED: no leakage between any partitions")

    sorted_pos_to_orig_pos = order.to_numpy()
    tr_orig = sorted_pos_to_orig_pos[tr]
    va_orig = sorted_pos_to_orig_pos[va]
    te_orig = sorted_pos_to_orig_pos[te]

    return tr_orig, va_orig, te_orig


if __name__ == "__main__":
    df = pd.read_csv("phishguard_gh_dataset.csv")
    df['c'] = df.url.map(canonical)
    before = len(df)
    df = df.drop_duplicates('c').drop(columns='c').reset_index(drop=True)
    after = len(df)
    if after != before:
        df.to_csv("phishguard_gh_dataset.csv", index=False)
        print(f"dataset deduplicated: {before} -> {after} rows (file rewritten)")
    else:
        print(f"dataset already deduplicated: {after} rows (file NOT rewritten, no changes needed)")
    group_split(df)