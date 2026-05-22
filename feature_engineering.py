import pandas as pd, re, math
from urllib.parse import urlparse

print("PhishGuard-GH Feature Engineering")

FREE_TLDS={".ml",".tk",".cf",".ga",".gq",".pw",".xyz",".top",".click",".online",".site",".buzz"}
MFS_BRANDS={"mtn","vodafone","airtel","airteltigo","momo","gcb","ecobank","absa","calbank","stanbic","bog"}
SHORTENERS={"bit.ly","tinyurl.com","ow.ly","rb.gy","cutt.ly","is.gd"}
TRUSTED_GH={".com.gh",".gov.gh",".edu.gh",".org.gh",".net.gh"}

def extract(url):
    url=str(url).strip(); url_l=url.lower()
    try: hostname=urlparse(url).netloc or ""; path=urlparse(url).path or ""
    except: hostname=path=""
    freq={}
    for ch in url_l: freq[ch]=freq.get(ch,0)+1
    n=len(url_l) or 1
    entropy=-sum((c/n)*math.log2(c/n) for c in freq.values() if c>0)
    sub=[s for s in hostname.split(".") if s]
    return {"url_length":len(url),"num_dots":url.count("."),"num_hyphens":url.count("-"),
            "num_underscores":url.count("_"),"num_slashes":url.count("/"),"num_at":url.count("@"),
            "num_question":url.count("?"),"num_equals":url.count("="),"num_ampersand":url.count("&"),
            "num_percent":url.count("%"),
            "digit_ratio":round(sum(c.isdigit() for c in url)/max(len(url),1),4),
            "alpha_ratio":round(sum(c.isalpha() for c in url)/max(len(url),1),4),
            "url_entropy":round(entropy,4),
            "is_ip_hostname":int(bool(__import__('re').fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?",hostname))),
            "subdomain_depth":max(len(sub)-2,0),
            "is_free_tld":int(any(hostname.endswith(t) for t in FREE_TLDS)),
            "is_trusted_gh_tld":int(any(hostname.endswith(t) for t in TRUSTED_GH)),
            "brand_injected":int(any(b in url_l for b in MFS_BRANDS)),
            "brand_on_free_tld":int(any(b in url_l for b in MFS_BRANDS) and any(hostname.endswith(t) for t in FREE_TLDS)),
            "uses_shortener":int(any(s in url_l for s in SHORTENERS)),
            "has_at_symbol":int("@" in url),"is_https":int(url_l.startswith("https://")),"path_length":len(path)}

df=pd.read_csv("phishguard_gh_dataset.csv"); print(f"\nProcessing {len(df):,} URLs...")
rows=[]
for i,row in df.iterrows():
    feat=extract(row["url"]); feat["label"]=row["label"]; rows.append(feat)
    if (i+1)%5000==0: print(f"  {i+1:,}/{len(df):,} processed...")
pd.DataFrame(rows).to_csv("phishguard_features.csv",index=False)
print(" \n FEATURE EXTRACTION COMPLETE!")
print(f"  23 features | {len(rows):,} URLs \n Saved: phishguard_features.csv")

