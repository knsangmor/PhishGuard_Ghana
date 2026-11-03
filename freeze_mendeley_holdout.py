"""Audit and freeze a new holdout without loading any model or computing predictions.
Uses only raw URLs/status, ignores all supplied feature columns.
"""
import argparse
import hashlib
import io
import ipaddress
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter
SOURCE_SHA='21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c'
PHI_SHA='a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6'
PARTS={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
'validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
'test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
CITATION='Hannousse, Abdelhakim; Yahiouche, Salima (2021), Web page phishing detection, Mendeley Data, V3, doi: 10.17632/c2gw7fy2j4.3'

def sha(raw):return hashlib.sha256(raw).hexdigest()

def bare(u):
    s=str(u).strip()
    for p in ['https://','http://']:
        if s.lower().startswith(p):return s[len(p):]
    return s

def normalized(u):
    s=str(u).strip()
    if not s or any(c.isspace() for c in s) or any(c in s for c in "'\"\\"):return s
    m=re.match(r'^https?://',s,re.I)
    if m:prefix,rest=s[:m.end()],s[m.end():]
    elif s.startswith('//'):prefix,rest='//',s[2:]
    elif '://' in s:return s
    else:prefix,rest='',s
    try:
        p=urlsplit('https://'+rest)
        if not p.hostname or p.path!='':return s
        _=p.port
        return prefix+p.netloc+'/'+rest[len(p.netloc):]
    except ValueError:return s

def encoded(u,root_normalized):
    s=bare(normalized(u) if root_normalized else u).lower().strip()[:200]
    return bytes([ord(c)-31 if 32<=ord(c)<=126 else 96 for c in s]).ljust(200,b'\0').hex()

def phi_bytes(path):
    raw=path.read_bytes()
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            matches=[n for n in z.namelist() if Path(n).name=='PhiUSIIL_Phishing_URL_Dataset.csv']
            if len(matches)!=1:raise ValueError('Expected original PhiUSIIL CSV')
            raw=z.read(matches[0])
    if sha(raw)!=PHI_SHA:raise ValueError('PhiUSIIL source fingerprint differs')
    return raw

def audit(dataset,phi,reference_urls):
    import pandas as pd
    import tldextract
    from functools import lru_cache
    extract=tldextract.TLDExtract(suffix_list_urls=(),include_psl_private_domains=True)
    @lru_cache(maxsize=None)
    def group_host(h):
        try:return str(ipaddress.ip_address(h))
        except ValueError:return extract(h).top_domain_under_public_suffix or h
    def domain(u):
        try:
            s=str(u).strip();s='https:'+s if s.startswith('//') else s
            if '://' not in s:s='https://'+s
            h=(urlsplit(s).hostname or '').lower().rstrip('.')
            return group_host(h) if h else ''
        except ValueError:return ''
    raw=dataset.read_bytes()
    if sha(raw)!=SOURCE_SHA:raise ValueError('Mendeley CSV fingerprint differs from inspected file')
    d=pd.read_csv(io.BytesIO(raw),usecols=['url','status'],keep_default_na=False)
    if len(d)!=11430 or d.status.value_counts().to_dict()!={'legitimate':5715,'phishing':5715}:raise ValueError('Unexpected Mendeley labels/counts')
    d['label']=d.status.map({'legitimate':0,'phishing':1})
    d['source_row']=range(1,len(d)+1)
    d['canonical']=d.url.map(lambda u:bare(u).lower().strip())
    d['old_encoding']=d.url.map(lambda u:encoded(u,False))
    d['new_encoding']=d.url.map(lambda u:encoded(u,True))
    d['domain_group']=d.url.map(domain)
    flags={'invalid_hostname':d.domain_group.eq('')}
    for col in ['canonical','old_encoding','new_encoding']:
        bad=set(d.groupby(col).label.nunique().loc[lambda s:s>1].index)
        flags['conflicting_'+col+'_labels']=d[col].isin(bad)
    phi_raw=phi_bytes(phi)
    ph=pd.read_csv(io.BytesIO(phi_raw),usecols=['URL'],keep_default_na=False)
    sources={'frozen_v3':list(reference_urls),'full_phiusiil':ph.URL.tolist()}
    for name,urls in sources.items():
        print('Checking overlap against',name,f'({len(urls):,} URLs)...',flush=True)
        keys={bare(u).lower().strip() for u in urls}
        old={encoded(u,False) for u in urls};new={encoded(u,True) for u in urls}
        domains={domain(u) for u in urls};domains.discard('')
        flags[name+'_canonical_overlap']=d.canonical.isin(keys)
        flags[name+'_old_encoding_overlap']=d.old_encoding.isin(old)
        flags[name+'_new_encoding_overlap']=d.new_encoding.isin(new)
        flags[name+'_domain_overlap']=d.domain_group.isin(domains)
    reasons=pd.Series('',index=d.index)
    for name,mask in flags.items():reasons.loc[mask & reasons.eq('')]=name
    seen_old=set();seen_new=set();seen_keys=set()
    for i in d.index[reasons.eq('')]:
        r=d.loc[i]
        if r.canonical in seen_keys or r.old_encoding in seen_old or r.new_encoding in seen_new:
            reasons.at[i]='repeated_input'
        else:
            seen_keys.add(r.canonical);seen_old.add(r.old_encoding);seen_new.add(r.new_encoding)
    columns=['source_row','url','status','label','domain_group']
    kept=d.loc[reasons.eq(''),columns].copy()
    excluded=d.loc[reasons.ne(''),columns].copy();excluded['reason']=reasons[reasons.ne('')]
    if kept.label.nunique()!=2:raise ValueError('Both classes required after fixed exclusions; do not score models')
    report={'status':'AUDIT_COMPLETE_NO_MODEL_SCORING','citation':CITATION,
        'source_csv_sha256':SOURCE_SHA,'full_phiusiil_sha256':PHI_SHA,
        'raw_rows':len(d),'reference_v3_rows':len(sources['frozen_v3']),
        'reference_phiusiil_rows':len(ph),'label_mapping':{'legitimate':0,'phishing':1},
        'nonexclusive_flags':{k:int(v.sum()) for k,v in flags.items()},
        'exclusive_exclusions':dict(Counter(excluded.reason)),'retained_rows':len(kept),
        'retained_benign':int(kept.label.eq(0).sum()),'retained_phishing':int(kept.label.eq(1).sum()),
        'retained_domain_groups':int(kept.domain_group.nunique()),'tldextract_version':tldextract.__version__,
        'rules':'Fixed exclusions for invalid host, label conflicts, canonical/old/new encoded overlap and domain overlap with all v3 and full PhiUSIIL; stable source order deduplication.',
        'evaluation_policy':{'purpose':'reserved external comparison of frozen original vs corrected detector',
          'no_training_or_tuning_on_holdout':True,'model_scoring_performed':False,
          'development_data':'existing v3 train/validation; PhiUSIIL diagnostic only',
          'configuration_and_thresholds':'freeze before any holdout scoring; thresholds selected using original validation data only',
          'planned_seeds':[42,43,44,45,46],
          'report_metrics':['accuracy','precision','recall','F1','ROC-AUC','average precision','FPR','confusion matrices'],
          'report_negative_results':True},
        'limitations':['Independence of upstream collection is not proven by these exclusions.',
          'These are historical data, not a future-traffic test.',
          'Removing development-overlapping domains changes the source distribution.',
          'The final test must not be inspected for model-selection feedback; further revisions need a new evaluation.']}
    assert len(kept)+len(excluded)==len(d)
    return kept,excluded,report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--check-only',action='store_true');m.add_argument('--freeze',action='store_true')
    ap.add_argument('--dataset',default='dataset_B_05_2020.csv')
    ap.add_argument('--phiusiil',default='PhiUSIIL_Phishing_URL_Dataset.csv')
    a=ap.parse_args()
    import pandas as pd
    root=Path(__file__).resolve().parent
    reference=[]
    for name,h in PARTS.items():
        p=root/'dataset_fix_outputs_v3'/f'{name}.csv'
        if sha(p.read_bytes())!=h:raise ValueError('Frozen v3 partition mismatch: '+name)
        reference.extend(pd.read_csv(p,keep_default_na=False).url.tolist())
    def resolve(p):return Path(p) if Path(p).is_absolute() else root/p
    k,e,r=audit(resolve(a.dataset),resolve(a.phiusiil),reference)
    r['partition_sha256']=PARTS
    print(json.dumps(r,indent=2))
    if a.check_only:
        print('READ-ONLY AUDIT COMPLETE. No outputs, inference or training.');return
    target=root/'reserved_mendeley_holdout'
    target.mkdir(exist_ok=False)
    k.to_csv(target/'holdout.csv',index=False)
    e.to_csv(target/'excluded_records.csv',index=False)
    r['holdout_sha256']=sha((target/'holdout.csv').read_bytes())
    (target/'freeze_report.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print('HOLDOUT FROZEN:',target)
    print('No models loaded or scored. Existing data and results unchanged.')

if __name__=='__main__':main()
