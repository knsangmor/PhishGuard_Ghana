"""Prepare one fixed mixed-corpus development experiment; NO training/prediction.
PhiUSIIL is explicitly reclassified as development data. The existing Mendeley
holdout remains reserved. Original v3 datasets/results are never overwritten.
"""
import argparse
import hashlib
import io
import json
import re
import ipaddress
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter
PARTS={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
'validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
'test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
SOURCE_SHA='d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'
PHI_SHA='a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6'
HOLDOUT_SHA='b8a069c5d9671e74ad4c0f0b29ef1b3a40bd7c0c58c02d6884ae4e32bceeded8'
SALT='coverage-development-v1-seed42'


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def normalize(url):
    u=str(url).strip()
    if not u or any(c.isspace() for c in u) or any(c in u for c in "'\"\\"):return u
    m=re.match(r'^https?://',u,re.I)
    if m:prefix,rest=u[:m.end()],u[m.end():]
    elif u.startswith('//'):prefix,rest='//',u[2:]
    elif '://' in u:return u
    else:prefix,rest='',u
    try:
        p=urlsplit('https://'+rest)
        if not p.hostname or p.path!='':return u
        _=p.port
        return prefix+p.netloc+'/'+rest[len(p.netloc):]
    except ValueError:return u

def key(u,corrected=True):
    s=normalize(u) if corrected else str(u).strip()
    for p in ('https://','http://'):
        if s.lower().startswith(p):s=s[len(p):];break
    s=s.lower().strip()[:200]
    return sha_bytes(bytes([ord(c)-31 if 32<=ord(c)<=126 else 96 for c in s]).ljust(200,b'\0'))

def domain_bucket(group):
    return 'phi_validation' if int(sha_bytes((SALT+':'+group).encode())[:8],16)<int(.2*(2**32)) else 'phi_train'

def choose(frame,n_per_class):
    import pandas as pd
    selected=[]
    for label in [0,1]:
        candidates=frame[frame.label.eq(label)].sort_values(['sampling_key','source_row'])
        # Avoid a few large hosting domains dominating either class.
        candidates=candidates[candidates.groupby('domain_group').cumcount()<5]
        if len(candidates)<n_per_class:
            raise ValueError(f'Only {len(candidates)} eligible class-{label} rows after fixed per-domain cap; need {n_per_class}. No outputs created. Review coverage, do not automatically change sampling.')
        selected.append(candidates.iloc[:n_per_class])
    return pd.concat(selected).sort_values(['sampling_key','source_row'])

def self_test():
    assert key('http://example.com')==key('https://example.com/')
    assert key('www.example.com')!=key('example.com')
    assert normalize('https://example.com/a')=='https://example.com/a'
    assert domain_bucket('example.com')==domain_bucket('example.com')
    import pandas as pd
    toy=pd.DataFrame([{'label':y,'domain_group':f'g{i}.test','sampling_key':str(i),'source_row':i+10*y} for y in [0,1] for i in range(8)])
    selected=choose(toy,3)
    assert selected.label.value_counts().to_dict()=={0:3,1:3}
    print('SELF-TEST PASSED: encoder root equivalence and fixed class sampling.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--check-only',action='store_true');m.add_argument('--prepare',action='store_true');m.add_argument('--self-test',action='store_true')
    ap.add_argument('--phiusiil',default='PhiUSIIL_Phishing_URL_Dataset.csv')
    a=ap.parse_args()
    if a.self_test:self_test();return
    import numpy as np,pandas as pd
    root=Path(__file__).resolve().parent
    source=root/'malicious_phish.csv'
    if sha(source)!=SOURCE_SHA:raise ValueError('Original full-source fingerprint differs')
    reserve=root/'reserved_mendeley_holdout'
    freeze=json.loads((reserve/'freeze_report.json').read_text())
    if freeze['holdout_sha256']!=HOLDOUT_SHA or sha(reserve/'holdout.csv')!=HOLDOUT_SHA:
        raise ValueError('Reserved holdout identity mismatch')
    parts={}
    for name,h in PARTS.items():
        path=root/'dataset_fix_outputs_v3'/f'{name}.csv'
        if sha(path)!=h:raise ValueError('Frozen partition mismatch: '+name)
        parts[name]=pd.read_csv(path,keep_default_na=False)
        changed=sum(key(u)!=key(u,False) for u in parts[name].url)
        if changed!={'train':1736,'validation':474,'test':398}[name]:raise ValueError('Encoder differs from audited pilot')
    phipath=Path(a.phiusiil)
    if not phipath.is_absolute():phipath=root/phipath
    raw=phipath.read_bytes()
    if phipath.suffix.lower()=='.zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            matches=[n for n in z.namelist() if Path(n).name=='PhiUSIIL_Phishing_URL_Dataset.csv']
            if len(matches)!=1:raise ValueError('Original PhiUSIIL CSV not found uniquely')
            raw=z.read(matches[0])
    if sha_bytes(raw)!=PHI_SHA:raise ValueError('PhiUSIIL fingerprint differs')
    print('INPUT CHECK PASSED: frozen v3, complete source, PhiUSIIL, and reserved holdout hash.')
    if a.check_only:
        print('No outputs, training, inference, or holdout URL parsing.');return
    out=root/'coverage_development_v1'
    if out.exists():raise FileExistsError(f'Existing preparation preserved: {out}')
    import tldextract
    from functools import lru_cache
    extract=tldextract.TLDExtract(suffix_list_urls=(),include_psl_private_domains=True)
    @lru_cache(maxsize=None)
    def host_group(h):
        try:return str(ipaddress.ip_address(h))
        except ValueError:return extract(h).top_domain_under_public_suffix or h
    def domain(url):
        try:
            u=str(url).strip();u='https:'+u if u.startswith('//') else u
            if '://' not in u:u='https://'+u
            h=(urlsplit(u).hostname or '').lower().rstrip('.')
            return host_group(h) if h else ''
        except ValueError:return ''
    all_v3=pd.concat(parts.values(),ignore_index=True)
    v3_domains={domain(u) for u in all_v3.url};v3_domains.discard('')
    v3_new={key(u) for u in all_v3.url};v3_old={key(u,False) for u in all_v3.url}
    print('Auditing all original-source labels under the fixed root encoder...',flush=True)
    bits={};seen=0
    for chunk in pd.read_csv(source,usecols=['url','type'],keep_default_na=False,chunksize=100000):
        for u,t in zip(chunk.url,chunk.type):
            t=str(t).strip().lower()
            if t=='benign':b=1
            elif t in {'phishing','malware','defacement','malicious','1'}:b=2
            else:raise ValueError('Unrecognized original source label: '+t)
            k=key(u);bits[k]=bits.get(k,0)|b
        seen+=len(chunk);print(f'Original source: {seen:,} rows',flush=True)
    d=pd.read_csv(io.BytesIO(raw),usecols=['URL','label'],keep_default_na=False).rename(columns={'URL':'url','label':'source_label'})
    if len(d)!=235795 or d.source_label.value_counts().to_dict()!={1:134850,0:100945}:raise ValueError('Unexpected PhiUSIIL labels/counts')
    d['label']=1-d.source_label;d['source_row']=range(1,len(d)+1);d['source']='PhiUSIIL_development'
    d['key']=d.url.map(key);d['old_key']=d.url.map(lambda u:key(u,False));d['domain_group']=d.url.map(domain)
    conflicting=set(d.groupby('key').label.nunique().loc[lambda s:s>1].index)
    reasons=pd.Series('',index=d.index)
    flags={'invalid_host':d.domain_group.eq(''),
       'conflicting_phiusiil_encoded_labels':d.key.isin(conflicting),
       'opposite_full_original_source_label':pd.Series([bool(bits.get(k,0)&(2 if y==0 else 1)) for k,y in zip(d.key,d.label)],index=d.index),
       'domain_overlap_any_v3_partition':d.domain_group.isin(v3_domains),
       'old_input_overlap_any_v3_partition':d.old_key.isin(v3_old),
       'new_input_overlap_any_v3_partition':d.key.isin(v3_new)}
    for name,mask in flags.items():reasons.loc[mask & reasons.eq('')]=name
    available=d[reasons.eq('')].copy()
    duplicate=available.duplicated('key',keep='first');reasons.loc[available.index[duplicate]]='repeated_encoded_input'
    eligible=d[reasons.eq('')].copy()
    eligible['bucket']=eligible.domain_group.map(domain_bucket)
    eligible['sampling_key']=[sha_bytes((SALT+':'+u).encode()) for u in eligible.url]
    print('Eligible development candidates:',pd.crosstab(eligible.bucket,eligible.label).to_dict(),flush=True)
    tr=choose(eligible[eligible.bucket.eq('phi_train')],8000)
    va=choose(eligible[eligible.bucket.eq('phi_validation')],2000)
    assert not set(tr.domain_group)&set(va.domain_group)
    assert not set(tr.key)&set(va.key)
    assert not set(tr.domain_group)&v3_domains and not set(va.domain_group)&v3_domains
    for frame in [tr,va]:
        assert not set(frame.key)&v3_new
    columns=['url','label','source','source_row','domain_group']
    original_train=parts['train'][['url','label']].copy()
    original_train['source']='v3_original_train';original_train['source_row']=range(1,len(original_train)+1)
    original_train['domain_group']=original_train.url.map(domain)
    train=pd.concat([original_train[columns],tr[columns]],ignore_index=True)
    if len(train)!=51664:raise ValueError('Unexpected combined training size')
    # Source role changes are explicit and apply only to this new experiment.
    report={'status':'PREPARED_NO_TRAINING','experiment':'coverage_development_v1',
        'phiusiil_role_change':'Previously examined external/diagnostic corpus is now development data. It is not independent test evidence for revised models.',
        'partition_sha256':PARTS,'original_source_sha256':SOURCE_SHA,'phiusiil_sha256':PHI_SHA,
        'reserved_holdout_sha256':HOLDOUT_SHA,'tldextract_version':tldextract.__version__,
        'original_training_rows':35664,'added_training_rows':len(tr),'combined_training_rows':len(train),
        'added_training_class_counts':{str(k):int(v) for k,v in tr.label.value_counts().items()},
        'original_validation_rows':7668,'phi_development_validation_rows':len(va),
        'phi_validation_class_counts':{str(k):int(v) for k,v in va.label.value_counts().items()},
        'eligible_phi_candidates':len(eligible),'exclusive_exclusions':dict(Counter(reasons[reasons.ne('')])),
        'sampling':{'salt':SALT,'domain_hash_validation_fraction':.2,'maximum_rows_per_domain_per_class':5,
          'training_rows_per_class':8000,'validation_rows_per_class':2000,'prediction_scores_used':False},
        'planned_model':'same root-normalized architecture and optimizer; one seed-42 pilot before scaling',
        'planned_early_stopping':'equal-source-weighted validation binary crossentropy; maximum 50 epochs; patience 12',
        'selection_policy':'report original-v3 and PhiUSIIL-development validation separately; no model selection or threshold tuning on reserved holdout',
        'evaluation_policy_revision':'The earlier PhiUSIIL diagnostic-only development role is superseded for this explicitly documented mixed-corpus experiment; the reserved final comparison is unchanged.',
        'reserved_holdout_scored':False,'limitations':['Full PhiUSIIL has already been examined; its development validation is not independent confirmation.',
           'Corpus labels are inherited, not live verified.', 'Source-specific formatting may still create shortcuts.',
           'Broader coverage does not guarantee improvement or deployment readiness.']}
    out.mkdir()
    train.to_csv(out/'train.csv',index=False)
    parts['validation'][['url','label']].to_csv(out/'validation_v3.csv',index=False)
    va[columns].to_csv(out/'validation_phiusiil_development.csv',index=False)
    rejected=d[reasons.ne('')][['source_row','url','source_label','label']].copy();rejected['reason']=reasons[reasons.ne('')]
    rejected.to_csv(out/'excluded_phiusiil.csv',index=False)
    report['output_sha256']={name:sha(out/name) for name in ['train.csv','validation_v3.csv','validation_phiusiil_development.csv','excluded_phiusiil.csv']}
    (out/'preparation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('PREPARATION COMPLETE:',out)
    print('No training, prediction, reserved-holdout URL parsing or existing-file changes.')

if __name__=='__main__':main()
