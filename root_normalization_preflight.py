"""Audit a proposed empty-root-path normalization, without changing data or training.
RFC 3986 section 6.2.3: https://www.rfc-editor.org/rfc/rfc3986#section-6.2.3
Keeps www and nonempty paths intact. Schemeless corpus entries are treated as web URLs.
Outputs diagnostic reports only; proposed representation is not adopted automatically.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit
HASHES={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
'validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
'test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
SOURCE_SHA='d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def normalize(url):
    u=str(url).strip()
    if not u or any(c.isspace() for c in u) or any(c in u for c in "'\"\\"):
        return u
    m=re.match(r'^https?://',u,re.I)
    if m:prefix,rest=u[:m.end()],u[m.end():]
    elif u.startswith('//'):prefix,rest='//',u[2:]
    else:
        if '://' in u:return u
        prefix,rest='',u
    try:
        p=urlsplit('https://'+rest)
        if not p.hostname or p.path!='':return u
        _=p.port
        # Insert exactly one root slash before any query or fragment.
        return prefix+p.netloc+'/'+rest[len(p.netloc):]
    except ValueError:return u


def key(url,proposed=True):
    s=normalize(url) if proposed else str(url).strip()
    for prefix in ('https://','http://'):
        if s.lower().startswith(prefix):s=s[len(prefix):];break
    tokens=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in s.lower().strip()[:200]]
    return hashlib.sha256(bytes(tokens).ljust(200,b'\0')).hexdigest()


def self_test():
    cases={'http://example.com':'http://example.com/',
           'https://www.example.com?x=1#route':'https://www.example.com/?x=1#route',
           'example.com#':'example.com/#','//example.com':'//example.com/',
           'https://example.com/a':'https://example.com/a',
           'https://example.com/a/':'https://example.com/a/',
           "'example.com'":"'example.com'"}
    for u,expected in cases.items():
        assert normalize(u)==expected,(u,normalize(u))
        assert normalize(normalize(u))==normalize(u)
    assert key('http://www.example.com')==key('https://www.example.com/')
    assert key('example.com')!=key('www.example.com')
    print('SELF-TEST PASSED: root insertion, www preservation, nonempty paths and idempotence.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    import pandas as pd
    root=Path(__file__).resolve().parent
    source=root/'malicious_phish.csv'
    if sha(source)!=SOURCE_SHA:raise ValueError('Original full-source SHA256 mismatch')
    parts={}
    for name,h in HASHES.items():
        p=root/'dataset_fix_outputs_v3'/f'{name}.csv'
        if sha(p)!=h:raise ValueError(f'Frozen {name} SHA256 differs')
        parts[name]=pd.read_csv(p,keep_default_na=False)
    if args.check_only:
        print('INPUT CHECK PASSED: full source and frozen v3 partitions. No reports or training.');return
    source_bits={}
    seen=0
    print('Auditing proposed encoding against all original source labels...',flush=True)
    for chunk in pd.read_csv(source,usecols=['url','type'],keep_default_na=False,chunksize=100000):
        for u,t in zip(chunk.url,chunk.type):
            category=str(t).lower().strip()
            if category=='benign':bit=1
            elif category in {'phishing','malware','defacement','malicious','1'}:bit=2
            else:raise ValueError(f'Unexpected source category {category!r}')
            k=key(u);source_bits[k]=source_bits.get(k,0)|bit
        seen+=len(chunk);print(f'Source rows audited: {seen:,}',flush=True)
    sets={}; labels={}; summaries=[]; details=[]
    for name,d in parts.items():
        ks=[key(u) for u in d.url];sets[name]=set(ks);labels[name]=ks
        own={}
        for k,y in zip(ks,d.label):own[k]=own.get(k,0)|(1 if y==0 else 2)
        src_bad=own_bad=changed=0
        for i,(u,y,k) in enumerate(zip(d.url,d.label,ks)):
            opposite=2 if y==0 else 1
            a=bool(source_bits.get(k,0)&opposite);b=own[k]==3
            src_bad+=a;own_bad+=b;changed+=key(u,False)!=k
            if a or b:details.append({'partition':name,'row_index':i,'url':u,'label':int(y),
                'proposed_url':normalize(u),'opposite_source_label':a,'within_partition_conflict':b})
        summaries.append({'partition':name,'rows':len(d),'encoded_inputs_changed':changed,
            'opposite_source_label_rows':src_bad,'within_partition_conflict_rows':own_bad,
            'repeated_input_rows':len(ks)-len(set(ks))})
    overlaps=[]
    names=list(parts)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            shared=sets[a]&sets[b]
            overlaps.append({'first':a,'second':b,'shared_encoded_inputs':len(shared)})
            for name in [a,b]:
                for row,k in enumerate(labels[name]):
                    if k in shared:
                        r=parts[name].iloc[row]
                        details.append({'partition':name,'row_index':row,'url':r.url,'label':int(r.label),
                            'proposed_url':normalize(r.url),'cross_partition_overlap':a+'/'+b})
    out=root/'root_normalization_preflight'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    status='REVIEW_REQUIRED' if any(r['opposite_source_label_rows'] or r['within_partition_conflict_rows'] for r in summaries) or any(r['shared_encoded_inputs'] for r in overlaps) else 'NO_CONFLICTS_FOUND'
    report={'status':status,'source_sha256':SOURCE_SHA,'partition_sha256':HASHES,'source_rows':seen,
        'representation':'existing scheme stripping/lowercase/first200/rightpad; insert / only for empty web root paths',
        'summaries':summaries,'overlaps':overlaps,'note':'No data or models changed. No external test selected. Do not train before reviewing this audit and freezing a fresh external evaluation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(summaries).to_csv(out/'partition_summary.csv',index=False)
    pd.DataFrame(details,columns=['partition','row_index','url','label','proposed_url','opposite_source_label','within_partition_conflict','cross_partition_overlap']).to_csv(out/'flagged_records.csv',index=False)
    print(json.dumps(report,indent=2));print('Saved reports:',out)
    print('No training or dataset/model changes performed.')

if __name__=='__main__':main()
