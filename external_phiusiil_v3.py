"""Prepare and evaluate a separate PhiUSIIL holdout; never train or tune on it.
Citation supplied by researcher: Prasad, A., & Chandra, S. (2023). PhiUSIIL:
A diverse security profile empowered phishing URL detection framework based on
similarity index and incremental learning. Computers & Security, 103545.
https://doi.org/10.1016/j.cose.2023.103545
"""
import argparse
import hashlib
import io
import ipaddress
import json
import os
import zipfile
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
SOURCE_SHA = 'a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6'
PARTS = {'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
 'validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
 'test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
CITATION = 'Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. https://doi.org/10.1016/j.cose.2023.103545'


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p): return sha_bytes(p.read_bytes())

def canonical(u):
    s = str(u).strip()
    for p in ('https://', 'http://'):
        if s.lower().startswith(p):
            s = s[len(p):]
            break
    return s.lower().strip()

def encode(u):
    s = canonical(u)[:200]
    return bytes([ord(c)-31 if 32 <= ord(c) <= 126 else 96 for c in s]).ljust(200, b'\0')


def audit(source, references):
    import pandas as pd
    import tldextract
    from urllib.parse import urlsplit
    extractor = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
    def domain(u):
        try:
            s = str(u).strip()
            s = 'https:' + s if s.startswith('//') else s
            if '://' not in s: s = 'https://' + s
            host = (urlsplit(s).hostname or '').lower().rstrip('.')
            if not host: return ''
            try: return str(ipaddress.ip_address(host))
            except ValueError: return extractor(host).top_domain_under_public_suffix or host
        except ValueError: return ''
    raw = source.read_bytes()
    if source.suffix.lower() == '.zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if Path(n).name == 'PhiUSIIL_Phishing_URL_Dataset.csv']
            if len(names) != 1: raise ValueError('Expected one original PhiUSIIL CSV in ZIP')
            raw = z.read(names[0])
    if sha_bytes(raw) != SOURCE_SHA: raise ValueError('External CSV differs from the uploaded audited source')
    d = pd.read_csv(io.BytesIO(raw), usecols=['URL','label'], keep_default_na=False)
    if len(d) != 235795 or d.label.value_counts().to_dict() != {1:134850, 0:100945}:
        raise ValueError('Unexpected source counts or labels')
    d = d.rename(columns={'URL':'url','label':'source_label'})
    d['label'] = 1 - d.source_label  # UCI 1 legitimate / 0 phishing -> project 0 benign / 1 malicious.
    d['source_row'] = range(1, len(d)+1)
    d['canonical'] = d.url.map(canonical)
    d['encoding'] = d.url.map(lambda u: encode(u).hex())
    d['domain_group'] = d.url.map(domain)
    refs = list(references)
    keys = {canonical(u) for u in refs}
    encodings = {encode(u).hex() for u in refs}
    domains = {domain(u) for u in refs}
    domains.discard('')
    bad_keys = set(d.groupby('canonical').label.nunique().loc[lambda x:x>1].index)
    bad_enc = set(d.groupby('encoding').label.nunique().loc[lambda x:x>1].index)
    flags = {
        'invalid_hostname': d.domain_group.eq(''),
        'conflicting_canonical_labels': d.canonical.isin(bad_keys),
        'conflicting_encoded_labels': d.encoding.isin(bad_enc),
        'canonical_overlap_frozen_v3': d.canonical.isin(keys),
        'encoding_overlap_frozen_v3': d.encoding.isin(encodings),
        'domain_overlap_frozen_v3': d.domain_group.isin(domains),
    }
    reason = pd.Series('', index=d.index)
    for name, mask in flags.items(): reason.loc[mask & reason.eq('')] = name
    # Stable source-row order: only one representative per actual model input.
    seen = set()
    for i in d.index[reason.eq('')]:
        key = d.at[i,'encoding']
        if key in seen: reason.at[i] = 'repeated_encoded_input'
        else: seen.add(key)
    keep = reason.eq('')
    fields = ['source_row','url','source_label','label','domain_group']
    retained = d.loc[keep, fields].copy()
    excluded = d.loc[~keep, fields].copy()
    excluded['reason'] = reason.loc[~keep]
    if retained.label.nunique() != 2: raise ValueError('Both classes required after exclusions')
    report = {'source_csv_sha256':sha_bytes(raw), 'citation':CITATION,
        'raw_rows':len(d), 'source_label_mapping':{'0':'phishing -> project 1','1':'legitimate -> project 0'},
        'reference_rows':len(refs), 'reference_domain_groups':len(domains),
        'raw_flag_counts_nonexclusive':{k:int(v.sum()) for k,v in flags.items()},
        'exclusion_counts_exclusive':dict(Counter(excluded.reason)),
        'retained_rows':len(retained),'retained_benign':int(retained.label.eq(0).sum()),
        'retained_phishing':int(retained.label.eq(1).sum()),
        'domain_groups':int(retained.domain_group.nunique()),
        'external_encoded_lengths_over_200':int(retained.url.map(lambda u:len(canonical(u))>200).sum()),
        'tldextract_version':tldextract.__version__,
        'public_suffix_handling':'bundled offline snapshot; private suffixes included; IPs grouped by address',
        'limitations':['A separate published corpus does not prove independent upstream collection.',
          'Overlap removal is against all frozen v3 partitions, not the entire unused original corpus.',
          'No per-URL collection timestamps supplied: no temporal generalization claim.',
          'External positives are phishing; training positives include malware and defacement.',
          'Domain-disjoint filtering changes the external class and domain distribution.']}
    return retained, excluded, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--seed', type=int, choices=range(42,47))
    ap.add_argument('--source', default='phiusiil+phishing+url+dataset.zip')
    a = ap.parse_args()
    import pandas as pd
    import numpy as np
    root = Path(__file__).resolve().parent
    target = root / 'external_phiusiil_v3'
    frames = []
    for name, expected in PARTS.items():
        p = root / 'dataset_fix_outputs_v3' / f'{name}.csv'
        if sha(p) != expected: raise ValueError(f'Frozen {name} fingerprint differs')
        frames.append(pd.read_csv(p, keep_default_na=False))
    if a.check_only or a.prepare:
        src = Path(a.source)
        if not src.is_absolute(): src = root / src
        retained, excluded, report = audit(src, pd.concat(frames,ignore_index=True).url)
        report['partition_sha256'] = PARTS
        print(json.dumps(report, indent=2))
        if a.check_only:
            print('CHECK COMPLETE: no models loaded, training or output folders created.')
            return
        target.mkdir(exist_ok=False)
        retained.to_csv(target / 'external_test.csv', index=False)
        excluded.to_csv(target / 'excluded_records.csv', index=False)
        report['external_test_sha256'] = sha(target / 'external_test.csv')
        (target / 'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('PREPARED:', target)
        print('Frozen model and dataset files unchanged. No training or inference performed.')
        return
    report = json.loads((target / 'audit_report.json').read_text())
    if report['partition_sha256'] != PARTS or sha(target/'external_test.csv') != report['external_test_sha256']:
        raise ValueError('Prepared holdout identity mismatch')
    d = pd.read_csv(target/'external_test.csv',keep_default_na=False)
    run = root/'corrected_results_v3'/f'seed_{a.seed}'
    test = frames[2]
    expected_x = np.asarray([list(encode(u)) for u in test.url],dtype=np.int32)
    if not np.array_equal(expected_x,np.load(run/'X_test.npy',allow_pickle=False)):
        raise ValueError('Main-model encoding mismatch')
    output = target/f'seed_{a.seed}'
    if output.exists(): raise FileExistsError(f'Existing results preserved: {output}')
    import keras
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,roc_curve
    model_path = run/'phishguard_gh_model.keras'
    model = keras.models.load_model(model_path,compile=False)
    saved = pd.read_csv(run/'test_predictions.csv',keep_default_na=False)
    if test.url.tolist()!=saved.url.tolist() or test.label.tolist()!=saved.true_label.tolist():
        raise ValueError('Original test alignment mismatch')
    base = model.predict(expected_x,batch_size=256,verbose=0).reshape(-1)
    if not np.allclose(base,saved.probability.to_numpy(),rtol=1e-5,atol=1e-5):
        raise ValueError('Loaded model differs from saved test predictions')
    # Select operating thresholds using only the original frozen validation scores.
    va=pd.read_csv(run/'validation_predictions.csv',keep_default_na=False)
    if va.url.tolist()!=frames[1].url.tolist() or va.true_label.tolist()!=frames[1].label.tolist():
        raise ValueError('Validation alignment mismatch')
    thresholds={'default_0.5':.5}
    fpr,tpr,ts=roc_curve(va.true_label,va.probability,drop_intermediate=False)
    for cap in (.001,.005,.01):
        ix=np.where(fpr<=cap)[0]; best=ix[np.argmax(tpr[ix])]
        thresholds[f'validation_fpr_cap_{cap}']=float(ts[best])
    print(f'Predicting {len(d):,} external URLs, seed {a.seed}...',flush=True)
    chunks=[]
    for start in range(0,len(d),4096):
        x=np.asarray([list(encode(u)) for u in d.url.iloc[start:start+4096]],dtype=np.int32)
        chunks.append(model.predict(x,batch_size=256,verbose=0).reshape(-1))
        print(f'{min(start+4096,len(d)):,}/{len(d):,}',flush=True)
    p=np.concatenate(chunks); y=d.label.to_numpy()
    operating={}
    for name,t in thresholds.items():
        q=p>=t;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
        operating[name]={'threshold':t if np.isfinite(t) else None,'threshold_is_positive_infinity':bool(np.isposinf(t)),
          'accuracy':float(accuracy_score(y,q)),'precision':float(precision_score(y,q,zero_division=0)),
          'recall':float(recall_score(y,q,zero_division=0)),'f1':float(f1_score(y,q,zero_division=0)),
          'false_positive_rate':float(fp/(tn+fp)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    result={'seed':a.seed,'external_rows':len(d),'external_test_sha256':report['external_test_sha256'],
       'model_sha256':sha(model_path),'auc_roc':float(roc_auc_score(y,p)),
       'average_precision':float(average_precision_score(y,p)), 'operating_points':operating,
       'threshold_selection':'original validation set only; no external tuning',
       'citation':CITATION,'limitations':report['limitations']}
    output.mkdir()
    d['probability']=p;d['prediction_at_0_5']=(p>=.5).astype(int)
    d.to_csv(output/'predictions.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))
    print('COMPLETE:',output,'; no training or source changes.')

if __name__=='__main__': main()
