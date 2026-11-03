"""Validation-only 2x2 formatting diagnosis using frozen v3 models.
No training, network access, external test selection, or dataset modification.
Synthetic variants are not verified equivalent live resources; source labels
are used only to stratify sensitivity, not to score modified-URL accuracy.
"""
import argparse
import hashlib
import ipaddress
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
VAL_HASH = '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e'
NAMES = ('no_www_empty', 'www_empty', 'no_www_slash', 'www_slash')
LIMITATION = ('Paired synthetic sensitivity diagnosis on eligible original validation root URLs. '
 'www and non-www hosts may serve different resources; a root slash may affect application handling. '
 'No live resource equivalence was verified. Source labels stratify the analysis only. '
 'Do not interpret variant positive rates as phishing recall or false-positive rates.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def encode(url):
    u = str(url).strip()
    for prefix in ('https://','http://'):
        if u.lower().startswith(prefix):
            u = u[len(prefix):]
            break
    seq = [ord(c)-31 if 32 <= ord(c) <= 126 else 96 for c in u.lower().strip()[:200]]
    return seq + [0]*(200-len(seq))


def variants(url):
    u = str(url).strip()
    if not u or any(c.isspace() for c in u): return None, 'whitespace_or_empty'
    if u[0] in "'\"": return None, 'quoted_url'
    match = re.match(r'^https?://',u,re.I)
    if match: prefix, rest = u[:match.end()], u[match.end():]
    elif u.startswith('//'): prefix, rest = '//', u[2:]
    else:
        if '://' in u: return None, 'other_scheme'
        prefix, rest = '', u
    try:
        p = urlsplit('https://' + rest)
        if p.path not in ('','/'): return None, 'non_root_path'
        if p.username is not None or p.password is not None: return None, 'credentials'
        host = p.hostname or ''
        if not host: return None, 'missing_host'
        _ = p.port
        try:
            ipaddress.ip_address(host)
            return None, 'ip_hostname'
        except ValueError: pass
        ascii_host = host.encode('idna').decode('ascii')
        if not re.fullmatch(r'[A-Za-z0-9.-]+',ascii_host): return None, 'non_dns_host'
        if any(not label or len(label)>63 or label.startswith('-') or label.endswith('-')
               for label in ascii_host.rstrip('.').split('.')): return None, 'invalid_dns_label'
        netloc = p.netloc[4:] if p.netloc.lower().startswith('www.') else p.netloc
        if not netloc: return None, 'missing_base_host'
        # Retain query/fragment delimiters exactly (including empty ? or #).
        tail_start = len(p.netloc) + len(p.path)
        tail = rest[tail_start:]
        result = {
            'no_www_empty':prefix+netloc+tail,
            'www_empty':prefix+'www.'+netloc+tail,
            'no_www_slash':prefix+netloc+'/'+tail,
            'www_slash':prefix+'www.'+netloc+'/'+tail,
        }
        assert tuple(encode(u)) in {tuple(encode(v)) for v in result.values()}
        for v in result.values():
            vr = v[len(prefix):]
            vp = urlsplit('https://'+vr)
            assert vp.query == p.query and vp.fragment == p.fragment
        return result, None
    except (ValueError, UnicodeError): return None, 'parse_error'


def self_test():
    for u in ('HTTPS://www.example.com:443/?x=1#route', 'example.org?', '//www.example.net/#', 'example.com/'):
        v, why = variants(u)
        assert why is None and len(v)==4
        assert len({tuple(encode(x)) for x in v.values()}) == 4
    assert variants('https://example.org/login')[0] is None
    assert variants('http://127.0.0.1/')[0] is None
    assert variants("'example.com'")[0] is None
    print('SELF-TEST PASSED: paired forms, query/fragment preservation and exclusions.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test',action='store_true')
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--all-seeds',action='store_true')
    modes.add_argument('--seed',type=int,choices=range(42,47))
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    import numpy as np
    import pandas as pd
    from collections import Counter
    root = Path(__file__).resolve().parent
    valpath = root/'dataset_fix_outputs_v3'/'validation.csv'
    if sha(valpath) != VAL_HASH: raise ValueError('Frozen validation fingerprint mismatch')
    d = pd.read_csv(valpath,keep_default_na=False)
    if len(d)!=7668 or not d.label.isin([0,1]).all(): raise ValueError('Invalid validation rows')
    saved = {}
    for seed in range(42,47):
        run = root/'corrected_results_v3'/f'seed_{seed}'
        f = pd.read_csv(run/'validation_predictions.csv',keep_default_na=False)
        if d.url.tolist()!=f.url.tolist() or d.label.tolist()!=f.true_label.tolist():
            raise ValueError(f'Seed {seed}: validation alignment differs')
        if not (run/'phishguard_gh_model.keras').is_file(): raise FileNotFoundError(run)
        saved[seed] = f
    selected=[]; rows=[]; reasons=Counter()
    for i,row in d.iterrows():
        v,why=variants(row.url)
        if v is None:
            reasons[why]+=1
            continue
        selected.append(i)
        rows.append(v)
    if not selected: raise ValueError('No eligible root URLs')
    print(f'CHECK PASSED: {len(d):,} validation URLs; {len(rows):,} eligible paired roots; all five saved runs aligned.')
    print('Eligible source labels:',dict(Counter(int(d.label.iloc[i]) for i in selected)))
    print('Exclusions:',dict(reasons))
    print(LIMITATION)
    if args.check_only:
        print('No inference, training or output directories created.')
        return
    import keras
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
    seeds = list(range(42,47)) if args.all_seeds else [args.seed]
    output=root/'formatting_diagnosis_v3'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report={'validation_sha256':VAL_HASH,'eligible_rows':len(rows),'excluded_counts':dict(reasons),
        'threshold':.5,'source':'original frozen validation set only','limitation':LIMITATION,'seeds':seeds,'model_sha256':{}}
    x=np.asarray([encode(u) for u in d.url],dtype=np.int32)
    xv=np.asarray([encode(row[name]) for row in rows for name in NAMES],dtype=np.int32)
    details=d.iloc[selected][['url','label']].copy().reset_index().rename(columns={'index':'validation_index'})
    for name in NAMES: details[name+'_url']=[v[name] for v in rows]
    contrasts=[];conditions=[];original_metrics=[]
    for seed in seeds:
        print(f'Seed {seed}: checking original validation probabilities...',flush=True)
        modelpath=root/'corrected_results_v3'/f'seed_{seed}'/'phishguard_gh_model.keras'
        model=keras.models.load_model(modelpath,compile=False)
        base=model.predict(x,batch_size=256,verbose=0).reshape(-1)
        if not np.allclose(base,saved[seed].probability.to_numpy(),atol=1e-5,rtol=1e-5):
            raise ValueError(f'Seed {seed}: loaded-model prediction mismatch; output is incomplete')
        y=d.label.to_numpy();q=base>=.5
        original_metrics.append({'seed':seed,'accuracy':float(accuracy_score(y,q)),
          'precision':float(precision_score(y,q)),'recall':float(recall_score(y,q)),
          'f1':float(f1_score(y,q)),'auc':float(roc_auc_score(y,base))})
        print(f'Seed {seed}: predicting four forms per eligible root...',flush=True)
        ps=model.predict(xv,batch_size=256,verbose=0).reshape(len(rows),4)
        per=details.copy();per['original_probability']=base[selected]
        for col,name in enumerate(NAMES): per[name+'_probability']=ps[:,col]
        per.to_csv(output/f'seed_{seed}_paired_predictions.csv',index=False)
        for group,mask in [('all',np.ones(len(rows),dtype=bool)),('source_benign',details.label.to_numpy()==0),('source_malicious',details.label.to_numpy()==1)]:
            if not mask.any():continue
            p=ps[mask]
            for j,name in enumerate(NAMES):
                conditions.append({'seed':seed,'source_group':group,'form':name,'rows':len(p),
                  'mean_probability':float(p[:,j].mean()),'positive_prediction_rate':float((p[:,j]>=.5).mean())})
            for name,left,right in [('www_at_empty',0,1),('www_at_slash',2,3),('slash_without_www',0,2),('slash_with_www',1,3)]:
                delta=p[:,right]-p[:,left]
                contrasts.append({'seed':seed,'source_group':group,'contrast':name,'rows':len(p),
                 'mean_probability_change':float(delta.mean()),'mean_absolute_probability_change':float(abs(delta).mean()),
                 'verdict_flip_rate':float(((p[:,right]>=.5)!=(p[:,left]>=.5)).mean()),
                 'mean_interaction_probability':float((p[:,3]-p[:,2]-p[:,1]+p[:,0]).mean())})
        report['model_sha256'][str(seed)]=sha(modelpath)
        del model
        keras.backend.clear_session()
    pd.DataFrame(contrasts).to_csv(output/'paired_effects.csv',index=False)
    pd.DataFrame(conditions).to_csv(output/'form_summary.csv',index=False)
    pd.DataFrame(original_metrics).to_csv(output/'original_validation_metrics.csv',index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'COMPLETE.txt').write_text('All requested seeds completed.\n',encoding='utf-8')
    print(pd.DataFrame(contrasts).query("source_group == 'all'").to_string(index=False))
    print('COMPLETE:',output)
    print('No training, network access or changes to existing data/models/results.')

if __name__=='__main__':main()
