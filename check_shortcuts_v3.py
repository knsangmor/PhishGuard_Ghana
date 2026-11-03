"""Saved-model shortcut diagnostics. Perturbations are sensitivity probes, not verified label-preserving URLs."""
import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
import ipaddress
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
    root=Path(__file__).resolve().parent
    run=root/'corrected_results_v3'/f'seed_{args.seed}'
    test_path=root/'dataset_fix_outputs_v3'/'test.csv'
    metadata=json.loads((run/'metrics.json').read_text())
    if sha(test_path)!=metadata['partitions']['test']['sha256']:raise ValueError('Test fingerprint mismatch')
    frame=pd.read_csv(test_path,keep_default_na=False)
    saved=pd.read_csv(run/'test_predictions.csv',keep_default_na=False)
    if not frame[['url','label']].equals(saved[['url','true_label']].rename(columns={'true_label':'label'})):raise ValueError('Saved prediction alignment failure')
    spec=importlib.util.spec_from_file_location('shortcut_pipeline',root/'model_hash_v3.py')
    pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
    original=frame.url.tolist()
    x=np.array([pipeline.url_to_seq(u) for u in original],dtype=np.int32)
    def bare(u):
        text=u.strip()
        for prefix in ['https://','http://']:
            if text.lower().startswith(prefix):return text[len(prefix):]
        return text
    scheme_results={}
    for variant,prefix in [('schemeless',''),('http','http://'),('https','https://')]:
        variant_x=np.array([pipeline.url_to_seq(prefix+bare(u)) for u in original],dtype=np.int32)
        mismatches=int(np.any(x!=variant_x,axis=1).sum())
        scheme_results[variant]={'urls':len(frame),'encoding_mismatches':mismatches}
        print(f'{variant}: {mismatches} encoded-input mismatches out of {len(frame):,}')
    if any(v['encoding_mismatches'] for v in scheme_results.values()):raise ValueError('Scheme invariance failed; inspect before inference')
    print('CHECK PASSED: test alignment and exact scheme encoding invariance.')
    if args.check_only:
        print('No inference, training or output directories created.');return
    output=root/'shortcut_checks_v3'/f"seed_{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}"
    output.mkdir(parents=True,exist_ok=False)
    import keras
    model=keras.models.load_model(run/'phishguard_gh_model.keras',compile=False)
    print('Predicting original test URLs...',flush=True)
    base=model.predict(x,batch_size=256,verbose=0).reshape(-1)
    labels=frame.label.to_numpy();base_pred=base>=.5
    length=np.array([len(bare(u).lower().strip()) for u in original])
    length_rows=[]
    for lo,hi,name in [(0,50,'0-49'),(50,100,'50-99'),(100,201,'100-200'),(201,10**12,'over_200_truncated')]:
        mask=(length>=lo)&(length<hi);y=labels[mask];p=base[mask];q=p>=.5
        if not len(y):continue
        benign=y==0;malicious=y==1
        length_rows.append({'length_group':name,'rows':len(y),'benign':int(benign.sum()),'malicious':int(malicious.sum()),
            'accuracy':float(accuracy_score(y,q)),'precision':float(precision_score(y,q,zero_division=0)),
            'recall':float(recall_score(y,q,zero_division=0)) if malicious.any() else None,
            'f1':float(f1_score(y,q,zero_division=0)),
            'fpr':float(q[benign].mean()) if benign.any() else None,
            'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None})
    pd.DataFrame(length_rows).to_csv(output/'original_performance_by_length.csv',index=False)
    def perturb(url,kind):
        text=bare(url)
        if text.startswith('//'):text=text[2:]
        try:
            p=urlsplit('https://'+text)
            host=p.hostname
            if not host:return None
            if kind=='toggle_www':
                if p.username is not None or p.password is not None:return None
                try:ipaddress.ip_address(host);return None
                except ValueError:pass
                # Preserve netloc spelling and any explicit port.
                netloc=p.netloc[4:] if p.netloc.lower().startswith('www.') else 'www.'+p.netloc
                return urlunsplit(('',netloc,p.path,p.query,p.fragment))[2:]
            if kind=='toggle_root_slash':
                if p.path not in ['','/']:return None
                path='/' if p.path=='' else ''
            elif kind=='append_path_256_chars':
                path=p.path.rstrip('/')+'/'+'a'*256
            else:raise ValueError(kind)
            return urlunsplit(('',p.netloc,path,p.query,p.fragment))[2:]
        except ValueError:return None
    summaries=[]
    for kind in ['toggle_www','toggle_root_slash','append_path_256_chars']:
        pairs=[(i,perturb(u,kind)) for i,u in enumerate(original)]
        pairs=[(i,u) for i,u in pairs if u is not None]
        indices=np.array([i for i,_ in pairs],dtype=int)
        xp=np.array([pipeline.url_to_seq(u) for _,u in pairs],dtype=np.int32)
        if not len(pairs):continue
        changed=np.any(x[indices]!=xp,axis=1)
        print(f'Predicting {kind}: {len(pairs):,} eligible URLs...',flush=True)
        prob=model.predict(xp,batch_size=256,verbose=0).reshape(-1)
        flip=(prob>=.5)!=base_pred[indices]
        details=pd.DataFrame({'test_row':indices,'url':[original[i] for i in indices],'original_label':labels[indices],
            'perturbed_url':[u for _,u in pairs],'input_encoding_changed':changed,
            'original_probability':base[indices],'perturbed_probability':prob,'verdict_flipped':flip})
        details.to_csv(output/f'{kind}_details.csv',index=False)
        for group,mask in [('all',np.ones(len(pairs),dtype=bool)),('original_benign',labels[indices]==0),('original_malicious',labels[indices]==1)]:
            if not mask.any():continue
            summaries.append({'probe':kind,'original_class_group':group,'eligible_urls':int(mask.sum()),
                'encoding_changed':int(changed[mask].sum()),'verdict_flips':int(flip[mask].sum()),
                'flip_rate':float(flip[mask].mean()),'mean_absolute_probability_change':float(np.abs(prob[mask]-base[indices][mask]).mean()),
                'unchanged_encoding_flips':int((flip&~changed&mask).sum())})
    pd.DataFrame(summaries).to_csv(output/'perturbation_summary.csv',index=False)
    report={'seed':args.seed,'test_records':len(frame),'threshold':.5,'scheme_encoding_checks':scheme_results,
        'max_original_probability_difference_from_saved':float(np.max(np.abs(base-saved.probability.to_numpy()))),
        'original_verdict_agreement_with_saved':float(np.mean(base_pred==saved.pred_label.to_numpy())),
        'test_sha256':sha(test_path),'model_sha256':sha(run/'phishguard_gh_model.keras'),
        'interpretation':'Identical encoded inputs establish scheme invariance for this deterministic encoder/model in inference mode. They do not demonstrate improved generalization or the benefit of augmentation. www and slash changes can change the resource; appended paths are synthetic. Their original labels are retained only to stratify probes, not assigned as ground truth to transformed URLs. Do not call these flip rates attack success rates or transformed accuracy.',
        'remaining':'Before/after fix comparison and individual training-fix ablations remain untested. Length-group comparisons are descriptive and confounded by URL source/class/domain. These probes do not constitute validated adaptive-attack evaluation.'}
    (output/'shortcut_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nSHORTCUT CHECKS COMPLETE. No retraining or source-file changes.')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Saved to:',output)
if __name__=='__main__':main()
