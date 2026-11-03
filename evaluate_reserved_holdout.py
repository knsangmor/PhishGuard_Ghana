"""Evaluate all ten frozen models on the reserved Mendeley holdout.
No training, threshold adjustment, seed selection or internal-test scoring.
Requires train_coverage_pilot.py and freeze_coverage_runs.py beside this script.
"""
import argparse,json,os,time,hashlib
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pandas as pd
from scipy.stats import t
import train_coverage_pilot as pilot
from freeze_coverage_runs import sha,scores
ROOT=Path(__file__).resolve().parent
SEEDS=[42,43,44,45,46]
OPS=['default_0.5','fpr_cap_0.01','fpr_cap_0.005']


def original_encode(url):
    text=str(url).strip()
    for prefix in ('https://','http://'):
        if text.lower().startswith(prefix):text=text[len(prefix):];break
    seq=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in text.lower().strip()[:200]]
    return seq+[0]*(200-len(seq))


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:
        assert original_encode('https://example.com')==original_encode('http://example.com')
        assert original_encode('example.com')!=original_encode('example.com/')
        assert pilot.encode('example.com')==pilot.encode('example.com/')
        r=scores(np.array([0,0,1,1]),np.array([.1,.6,.7,.9]),.5)
        assert (r['tn'],r['fp'],r['fn'],r['tp'])==(1,1,0,2)
        print('SELF-TEST PASSED: family-specific encoders and metric arithmetic.');return
    protocol_paths={'original':ROOT/'original_coverage_frozen_v1'/'frozen_protocol.json',
                    'broader_data':ROOT/'coverage_frozen_v1'/'frozen_protocol.json'}
    protocols={};phashes={}
    for family,path in protocol_paths.items():
        phashes[family]=sha(path)
        if phashes[family]!=(path.parent/'frozen_protocol.sha256').read_text().strip():raise ValueError('Protocol hash mismatch: '+family)
        protocols[family]=json.loads(path.read_text())
        d=protocols[family]
        if d['seeds']!=SEEDS or set(d['runs'])!=set(map(str,SEEDS)):raise ValueError('Missing/unexpected seeds')
        if d['reserved_holdout_sha256']!=pilot.HOLDOUT_SHA or d['reserved_holdout_scored']:raise ValueError('Holdout protocol mismatch')
    if protocols['original']['broader_protocol_sha256']!=phashes['broader_data']:raise ValueError('Protocols are not linked')
    holdout=ROOT/'reserved_mendeley_holdout'/'holdout.csv'
    if sha(holdout)!=pilot.HOLDOUT_SHA:raise ValueError('Reserved holdout changed')
    jobs=[]
    for family,d in protocols.items():
        for seed in SEEDS:
            r=d['runs'][str(seed)];model=ROOT/r['model_path']
            if sha(model)!=r['model_sha256']:raise ValueError(f'Model changed: {family}/{seed}')
            if set(r['thresholds'])!=set(OPS):raise ValueError('Unexpected operating points')
            if r['thresholds']['default_0.5']!=.5:raise ValueError('Default threshold changed')
            if any(not np.isfinite(v) or v<0 for v in r['thresholds'].values()):raise ValueError('Invalid threshold')
            jobs.append((family,seed,model,r))
    # Validate only identity during preflight; no holdout prediction or metric.
    print('CHECK PASSED: linked protocols, ten model fingerprints, frozen thresholds and holdout fingerprint.')
    if a.check_only:print('No inference, holdout metrics or output directories created.');return
    frame=pd.read_csv(holdout,keep_default_na=False)
    if not {'url','label'}.issubset(frame.columns):raise ValueError('Holdout must have url and project label columns')
    if len(frame)!=7659 or not frame.label.isin([0,1]).all() or frame.label.value_counts().to_dict()!={0:3646,1:4013}:raise ValueError('Unexpected holdout counts/label mapping')
    y=frame.label.to_numpy(dtype=int)
    xs={family:np.array([encoder(u) for u in frame.url],dtype=np.float32) for family,encoder in [('original',original_encode),('broader_data',pilot.encode)]}
    commitment={'protocol_sha256':phashes,'holdout_sha256':pilot.HOLDOUT_SHA,
        'script_sha256':sha(Path(__file__)),'seeds':SEEDS,'primary_operating_point':'fpr_cap_0.01',
        'policy':'All ten models, frozen family-specific encoders and validation-selected thresholds; report all seeds. No selection or tuning on holdout.',
        'models':{f'{f}_{s}':{'sha256':r['model_sha256'],'thresholds':r['thresholds']} for f,s,_,r in jobs}}
    out=ROOT/'reserved_holdout_results_v1'
    if out.exists():
        if not (out/'evaluation_manifest.json').is_file() or json.loads((out/'evaluation_manifest.json').read_text())!=commitment:raise ValueError('Existing evaluation has a different manifest; do not overwrite')
        if (out/'COMPLETE.txt').exists():print('Evaluation already complete:',out);return
        print('Resuming identical evaluation; verified completed predictions will be reused.')
    else:
        out.mkdir();write_json(out/'evaluation_manifest.json',commitment)
    import keras
    rows=[];prediction_hashes={}
    for family,seed,model_path,r in jobs:
        key=f'{family}_seed_{seed}';cache=out/f'{key}_predictions.csv';cachemeta=out/f'{key}_cache.json'
        if cache.is_file() and cachemeta.is_file():
            meta=json.loads(cachemeta.read_text())
            if meta['sha256']!=sha(cache) or meta['model_sha256']!=r['model_sha256']:raise ValueError('Prediction cache changed')
            pred=pd.read_csv(cache,keep_default_na=False)
            if pred.url.tolist()!=frame.url.tolist() or not np.array_equal(pred.true_label,y):raise ValueError('Prediction cache alignment differs')
            p=pred.probability.to_numpy(dtype=np.float64);print('Reusing verified predictions:',key)
        else:
            keras.backend.clear_session();print(f'Predicting {key}: {len(y):,} holdout URLs...',flush=True)
            model=keras.models.load_model(model_path,compile=False)
            p=model.predict(xs[family],batch_size=256,verbose=1).reshape(-1).astype(np.float64)
            if len(p)!=len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid predictions')
            pd.DataFrame({'url':frame.url,'true_label':y,'probability':p}).to_csv(cache,index=False)
            write_json(cachemeta,{'sha256':sha(cache),'model_sha256':r['model_sha256']});del model
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid cached predictions')
        prediction_hashes[key]=sha(cache)
        for op,threshold in r['thresholds'].items():
            result=scores(y,p,threshold)
            # FPR is measured, never forced or tuned to the validation cap.
            rows.append(dict(family=family,seed=seed,operating_point=op,threshold=threshold,**result))
    table=pd.DataFrame(rows);summary=[]
    metrics=['accuracy','precision','recall','f1','auc_roc','average_precision','false_positive_rate']
    for (family,op),g in table.groupby(['family','operating_point'],sort=False):
        for metric in metrics:
            mean=float(g[metric].mean());sd=float(g[metric].std(ddof=1));half=float(t.ppf(.975,4)*sd/np.sqrt(5))
            summary.append(dict(family=family,operating_point=op,metric=metric,mean=mean,sample_sd=sd,seed_mean_ci95_low=mean-half,seed_mean_ci95_high=mean+half))
    report={**commitment,'holdout_rows':len(y),'benign':3646,'phishing':4013,'prediction_sha256':prediction_hashes,'summary':summary,
        'limitations':['One separately published historical holdout; upstream independence not proven.','Fixed filtered holdout; domain filtering changes its distribution.','Training positives include malware and defacement; holdout positives are phishing.','Seed confidence intervals are conditional on this fixed dataset, not population or domain sampling uncertainty.','After this evaluation, the holdout is examined and must not be reused as untouched confirmation for further tuned models.','Validation FPR caps do not guarantee holdout or deployment FPR.']}
    table.to_csv(out/'holdout_per_seed.csv',index=False);pd.DataFrame(summary).to_csv(out/'holdout_summary.csv',index=False)
    write_json(out/'report.json',report)
    (out/'COMPLETE.txt').write_text('All ten frozen models evaluated; no training or holdout tuning.\n')
    print('\nHOLDOUT MEANS ACROSS FIVE SEEDS (proportions)')
    print(table.groupby(['family','operating_point'])[['accuracy','recall','f1','auc_roc','false_positive_rate']].mean().to_string(float_format='%.6f'))
    print('Saved:',out)
    print('No training or threshold changes. Report all results, including negative findings.')

if __name__=='__main__':main()
