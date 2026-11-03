"""Summarize five fixed coverage runs and freeze validation-only thresholds.
No TensorFlow, inference, training, internal-test scoring or holdout scoring.
Requires train_coverage_pilot.py beside this script for reviewed fingerprints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import train_coverage_pilot as pilot
ROOT=Path(__file__).resolve().parent
SEEDS=[42,43,44,45,46]


def scores(y,p,threshold):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=threshold,labels=[0,1]).ravel())
    return dict(accuracy=(tn+tp)/len(y),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,false_positive_rate=fp/(fp+tn),auc_roc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),tn=tn,fp=fp,fn=fn,tp=tp)


def cap_threshold(y,p,cap):
    negatives=np.sort(p[y==0])[::-1]
    allowed=int(np.floor(cap*len(negatives)))
    return float(np.nextafter(negatives[allowed],np.inf))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--freeze',action='store_true')
    mode.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        y=np.array([0,0,0,0,1,1]);p=np.array([.8,.8,.3,.1,.9,.7])
        threshold=cap_threshold(y,p,.25)
        assert np.sum(p[y==0]>=threshold)<=1
        assert scores(y,p,threshold)['tp']==1
        print('SELF-TEST PASSED: tied-score cap and confusion counts.');return
    prepared=ROOT/'coverage_development_v1'
    references={}
    for name,filename in [('validation_v3','validation_v3.csv'),('validation_phiusiil','validation_phiusiil_development.csv')]:
        path=prepared/filename
        if sha(path)!=pilot.FILES[filename]:raise ValueError('Validation fingerprint changed: '+filename)
        references[name]=pd.read_csv(path,keep_default_na=False)
    locked={'seeds':SEEDS,'prepared_files_sha256':pilot.FILES,'partition_sha256':pilot.PARTS,
        'reserved_holdout_sha256':pilot.HOLDOUT_SHA,
        'threshold_policy':'Per seed, one threshold for both validation sources; maximum of their thresholds meeting the empirical FPR cap. Use probability >= threshold.',
        'operating_points':['default_0.5','fpr_cap_0.01','fpr_cap_0.005'],
        'primary_operating_point':'fpr_cap_0.01',
        'reserved_holdout_scored':False,'internal_test_scored':False,
        'evaluation_commitment':'Evaluate all five frozen broader-data models and all five original v3 models on the same reserved holdout; report all seeds, default 0.5, and validation-selected 1% and 0.5% FPR operating points. Do not select seeds or tune thresholds from holdout outcomes. Original models require thresholds frozen using these same development validation sources before holdout evaluation.',
        'limitations':['Development validation has informed selection.','Empirical FPR caps do not guarantee future FPR.','Five-seed variability is conditional on fixed data; confidence intervals do not capture dataset or domain sampling uncertainty.','PhiUSIIL is development data, not independent confirmation.'],
        'runs':{}}
    rows=[]
    for seed in SEEDS:
        folder=ROOT/'coverage_pilot_v1'/f'seed_{seed}'
        if not (folder/'COMPLETE.txt').is_file():raise ValueError('Run incomplete: '+str(seed))
        metadata=json.loads((folder/'metrics.json').read_text())
        if metadata['seed']!=seed or metadata['training_rows']!=51664 or metadata['prepared_files_sha256']!=pilot.FILES or metadata['partition_sha256']!=pilot.PARTS:
            raise ValueError('Run protocol mismatch: '+str(seed))
        if metadata['reserved_holdout_scored'] or metadata['internal_test_scored']:raise ValueError('Unexpected prior scoring')
        model=folder/'model.keras'
        if sha(model)!=metadata['model_sha256']:raise ValueError('Model changed: '+str(seed))
        predictions={};hashes={}
        for name,ref in references.items():
            path=folder/f'{name}_predictions.csv'
            frame=pd.read_csv(path,keep_default_na=False)
            if frame.url.tolist()!=ref.url.tolist() or not np.array_equal(frame.true_label,ref.label):raise ValueError('Prediction alignment mismatch')
            y=frame.true_label.to_numpy(dtype=int);p=frame.probability.to_numpy(dtype=np.float64)
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
            predictions[name]=(y,p);hashes[name]=sha(path)
            reproduced=scores(y,p,.5)
            for key in ('accuracy','recall','auc_roc','false_positive_rate'):
                if abs(reproduced[key]-metadata['validation'][name][key])>1e-6:raise ValueError('Saved metric mismatch: '+key)
        thresholds={'default_0.5':.5}
        for cap in [.01,.005]:thresholds[f'fpr_cap_{cap}']=max(cap_threshold(y,p,cap) for y,p in predictions.values())
        locked['runs'][str(seed)]={'model_path':model.relative_to(ROOT).as_posix(),'model_sha256':sha(model),'metrics_sha256':sha(folder/'metrics.json'),'prediction_sha256':hashes,'thresholds':thresholds}
        for op,threshold in thresholds.items():
            for name,(y,p) in predictions.items():
                result=scores(y,p,threshold)
                if op.startswith('fpr_cap_') and result['false_positive_rate']>float(op.removeprefix('fpr_cap_'))+1e-12:raise ValueError('FPR cap exceeded')
                rows.append(dict(seed=seed,source=name,operating_point=op,threshold=threshold,**result))
        print(f'Seed {seed}: completed model, validation alignment and thresholds verified.',flush=True)
    table=pd.DataFrame(rows);summary=[]
    for (source,op),group in table.groupby(['source','operating_point'],sort=False):
        for metric in ['accuracy','precision','recall','f1','auc_roc','average_precision','false_positive_rate']:
            values=group[metric].to_numpy();mean=float(values.mean());sd=float(values.std(ddof=1));half=float(t.ppf(.975,4)*sd/np.sqrt(5))
            summary.append(dict(source=source,operating_point=op,metric=metric,mean=mean,sample_sd=sd,seed_mean_ci95_low=mean-half,seed_mean_ci95_high=mean+half))
    print('CHECK PASSED: all five runs; no holdout read or scored.')
    if args.check_only:print('No output directories created.');return
    out=ROOT/'coverage_frozen_v1';out.mkdir(exist_ok=False)
    text=json.dumps(locked,indent=2,allow_nan=False)
    (out/'frozen_protocol.json').write_text(text,encoding='utf-8')
    (out/'frozen_protocol.sha256').write_text(sha(out/'frozen_protocol.json')+'\n',encoding='utf-8')
    table.to_csv(out/'validation_per_seed.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'validation_summary.csv',index=False)
    print('\nDEFAULT 0.5: mean +/- sample SD across five seeds')
    for source in references:
        g=table[(table.source==source)&(table.operating_point=='default_0.5')]
        print(source)
        for metric in ['accuracy','recall','f1','auc_roc','false_positive_rate']:
            scale=1 if metric=='auc_roc' else 100
            print(f'  {metric}: {scale*g[metric].mean():.4f} +/- {scale*g[metric].std(ddof=1):.4f}')
    print('\nFROZEN THRESHOLDS')
    print(table[['seed','operating_point','threshold']].drop_duplicates().to_string(index=False))
    print('Saved:',out)
    print('No training, inference, internal-test or reserved-holdout scoring.')

if __name__=='__main__':main()
