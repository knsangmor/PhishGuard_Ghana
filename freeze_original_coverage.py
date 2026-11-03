"""Freeze original-model thresholds on both development validation sources.
No training or holdout access. Requires freeze_coverage_runs.py and
train_coverage_pilot.py beside this file. Existing outputs never overwritten.
"""
import argparse,json,os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pandas as pd
import train_coverage_pilot as pilot
from freeze_coverage_runs import sha,cap_threshold,scores
ROOT=Path(__file__).resolve().parent


def encode(url):
    text=str(url).strip()
    for prefix in ('https://','http://'):
        if text.lower().startswith(prefix):text=text[len(prefix):];break
    seq=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in text.lower().strip()[:200]]
    return seq+[0]*(200-len(seq))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true');mode.add_argument('--freeze',action='store_true')
    a=ap.parse_args()
    frozen=ROOT/'coverage_frozen_v1'/'frozen_protocol.json'
    expected=(frozen.parent/'frozen_protocol.sha256').read_text().strip()
    if sha(frozen)!=expected:raise ValueError('Broader-data frozen protocol hash mismatch')
    protocol=json.loads(frozen.read_text())
    if protocol['seeds']!=[42,43,44,45,46] or protocol['prepared_files_sha256']!=pilot.FILES:raise ValueError('Protocol differs')
    frames={};xs={};models={}
    for name,filename in [('validation_v3','validation_v3.csv'),('validation_phiusiil','validation_phiusiil_development.csv')]:
        path=ROOT/'coverage_development_v1'/filename
        if sha(path)!=pilot.FILES[filename]:raise ValueError('Validation fingerprint differs')
        frame=pd.read_csv(path,keep_default_na=False)
        if not frame.label.isin([0,1]).all():raise ValueError('Invalid labels')
        frames[name]=frame
        xs[name]=np.array([encode(u) for u in frame.url],dtype=np.float32)
    for seed in protocol['seeds']:
        folder=ROOT/'corrected_results_v3'/f'seed_{seed}'
        model=folder/'phishguard_gh_model.keras'
        if not model.is_file():raise FileNotFoundError(model)
        ref=pd.read_csv(folder/'validation_predictions.csv',keep_default_na=False)
        frame=frames['validation_v3']
        if ref.url.tolist()!=frame.url.tolist() or not np.array_equal(ref.true_label,frame.label):raise ValueError('Original prediction alignment mismatch')
        models[seed]=(model,sha(model))
        print(f'Seed {seed}: original model and saved validation alignment verified.',flush=True)
    for seed,run in protocol['runs'].items():
        if sha(ROOT/run['model_path'])!=run['model_sha256']:raise ValueError('Frozen broader-data model changed')
    print('CHECK PASSED: frozen protocol, all models and validation inputs. No holdout accessed.')
    if a.check_only:print('No inference or output directories created.');return
    import keras
    out=ROOT/'original_coverage_frozen_v1';out.mkdir(exist_ok=False)
    # Bind to all original weights before the first prediction.
    manifest={'broader_protocol_sha256':expected,'seeds':protocol['seeds'],
        'threshold_policy':protocol['threshold_policy'],'primary_operating_point':'fpr_cap_0.01',
        'reserved_holdout_sha256':protocol['reserved_holdout_sha256'],
        'reserved_holdout_scored':False,'internal_test_scored':False,
        'validation_files_sha256':{k:pilot.FILES[k] for k in ['validation_v3.csv','validation_phiusiil_development.csv']},
        'encoding':'Original v3: scheme stripped, lowercase, first 200 characters, right padding; no root normalization.',
        'runs':{str(s):{'model_path':m.relative_to(ROOT).as_posix(),'model_sha256':h} for s,(m,h) in models.items()}}
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[]
    for seed,(path,h) in models.items():
        keras.backend.clear_session();print(f'Loading original seed {seed}...',flush=True)
        model=keras.models.load_model(path,compile=False);preds={};predhash={}
        for name,frame in frames.items():
            print(f'Predicting {name}...',flush=True)
            y=frame.label.to_numpy(dtype=int)
            p=model.predict(xs[name],batch_size=256,verbose=1).reshape(-1).astype(np.float64)
            if len(p)!=len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid predictions')
            if name=='validation_v3':
                ref=pd.read_csv(path.parent/'validation_predictions.csv')
                if not np.allclose(p,ref.probability,rtol=1e-5,atol=1e-5):raise ValueError('Original predictions do not reproduce')
            predfile=out/f'seed_{seed}_{name}_predictions.csv'
            pd.DataFrame({'url':frame.url,'true_label':y,'probability':p,'pred_label':(p>=.5).astype(int)}).to_csv(predfile,index=False)
            preds[name]=(y,p);predhash[name]=sha(predfile)
        thresholds={'default_0.5':.5}
        for cap in [.01,.005]:thresholds[f'fpr_cap_{cap}']=max(cap_threshold(y,p,cap) for y,p in preds.values())
        for op,threshold in thresholds.items():
            for name,(y,p) in preds.items():
                result=scores(y,p,threshold)
                if op.startswith('fpr_cap_') and result['false_positive_rate']>float(op[8:])+1e-12:raise ValueError('FPR cap exceeded')
                rows.append(dict(seed=seed,source=name,operating_point=op,threshold=threshold,**result))
        manifest['runs'][str(seed)].update(thresholds=thresholds,prediction_sha256=predhash)
        del model
    pd.DataFrame(rows).to_csv(out/'validation_per_seed.csv',index=False)
    f=out/'frozen_protocol.json';f.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    (out/'frozen_protocol.sha256').write_text(sha(f)+'\n',encoding='utf-8')
    (out/'COMPLETE.txt').write_text('All five original models and validation-only thresholds frozen.\n')
    print('\nFROZEN ORIGINAL-MODEL THRESHOLDS')
    print(pd.DataFrame(rows)[['seed','operating_point','threshold']].drop_duplicates().to_string(index=False))
    print('Saved:',out)
    print('No training, internal-test or reserved-holdout scoring.')

if __name__=='__main__':main()
