"""Compare the saved root-normalized seed-42 pilot on already-examined PhiUSIIL.
DIAGNOSTIC ONLY: not a fresh external test. No training or reserved-holdout scoring.
Requires the previously supplied train_root_normalized_pilot.py encoder unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
EXTERNAL_SHA='c3f6b91e1d20b96aee4bd860c769525683d0adee736e00a2b40e8e33e5744a88'
VAL_SHA='8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 modes=ap.add_mutually_exclusive_group(required=True)
 modes.add_argument('--check-only',action='store_true');modes.add_argument('--run',action='store_true')
 a=ap.parse_args()
 import numpy as np,pandas as pd
 root=Path(__file__).resolve().parent
 pilot=root/'root_normalized_pilot_v3'/'seed_42'
 metadata=json.loads((pilot/'metrics.json').read_text())
 protocol=json.loads((pilot/'protocol.json').read_text())
 model_path=pilot/'model.keras'
 if metadata['seed']!=42 or metadata['experiment']!='root_normalized_validation_pilot':raise ValueError('Unexpected pilot identity')
 if sha(model_path)!=metadata['model_sha256']:raise ValueError('Pilot model changed')
 encoder_path=root/'train_root_normalized_pilot.py'
 if sha(encoder_path)!=protocol['script_sha256']:raise ValueError('Pilot encoder script changed')
 spec=importlib.util.spec_from_file_location('saved_root_pilot_encoder',encoder_path)
 encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
 prepared=root/'external_phiusiil_v3'
 audit=json.loads((prepared/'audit_report.json').read_text())
 if audit['external_test_sha256']!=EXTERNAL_SHA or sha(prepared/'external_test.csv')!=EXTERNAL_SHA:
  raise ValueError('Previously examined PhiUSIIL holdout changed')
 d=pd.read_csv(prepared/'external_test.csv',keep_default_na=False)
 original=pd.read_csv(prepared/'seed_42'/'predictions.csv',keep_default_na=False)
 if not original[['source_row','url','label']].equals(d[['source_row','url','label']]):raise ValueError('Original external-prediction alignment mismatch')
 if len(d)!=224879 or int(d.label.eq(0).sum())!=131488 or int(d.label.eq(1).sum())!=93391:raise ValueError('External class counts differ')
 val_path=root/'dataset_fix_outputs_v3'/'validation.csv'
 if sha(val_path)!=VAL_SHA:raise ValueError('Validation identity differs')
 val=pd.read_csv(val_path,keep_default_na=False)
 saved_val=pd.read_csv(pilot/'validation_predictions.csv',keep_default_na=False)
 if val.url.tolist()!=saved_val.url.tolist() or val.label.tolist()!=saved_val.true_label.tolist():raise ValueError('Pilot validation alignment differs')
 print('CHECK PASSED: saved pilot/encoder, original validation, and previously examined PhiUSIIL snapshot.')
 print('No Mendeley holdout files are read or scored by this script.')
 if a.check_only:
  print('No inference, training or output folders created.');return
 out=pilot/'phiusiil_diagnostic'
 if out.exists():raise FileExistsError(f'Existing diagnostic results preserved: {out}')
 import keras
 from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
 model=keras.models.load_model(model_path,compile=False)
 vx=np.asarray([encoder.encode(u) for u in val.url],dtype=np.float32)
 vp=model.predict(vx,batch_size=256,verbose=0).reshape(-1)
 if not np.allclose(vp,saved_val.probability.to_numpy(),atol=1e-5,rtol=1e-5):raise ValueError('Pilot validation probabilities do not reproduce')
 chunks=[]
 for start in range(0,len(d),4096):
  x=np.asarray([encoder.encode(u) for u in d.url.iloc[start:start+4096]],dtype=np.float32)
  chunks.append(model.predict(x,batch_size=256,verbose=0).reshape(-1))
  print(f'Diagnostic URLs: {min(start+4096,len(d)):,}/{len(d):,}',flush=True)
 p=np.concatenate(chunks);y=d.label.to_numpy();old=original.probability.to_numpy()
 def metrics(prob,threshold=.5):
  q=prob>=threshold;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
  return {'accuracy':float(accuracy_score(y,q)),'precision':float(precision_score(y,q,zero_division=0)),
   'recall':float(recall_score(y,q)),'f1_score':float(f1_score(y,q,zero_division=0)),
   'auc_roc':float(roc_auc_score(y,prob)),'average_precision':float(average_precision_score(y,prob)),
   'false_positive_rate':float(fp/(tn+fp)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
 corrected=metrics(p);reference=metrics(old)
 operating={}
 for cap,v in metadata['validation_selected_operating_points'].items():
  t=float('inf') if v['positive_infinity'] else v['threshold']
  operating[cap]={'threshold':v['threshold'],'positive_infinity':v['positive_infinity'],
   'selected_on':'original v3 validation set, corrected pilot scores','external_metrics':metrics(p,t)}
 parsed=d.url.map(urlsplit)
 root_path=parsed.map(lambda u:u.path in ('','/')).to_numpy()
 has_www=parsed.map(lambda u:(u.hostname or '').lower().startswith('www.')).to_numpy()
 groups=[]
 for label in [0,1]:
  for name,mask in [('all',np.ones(len(d),dtype=bool)),('root',root_path),('nonroot',~root_path),('www',has_www),('no_www',~has_www)]:
   ix=(y==label)&mask
   if not ix.any():continue
   groups.append({'source_label':label,'group':name,'rows':int(ix.sum()),
    'original_error_rate':float(((old[ix]>=.5)!=y[ix]).mean()),
    'corrected_error_rate':float(((p[ix]>=.5)!=y[ix]).mean())})
 report={'seed':42,'purpose':'development diagnosis on already examined PhiUSIIL; not independent confirmation',
  'external_test_sha256':EXTERNAL_SHA,'model_sha256':metadata['model_sha256'],
  'original_at_0_5':reference,'corrected_at_0_5':corrected,
  'differences':{k:corrected[k]-reference[k] for k in ['accuracy','precision','recall','f1_score','auc_roc','average_precision','false_positive_rate']},
  'validation_selected_operating_points':operating,'reserved_holdout_scored':False,'training_performed':False,
  'note':'Do not choose thresholds using these external scores. Mendeley remains the reserved final comparison.'}
 out.mkdir()
 result=d.copy();result['original_probability']=old;result['corrected_probability']=p
 result.to_csv(out/'predictions.csv',index=False)
 pd.DataFrame(groups).to_csv(out/'group_comparison.csv',index=False)
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(report,indent=2,allow_nan=False));print('DIAGNOSTIC COMPLETE:',out)

if __name__=='__main__':main()
