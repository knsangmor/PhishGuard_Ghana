"""Separate exploratory masking control. Frozen inputs and original results are read only."""
import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import zipfile
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
HASHES={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
SIZES={'train':35664,'validation':7668,'test':7653}
SOURCE_SHA='d006f9ef69906689c39b3b8a85fc37fc4f181a65b28637cb7b6681900aeda2bc'
SEEDS=range(42,47)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def save(path,data):
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def encode(u):
 s=str(u).strip()
 for prefix in ('https://','http://'):
  if s.lower().startswith(prefix):s=s[len(prefix):];break
 t=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in s.lower().strip()[:200]]
 return t+[0]*(200-len(t))

def self_test():
 assert encode(' HTTPS://EXAMPLE.COM ')==encode('example.com')
 assert encode('a')==[66]+[0]*199
 assert encode('a'*201)==[66]*200
 assert encode('\u2603')[0]==96
 x=np.array([encode('a'),encode('abc')]);lens=(x!=0).sum(axis=1)
 assert np.array_equal(x==0,np.arange(200)[None,:]>=lens[:,None])
 print('SELF-TEST PASSED: scheme stripping, encoding, truncation and right padding. No TensorFlow or training.')

def preflight(root):
 base=root/'baseline_results_v3';protocol=json.loads((base/'protocol.json').read_text())
 if sha(root/'baseline_comparison_v3.py')!=SOURCE_SHA or protocol['script_sha256']!=SOURCE_SHA:raise ValueError('Original baseline source/protocol mismatch')
 selected=json.loads((base/'selected_configs.json').read_text())['BiLSTM']
 if selected['config']!={'units':128}:raise ValueError('Expected original selected 128-unit BiLSTM')
 frames={};refs={};manifest={'script_sha256':sha(__file__),'partition_sha256':HASHES,'source_sha256':SOURCE_SHA,'baseline_protocol_sha256':sha(base/'protocol.json'),'selected_configs_sha256':sha(base/'selected_configs.json'),'reference_files':{},'control':'mask_zero=True; same selected 128 units, 64-dimensional embedding, Adam .001, batch256, max50 epochs, val_loss patience12, ReduceLR patience4 factor.5 min1e-5','selection':'best validation loss within each run; no new architecture tuning; threshold .5','scope':'Exploratory five-seed control on previously examined fixed partitions; not new independent confirmation.'}
 for n,h in HASHES.items():
  p=root/'dataset_fix_outputs_v3'/f'{n}.csv'
  if sha(p)!=h:raise ValueError('Frozen partition hash differs: '+n)
  d=pd.read_csv(p,keep_default_na=False)
  if len(d)!=SIZES[n] or not d.label.isin([0,1]).all() or d.url.duplicated().any():raise ValueError('Invalid partition '+n)
  frames[n]=d
 for seed in SEEDS:
  folder=base/f'seed_{seed}'/'BiLSTM';model=folder/'model.keras'
  with zipfile.ZipFile(model) as z:c=json.loads(z.read('config.json'))
  layers=c['config']['layers'];e=next(v['config'] for v in layers if v['class_name']=='Embedding');b=next(v['config'] for v in layers if v['class_name']=='Bidirectional')
  if e['mask_zero'] or e['input_dim']!=97 or e['output_dim']!=64 or b['layer']['config']['units']!=128:raise ValueError('Reference architecture differs')
  for n in ('validation','test'):
   p=folder/f'{n}_predictions.csv';d=pd.read_csv(p,keep_default_na=False)
   if not d[['url','true_label']].rename(columns={'true_label':'label'}).equals(frames[n][['url','label']]):raise ValueError('Reference alignment differs')
   probs=d.probability.to_numpy(float)
   if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any() or not np.array_equal(d.pred_label.to_numpy(),probs>=.5):raise ValueError('Reference scores/verdicts invalid')
   refs[seed,n]=d;manifest['reference_files'][str(p.relative_to(root))]=sha(p)
  manifest['reference_files'][str(model.relative_to(root))]=sha(model)
 print('INPUT CHECK PASSED: frozen partitions, original baseline protocol, five unmasked models and aligned saved predictions.')
 return frames,refs,manifest

def build(seed,masked):
 import keras
 from keras import layers
 keras.backend.clear_session();keras.utils.set_random_seed(seed)
 inp=keras.Input((200,),dtype='int32');e=layers.Embedding(97,64,mask_zero=masked);r=layers.Bidirectional(layers.LSTM(128))
 h=r(e(inp));h=layers.Dropout(.4)(h);h=layers.Dense(64,activation='relu')(h);out=layers.Dense(1,activation='sigmoid')(h)
 m=keras.Model(inp,out);m.compile(optimizer=keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=['accuracy'])
 return m,e,r

def runtime_check():
 import keras,tensorflow as tf
 # Identical initial parameter values isolate the sole architecture setting changed.
 a,_,_=build(42,False);weights=[w.copy() for w in a.get_weights()]
 m,e,r=build(42,True)
 if not all(np.array_equal(x,y) for x,y in zip(weights,m.get_weights())):raise ValueError('Initial weights differ between masking settings')
 for length in (1,8,43,199):
  seq=np.zeros((1,200),dtype='int32');seq[0,:length]=1+np.arange(length)%95
  emb=e(seq);mask=e.compute_mask(seq)
  if not np.array_equal(np.asarray(mask),seq!=0):raise ValueError('Mask values differ')
  full=np.asarray(r(emb,mask=mask,training=False));prefix=np.asarray(r(emb[:,:length],mask=np.ones((1,length),dtype=bool),training=False))
  if not np.allclose(full,prefix,rtol=0,atol=2e-5):raise ValueError('Masked output differs from unpadded prefix')
 print('RUNTIME CHECK PASSED: identical initial weights and verified recurrent padding-mask behavior.')
 keras.backend.clear_session()
 return {'tensorflow':tf.__version__,'keras':keras.__version__,'numpy':np.__version__,'pandas':pd.__version__,'devices':[d.name for d in tf.config.list_physical_devices()]}

def complete(out,seed,manifest):
 good=[]
 for p in sorted(out.glob(f'seed_{seed}_*/training_complete.json')):
  d=json.loads(p.read_text())
  if d['manifest']!=manifest:raise ValueError('Completed control has a different protocol; preserve it and use a new --output')
  if sha(p.parent/'model.keras')!=d['model_sha256'] or sha(p.parent/'validation_predictions.csv')!=d['validation_predictions_sha256']:raise ValueError('Completed artifact hash differs')
  good.append((p.parent,d))
 if len(good)>1:raise ValueError('Multiple completed runs; review rather than choose by score')
 return good[0] if good else None

def metrics(y,p):
 from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
 q=p>=.5
 return {k:float(v) for k,v in {'accuracy':accuracy_score(y,q),'precision':precision_score(y,q,zero_division=0),'recall':recall_score(y,q),'f1':f1_score(y,q),'auc_roc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'false_positive_rate':q[y==0].mean()}.items()}

def train(out,seed,frames,refs,manifest,env):
 import keras,time
 old=complete(out,seed,manifest)
 if old:print('Reusing completed masked seed',seed,old[0]);return
 folder=out/f'seed_{seed}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}';folder.mkdir(parents=True,exist_ok=False)
 save(folder/'protocol.json',{'manifest':manifest,'environment':env,'seed':seed,'test_scored':False})
 # An interrupted attempt remains preserved; a new invocation starts a new attempt.
 x=np.array([encode(u) for u in frames['train'].url],dtype=np.int32);vx=np.array([encode(u) for u in frames['validation'].url],dtype=np.int32)
 model,_,_=build(seed,True);start=time.perf_counter()
 hist=model.fit(x,frames['train'].label.to_numpy(),validation_data=(vx,frames['validation'].label.to_numpy()),epochs=50,batch_size=256,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True,verbose=1),keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,factor=.5,min_lr=1e-5,verbose=1),keras.callbacks.CSVLogger(str(folder/'epoch_history.csv'))],verbose=1)
 elapsed=time.perf_counter()-start;model.save(folder/'model.keras')
 p=model.predict(vx,verbose=0).reshape(-1);d=frames['validation'];pd.DataFrame({'url':d.url,'true_label':d.label,'probability':p,'pred_label':p>=.5}).to_csv(folder/'validation_predictions.csv',index=False)
 save(folder/'history.json',{k:[float(v) for v in vals] for k,vals in hist.history.items()})
 result={'manifest':manifest,'environment':env,'seed':seed,'training_seconds':elapsed,'epochs':len(hist.history['loss']),'best_epoch':int(np.argmin(hist.history['val_loss'])+1),'validation_metrics':metrics(d.label.to_numpy(),p),'unmasked_saved_validation_metrics':metrics(d.label.to_numpy(),refs[seed,'validation'].probability.to_numpy()),'model_sha256':sha(folder/'model.keras'),'validation_predictions_sha256':sha(folder/'validation_predictions.csv'),'test_scored':False}
 save(folder/'training_complete.json',result);print(json.dumps(result,indent=2));print('COMPLETE:',folder,'; no test inference performed.')

def evaluate(out,frames,refs,manifest):
 import keras
 runs={s:complete(out,s,manifest) for s in SEEDS}
 if not all(runs.values()):raise ValueError('All five masked runs must complete before test evaluation')
 folder=out/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));folder.mkdir(parents=True,exist_ok=False)
 x=np.array([encode(u) for u in frames['test'].url],dtype=np.int32);d=frames['test'];rows=[]
 for seed,(run,report) in runs.items():
  model=keras.models.load_model(run/'model.keras',compile=False);p=model.predict(x,batch_size=256,verbose=0).reshape(-1)
  pd.DataFrame({'url':d.url,'true_label':d.label,'probability':p,'pred_label':p>=.5}).to_csv(folder/f'seed_{seed}_test_predictions.csv',index=False)
  for name,vals in [('masked',p),('unmasked_saved',refs[seed,'test'].probability.to_numpy())]:rows.append({'seed':seed,'model':name,**metrics(d.label.to_numpy(),vals)})
  keras.backend.clear_session()
 table=pd.DataFrame(rows);table.to_csv(folder/'per_seed.csv',index=False)
 summary=table.groupby('model')[[c for c in table if c not in ('seed','model')]].agg(['mean','std'])
 summary.to_csv(folder/'summary.csv');save(folder/'report.json',{'manifest':manifest,'threshold':.5,'limitations':['Exploratory control on previously examined fixed partitions.','Mean and sample SD describe training randomness, not population uncertainty.','No paired significance claim; runtime versions may differ from original training.','Do not replace original baseline scores or retune on these test results.']})
 print(summary.to_string());print('Saved:',folder)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--output',type=Path);mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--self-test',action='store_true');mode.add_argument('--check-only',action='store_true');mode.add_argument('--seed',type=int,choices=SEEDS);mode.add_argument('--evaluate',action='store_true');a=ap.parse_args()
 if a.self_test:self_test();return
 root=a.root.resolve();out=(a.output or root/'masked_bilstm_control_v1').resolve()
 if out==root or out==root/'baseline_results_v3' or root/'baseline_results_v3' in out.parents:raise ValueError('Use a separate control output directory')
 frames,refs,manifest=preflight(root);env=runtime_check()
 if a.check_only:print('CHECK PASSED. No training, test inference or output directories created.');return
 if a.evaluate:evaluate(out,frames,refs,manifest)
 else:train(out,a.seed,frames,refs,manifest,env)
if __name__=='__main__':main()
