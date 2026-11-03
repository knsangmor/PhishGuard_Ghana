"""Compact character Transformer reference, not a published-model reproduction.
Frozen original v3 data only. --check-only; --tune; --seed 42..46; --benchmark.
Test outcomes never select settings. Completed training runs are reused.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse, hashlib, json, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parent
HASHES={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
SIZES={'train':35664,'validation':7668,'test':7653}
CONFIGS=[{'width':64,'heads':4,'blocks':2,'ff_dim':128},{'width':96,'heads':4,'blocks':2,'ff_dim':192}]
MAIN_SHA='cdc771274f9512c5bf24bead1742f5ee3cbc8fa3af182b2bd357ed5d6f0b88e4'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p); q=p.with_suffix('.tmp'); q.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8'); q.replace(p)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def encode(u):
 u=str(u).strip()
 for scheme in ('https://','http://'):
  if u.lower().startswith(scheme): u=u[len(scheme):]; break
 v=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in u.lower().strip()[:200]]
 return v+[0]*(200-len(v))

def register(keras):
 ops=keras.ops
 @keras.saving.register_keras_serializable(package='PhishGuardReference')
 class TokenPosition(keras.layers.Layer):
  def __init__(self,width,**kw):
   super().__init__(**kw); self.width=width
   self.token=keras.layers.Embedding(97,width); self.position=keras.layers.Embedding(200,width)
  def build(self,shape):
   self.token.build(shape); self.position.build((200,)); super().build(shape)
  def call(self,ids): return self.token(ids)+self.position(ops.arange(200))
  def get_config(self): return {**super().get_config(),'width':self.width}
 @keras.saving.register_keras_serializable(package='PhishGuardReference')
 class Block(keras.layers.Layer):
  def __init__(self,width,heads,ff_dim,**kw):
   super().__init__(**kw); self.width=width; self.heads=heads; self.ff_dim=ff_dim
   self.att=keras.layers.MultiHeadAttention(heads,width//heads,dropout=.1)
   self.n1=keras.layers.LayerNormalization(epsilon=1e-6); self.n2=keras.layers.LayerNormalization(epsilon=1e-6)
   self.f1=keras.layers.Dense(ff_dim,activation='gelu'); self.f2=keras.layers.Dense(width)
   self.d1=keras.layers.Dropout(.2); self.d2=keras.layers.Dropout(.2)
  def build(self,shapes):
   s=shapes[0]; self.att.build(s,s); self.n1.build(s); self.n2.build(s); self.f1.build(s)
   self.f2.build(tuple(s[:-1])+(self.ff_dim,)); super().build(shapes)
  def call(self,inputs,training=None):
   x,ids=inputs; mask=ops.expand_dims(ops.not_equal(ids,0),1)
   a=self.att(x,x,attention_mask=mask,training=training)
   x=self.n1(x+self.d1(a,training=training))
   return self.n2(x+self.d2(self.f2(self.f1(x)),training=training))
  def get_config(self): return {**super().get_config(),'width':self.width,'heads':self.heads,'ff_dim':self.ff_dim}
 @keras.saving.register_keras_serializable(package='PhishGuardReference')
 class MaskedMean(keras.layers.Layer):
  def call(self,inputs):
   x,ids=inputs; w=ops.cast(ops.expand_dims(ops.not_equal(ids,0),-1),x.dtype)
   return ops.sum(x*w,axis=1)/ops.maximum(ops.sum(w,axis=1),1)
 return TokenPosition,Block,MaskedMean

def build(config,seed,keras):
 keras.backend.clear_session(); keras.utils.set_random_seed(seed)
 TokenPosition,Block,MaskedMean=register(keras)
 ids=keras.Input((200,),dtype='int32'); x=TokenPosition(config['width'])(ids)
 for _ in range(config['blocks']): x=Block(config['width'],config['heads'],config['ff_dim'])([x,ids])
 x=MaskedMean()([x,ids]); x=keras.layers.Dense(64,activation='relu')(x); x=keras.layers.Dropout(.3)(x)
 model=keras.Model(ids,keras.layers.Dense(1,activation='sigmoid')(x))
 model.compile(optimizer=keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=['accuracy'])
 return model

def preflight():
 frames={}; X={}; y={}
 for name in HASHES:
  p=ROOT/'dataset_fix_outputs_v3'/f'{name}.csv'
  if sha(p)!=HASHES[name]: raise ValueError(f'{name}: frozen fingerprint differs')
  f=pd.read_csv(p,keep_default_na=False)
  if len(f)!=SIZES[name] or not f.label.isin([0,1]).all(): raise ValueError(f'{name}: invalid size or labels')
  frames[name]=f; X[name]=np.asarray([encode(u) for u in f.url],dtype=np.int32); y[name]=f.label.to_numpy(dtype=np.int32)
 run=ROOT/'corrected_results_v3'/'seed_42'
 if sha(run/'phishguard_gh_model.keras')!=MAIN_SHA: raise ValueError('Main seed-42 model fingerprint differs')
 if not np.array_equal(X['test'],np.load(run/'X_test.npy',allow_pickle=False)): raise ValueError('Main test encoding differs')
 for seed in range(42,47):
  d=ROOT/'corrected_results_v3'/f'seed_{seed}'
  p=np.load(d/'y_prob.npy',allow_pickle=False).reshape(-1)
  if not np.array_equal(y['test'],np.load(d/'y_test.npy',allow_pickle=False).reshape(-1)) or p.shape!=y['test'].shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError(f'Seed {seed}: saved scores/labels invalid')
 return frames,X,y

def protocol(base,keras,tf):
 value={'script_sha256':sha(__file__),'partition_sha256':HASHES,'candidates':CONFIGS,'selection':'seed42 validation average precision at minimum-validation-loss checkpoint; first candidate breaks ties','training':'Adam .001, batch256, max50, early stopping val_loss patience12 restore best; LR half after4, minimum1e-5','encoding':'scheme stripped, lowercase, first200 ASCII1..95 unknown96 rightpad0','scope':'compact from-scratch character Transformer reference; not URLBERT/TransURL reproduction, no pretraining','mask':'padding keys excluded from attention; padding excluded from mean pooling','python':platform.python_version(),'tensorflow':tf.__version__,'keras':keras.__version__}
 base.mkdir(exist_ok=True)
 p=base/'protocol.json'
 if p.exists() and read(p)!=value: raise ValueError('Protocol/software changed: preserve existing outputs; review before proceeding')
 if not p.exists(): write(p,value)

def train(folder,c,seed,X,y,keras):
 folder.mkdir(parents=True,exist_ok=True); done=folder/'complete.json'
 if done.exists():
  m=read(done)
  if m['config']!=c or m['seed']!=seed or m['model_sha256']!=sha(folder/'model.keras') or m['validation_scores_sha256']!=sha(folder/'validation_scores.npy'): raise ValueError('Completed training artifact mismatch')
  print('Reusing:',folder); return m
 print(f'TRAINING seed {seed}: {c}',flush=True)
 model=build(c,seed,keras); start=time.perf_counter()
 h=model.fit(X['train'],y['train'],validation_data=(X['validation'],y['validation']),epochs=50,batch_size=256,verbose=1,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True,verbose=1),keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=4,min_lr=1e-5,verbose=1)])
 seconds=time.perf_counter()-start
 p=model.predict(X['validation'],batch_size=256,verbose=0).reshape(-1)
 model.save(folder/'model.keras'); np.save(folder/'validation_scores.npy',p)
 write(folder/'history.json',{k:[float(z) for z in v] for k,v in h.history.items()})
 m={'config':c,'seed':seed,'training_seconds':seconds,'epochs':len(h.history['loss']),'best_epoch':int(np.argmin(h.history['val_loss']))+1,'validation_average_precision':float(average_precision_score(y['validation'],p)),'parameters':model.count_params(),'model_sha256':sha(folder/'model.keras'),'validation_scores_sha256':sha(folder/'validation_scores.npy')}
 write(done,m); return m

def tune(base,X,y,keras):
 records=[train(base/'tuning'/f'candidate_{i}',c,42,X,y,keras) for i,c in enumerate(CONFIGS)]
 i=max(range(len(records)),key=lambda j:records[j]['validation_average_precision'])
 selected={'candidate':i,'config':CONFIGS[i],'validation_average_precision':records[i]['validation_average_precision']}
 p=base/'selected.json'
 if p.exists() and read(p)!=selected: raise ValueError('Saved selection differs')
 write(p,selected); print('SELECTED:',selected); return selected

def metrics(y,p):
 pred=p>=.5
 return {k:float(v) for k,v in {'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'auc_roc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p)}.items()}

def evaluate(base,selected,seed,frames,X,y,keras):
 folder=base/'tuning'/f"candidate_{selected['candidate']}" if seed==42 else base/f'seed_{seed}'
 m=train(folder,selected['config'],seed,X,y,keras)
 out=base/f'seed_{seed}'; out.mkdir(exist_ok=True); report=out/'metrics.json'
 if report.exists():
  r=read(report)
  if r['model_sha256']!=m['model_sha256'] or r['scores_sha256']!=sha(out/'y_prob.npy'): raise ValueError('Evaluation artifact mismatch')
  print('Reusing test evaluation:',out); return
 model=keras.models.load_model(folder/'model.keras',compile=False)
 p=model.predict(X['test'],batch_size=256,verbose=0).reshape(-1)
 np.save(out/'y_prob.npy',p); f=frames['test'].copy(); f['probability']=p; f['pred_label']=(p>=.5).astype(int); f.to_csv(out/'test_predictions.csv',index=False)
 r={'seed':seed,'threshold':.5,'model_sha256':m['model_sha256'],'scores_sha256':sha(out/'y_prob.npy'),'metrics':metrics(y['test'],p)}; write(report,r); print(r)
 if all((base/f'seed_{s}'/'metrics.json').exists() for s in range(42,47)):
  from scipy.stats import t
  rows=[{'seed':s,**read(base/f'seed_{s}'/'metrics.json')['metrics']} for s in range(42,47)]
  table=pd.DataFrame(rows); table.to_csv(base/'test_metrics_per_seed.csv',index=False)
  summary=[]
  for k in rows[0]:
   if k=='seed': continue
   a=table[k]; mean=float(a.mean()); sd=float(a.std(ddof=1)); margin=float(t.ppf(.975,4)*sd/np.sqrt(5))
   summary.append({'metric':k,'mean':mean,'sample_sd':sd,'ci95_lower':mean-margin,'ci95_upper':mean+margin})
  pd.DataFrame(summary).to_csv(base/'test_summary.csv',index=False)
  print('Five-seed summary saved; CIs describe training randomness on fixed data.')

def benchmark(base,selected,X,keras):
 out=base/'cpu_benchmark.json'
 if out.exists(): raise ValueError('Benchmark already exists; preserved. Review it before repeating.')
 trans=base/'tuning'/f"candidate_{selected['candidate']}"/'model.keras'
 original=ROOT/'corrected_results_v3'/'seed_42'/'phishguard_gh_model.keras'
 if not (base/'seed_42'/'metrics.json').exists(): raise ValueError('Evaluate seed42 first')
 models={'cnn_lstm':keras.models.load_model(original,compile=False),'transformer':keras.models.load_model(trans,compile=False)}
 expected={'cnn_lstm':np.load(original.parent/'y_prob.npy').reshape(-1),'transformer':np.load(base/'seed_42'/'y_prob.npy').reshape(-1)}
 indices=np.random.default_rng(142).choice(len(X['test']),size=512,replace=False)
 rows=[]
 for batch,inputs in [(1,X['test'][indices]),(256,X['test'])]:
  for model in models.values(): model.predict(inputs[:min(256,len(inputs))],batch_size=batch,verbose=0)
  for repeat in range(5):
   order=['cnn_lstm','transformer'] if repeat%2==0 else ['transformer','cnn_lstm']
   for name in order:
    start=time.perf_counter(); p=models[name].predict(inputs,batch_size=batch,verbose=0).reshape(-1); elapsed=time.perf_counter()-start
    ref=expected[name][indices] if batch==1 else expected[name]
    if not np.allclose(p,ref,atol=2e-5,rtol=0): raise ValueError(f'{name}: benchmark prediction mismatch')
    rows.append({'model':name,'batch_size':batch,'repeat':repeat+1,'rows':len(inputs),'milliseconds_per_url':1000*elapsed/len(inputs)})
    print(name,'batch',batch,'repeat',repeat+1,flush=True)
 pd.DataFrame(rows).to_csv(base/'cpu_timings.csv',index=False)
 summary=pd.DataFrame(rows).groupby(['model','batch_size']).milliseconds_per_url.agg(['mean','std']).reset_index()
 write(out,{'measurements':summary.to_dict('records'),'models_sha256':{'cnn_lstm':sha(original),'transformer':sha(trans)},'platform':platform.platform(),'processor':platform.processor(),'limitations':['Warm CPU model.predict timing; excludes URL encoding, loading, explanations and logging.','Batch1 is amortized predict-call throughput, not independent single-request latency.','Shared machine load can affect results; five alternated-order repeats.','Compact reference only; cannot establish costs of published transformers.']})
 print(summary.to_string(index=False))

def main():
 ap=argparse.ArgumentParser(description=__doc__); g=ap.add_mutually_exclusive_group(required=True)
 g.add_argument('--check-only',action='store_true'); g.add_argument('--tune',action='store_true'); g.add_argument('--seed',type=int,choices=range(42,47)); g.add_argument('--benchmark',action='store_true'); a=ap.parse_args()
 frames,X,y=preflight()
 import tensorflow as tf
 tf.config.set_visible_devices([], 'GPU')
 import keras
 register(keras)
 for c in CONFIGS:
  model=build(c,42,keras); p=model(X['validation'][:4],training=False).numpy()
  if p.shape!=(4,1) or not np.isfinite(p).all(): raise ValueError('Architecture forward check failed')
 print('CHECK PASSED: frozen v3 inputs, saved main-model test alignment and both Transformer architectures.')
 if a.check_only: print('No training or output directories created.'); return
 base=ROOT/'transformer_baseline_v3'; protocol(base,keras,tf)
 if a.tune: tune(base,X,y,keras); return
 if not (base/'selected.json').exists(): raise ValueError('Run --tune first; settings must be selected on validation data')
 selected=read(base/'selected.json')
 if selected['config']!=CONFIGS[selected['candidate']]: raise ValueError('Invalid selection')
 if a.seed is not None: evaluate(base,selected,a.seed,frames,X,y,keras)
 else: benchmark(base,selected,X,keras)
 print('Done. Original datasets, models and baseline results preserved.')
if __name__=='__main__': main()
