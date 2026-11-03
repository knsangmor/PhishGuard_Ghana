"""Eight exploratory one-seed experiments on frozen v3 partitions. No evidence-chain generation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
HASHES={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
AUG_HASH='a9c037a1546e996c874e5a07c9f58859b625224b639598f30c7837d7420b3f53'
BASE={'length':200,'embedding':64,'kernels':[3,5],'patience':12,'strip_scheme':True,'augmentation':True}
CHANGES={'retain_scheme':{'strip_scheme':False},'no_augmentation':{'augmentation':False},'length_300':{'length':300},'embedding_32':{'embedding':32},'embedding_128':{'embedding':128},'kernel_3_only':{'kernels':[3]},'kernel_5_only':{'kernels':[5]},'patience_6':{'patience':6}}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def canonical(url):
 text=url.strip()
 if text.startswith('https://'):text=text[8:]
 elif text.startswith('http://'):text=text[7:]
 return text.lower().strip()

def encode(url,length=200,strip_scheme=True):
 text=str(url).strip()
 if strip_scheme:
  for prefix in ['https://','http://']:
   if text.lower().startswith(prefix):text=text[len(prefix):];break
 tokens=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in text.lower().strip()[:length]]
 return tokens+[0]*(length-len(tokens))

def build(config,keras):
 from keras import layers,regularizers
 inp=keras.Input(shape=(config['length'],),name='url_input')
 emb=layers.Embedding(97,config['embedding'],name='char_embedding')(inp)
 h=layers.SpatialDropout1D(.2,name='embed_dropout')(emb)
 branches=[layers.Conv1D(128,k,activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4),name=f'conv_k{k}')(h) for k in config['kernels']]
 h=layers.Concatenate(name='merge_cnn')(branches) if len(branches)>1 else branches[0]
 h=layers.MaxPooling1D(2,name='maxpool')(h)
 h=layers.Bidirectional(layers.LSTM(64,kernel_regularizer=regularizers.l2(1e-4),name='lstm'),name='bilstm')(h)
 h=layers.Dropout(.5,name='dropout')(h)
 h=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-4),name='dense64')(h)
 h=layers.Dropout(.3,name='dropout2')(h)
 output=layers.Dense(1,activation='sigmoid',name='output')(h)
 model=keras.Model(inp,output,name='PhishGuard_GH')
 model.compile(optimizer=keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=['accuracy'])
 return model

def write_json(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 mode=parser.add_mutually_exclusive_group(required=True)
 mode.add_argument('--check-only',action='store_true')
 mode.add_argument('--experiment',choices=['all']+list(CHANGES))
 args=parser.parse_args()
 import numpy as np,pandas as pd,keras
 from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
 parts={}
 for name,expected in HASHES.items():
  path=ROOT/'dataset_fix_outputs_v3'/f'{name}.csv'
  if sha(path)!=expected:raise ValueError(f'Partition hash mismatch: {name}')
  parts[name]=pd.read_csv(path,keep_default_na=False)
 aug_path=ROOT/'modern_legit_augmentation.csv'
 if sha(aug_path)!=AUG_HASH:raise ValueError('Augmentation fingerprint mismatch')
 generated=set(pd.read_csv(aug_path,keep_default_na=False).url.map(canonical))
 mask=~parts['train'].url.map(canonical).isin(generated)
 if len(parts['train'])!=35664 or mask.sum()!=34028:raise ValueError('Unexpected no-augmentation training membership')
 ref=ROOT/'corrected_results_v3'/'seed_42'
 metadata=json.loads((ref/'metrics.json').read_text())
 if metadata['seed']!=42 or metadata['dataset_version']!='v3':raise ValueError('Reference identity mismatch')
 for name in HASHES:
  if metadata['partitions'][name]['sha256']!=HASHES[name]:raise ValueError('Reference partition mismatch')
 for name in ['validation','test']:
  pred=pd.read_csv(ref/f'{name}_predictions.csv',keep_default_na=False)
  if not parts[name][['url','label']].equals(pred[['url','true_label']].rename(columns={'true_label':'label'})):raise ValueError('Reference prediction alignment mismatch')
 # Verify reference layer configurations before any new training.
 keras.backend.clear_session();keras.utils.set_random_seed(42)
 reference_arch=build(BASE,keras)
 saved=keras.models.load_model(ref/'phishguard_gh_model.keras',compile=False)
 if reference_arch.count_params()!=saved.count_params() or reference_arch.count_params()!=244673:raise ValueError('Reference parameter mismatch')
 def normalize_config(value):
  # Keras object IDs are process-specific serialization metadata.
  if isinstance(value,dict):
   return {k:normalize_config(v) for k,v in value.items() if k!='shared_object_id'}
  if isinstance(value,(tuple,list)):
   return [normalize_config(v) for v in value]
  return value
 def layers_signature(model):
  return [(layer.__class__.__name__,normalize_config(layer.get_config())) for layer in model.layers]
 def differences(a,b,path='config'):
  if type(a)!=type(b):
   return [f'{path}: new={a!r}; saved={b!r}']
  if isinstance(a,dict):
   result=[]
   for key in sorted(set(a)|set(b)):
    if key not in a or key not in b:result.append(f'{path}.{key}: missing on one side')
    else:result.extend(differences(a[key],b[key],f'{path}.{key}'))
   return result
  if isinstance(a,(list,tuple)):
   if len(a)!=len(b):return [f'{path}: different lengths']
   return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,f'{path}[{i}]')]
  return [] if a==b else [f'{path}: new={a!r}; saved={b!r}']
 new_sig=layers_signature(reference_arch);saved_sig=layers_signature(saved)
 if new_sig!=saved_sig:
  for detail in differences(new_sig,saved_sig)[:20]:print(detail,flush=True)
  raise ValueError('Reference architecture differs after removing serialization IDs; no training started')
 test_x=np.array([encode(u) for u in parts['test'].url],dtype=np.float32)
 if not np.array_equal(test_x,np.load(ref/'X_test.npy')):raise ValueError('Reference encoder mismatch')
 del reference_arch,saved,test_x
 # Fast overlap recheck for all encodings actually proposed.
 for length,strip in [(200,True),(300,True),(200,False)]:
  sets={n:{hashlib.sha256(bytes(encode(u,length,strip))).digest() for u in d.url} for n,d in parts.items()}
  if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']:raise ValueError('Encoded overlap for proposed experiment')
 print('CHECK PASSED: frozen v3 files, reference architecture/encoding and proposed encoding overlaps.')
 print('Reference training: 35,664; no-augmentation training: 34,028; validation: 7,668; test: 7,653.')
 print('Eight exploratory seed-42 experiments. Completed experiments can be reused; unfinished ones restart.')
 if args.check_only:
  print('No training or output directories created.');return
 output=ROOT/'controlled_experiments_v3';output.mkdir(exist_ok=True)
 protocol={'seed':42,'base_config':BASE,'experiments':CHANGES,'partition_sha256':HASHES,'augmentation_sha256':AUG_HASH,'script_sha256':sha(__file__),'reference_model_sha256':sha(ref/'phishguard_gh_model.keras'),'epochs':50,'batch_size':256,'learning_rate':.001,'reduce_lr_patience':4,'reference_results':'corrected_results_v3/seed_42','note':'One-seed exploratory comparisons, fixed settings. Not exhaustive sensitivity or historical before/after replication. Single-kernel variants also change capacity. No-augmentation compares different training sizes/composition. Test outputs are not for choosing new settings.'}
 lock=output/'protocol.json'
 if lock.exists() and json.loads(lock.read_text())!=protocol:raise ValueError('Experiment protocol changed; preserve outputs and inspect before continuing')
 if not lock.exists():write_json(lock,protocol)
 names=list(CHANGES) if args.experiment=='all' else [args.experiment]
 for name in names:
  config={**BASE,**CHANGES[name]};folder=output/name;folder.mkdir(exist_ok=True)
  result_path=folder/'metrics.json'
  if result_path.exists():
   print(f'Reusing completed experiment: {name}');continue
  train=parts['train'] if config['augmentation'] else parts['train'].loc[mask]
  xs={n:np.array([encode(u,config['length'],config['strip_scheme']) for u in d.url],dtype=np.float32) for n,d in [('train',train),('validation',parts['validation']),('test',parts['test'])]}
  ys={'train':train.label.to_numpy(dtype=np.float32),'validation':parts['validation'].label.to_numpy(dtype=np.float32),'test':parts['test'].label.to_numpy(dtype=np.float32)}
  keras.backend.clear_session();keras.utils.set_random_seed(42)
  model=build(config,keras)
  print(f'\nTRAINING {name}: {config}; {len(train):,} training URLs',flush=True)
  tick=time.perf_counter()
  history=model.fit(xs['train'],ys['train'],validation_data=(xs['validation'],ys['validation']),epochs=50,batch_size=256,verbose=1,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'],restore_best_weights=True,verbose=1),keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=4,min_lr=1e-5,verbose=1),keras.callbacks.CSVLogger(str(folder/'epoch_history.csv'))])
  elapsed=time.perf_counter()-tick
  model.save(folder/'model.keras');write_json(folder/'history.json',history.history)
  result={'experiment':name,'seed':42,'config':config,'partition_sha256':HASHES,'training_rows':len(train),'model_parameters':model.count_params(),'training_seconds':elapsed,'best_epoch':int(np.argmin(history.history['val_loss']))+1,'metrics':{}}
  for split in ['validation','test']:
   p=model.predict(xs[split],batch_size=256,verbose=0).reshape(-1);y=ys[split];q=p>=.5
   pd.DataFrame({'url':parts[split].url,'true_label':y.astype(int),'probability':p,'pred_label':q.astype(int)}).to_csv(folder/f'{split}_predictions.csv',index=False)
   result['metrics'][split]={k:float(v) for k,v in dict(accuracy=accuracy_score(y,q),precision=precision_score(y,q,zero_division=0),recall=recall_score(y,q),f1_score=f1_score(y,q),auc_roc=roc_auc_score(y,p),average_precision=average_precision_score(y,p)).items()}
  write_json(result_path,result)
  print(f'EXPERIMENT COMPLETE: {name}; results saved to {folder}',flush=True)
  del model,xs
 print('\nREQUESTED EXPERIMENTS COMPLETE. Frozen data, main-model and baseline results preserved.')
if __name__=='__main__':main()
