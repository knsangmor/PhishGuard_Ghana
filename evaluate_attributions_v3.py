"""Frozen seed-42 attribution audit. Inference only; resumable per-URL checkpoints."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import argparse
import hashlib
import json
import time
import platform
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_SHA='cdc771274f9512c5bf24bead1742f5ee3cbc8fa3af182b2bd357ed5d6f0b88e4'
PART_SHA={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
N_EXPLAIN=200
N_BACKGROUND=32
SHAP_SAMPLES=64
IG_STEPS=64
RANDOM_REPEATS=10
FRACTIONS=np.array([0.,.1,.2,.5,1.])


def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def normalize(u):
 u=str(u).strip()
 for prefix in ['https://','http://']:
  if u.lower().startswith(prefix):u=u[len(prefix):];break
 return u.lower().strip()

def encode(u):
 a=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in normalize(u)[:200]]
 return a+[0]*(200-len(a))

def sample_indices(labels,n,seed):
 rng=np.random.default_rng(seed)
 indices=np.concatenate([rng.choice(np.flatnonzero(np.asarray(labels)==label),n//2,replace=False) for label in [0,1]])
 rng.shuffle(indices);return indices

def perturb(seq,order,fraction,kind):
 active=np.flatnonzero(seq!=0);k=int(np.ceil(len(active)*fraction));selected=np.asarray(order[:k],dtype=int)
 result=np.zeros_like(seq)
 if kind=='deletion':
  kept=[i for i in active if i not in set(selected)];result[:len(kept)]=seq[kept]
 elif kind=='insertion':result[selected]=seq[selected]
 else:raise ValueError(kind)
 return result

def collapse_shap(values):
 if isinstance(values,list):
  if len(values)!=1:raise ValueError('Unexpected multi-output SHAP')
  values=values[0]
 a=np.asarray(values)
 if a.shape==(1,200,64,1):a=a[...,0]
 if a.shape!=(1,200,64):raise ValueError(f'Unexpected SHAP shape: {a.shape}')
 return a[0].sum(axis=-1)

def region(text,pos):
 if pos>=len(text):return 'padding'
 # Structural string regions only; does not assert a registrable hostname.
 end=min([text.find(c) for c in '/?#' if c in text] or [len(text)])
 if pos<end:return 'authority_text'
 fragment=text.find('#');query=text.find('?')
 if fragment>=0 and pos>=fragment:return 'fragment'
 if query>=0 and pos>=query and (fragment<0 or query<fragment):return 'query'
 return 'path'

def make_tail(original,keras):
 expected=['url_input','char_embedding','embed_dropout','conv_k3','conv_k5','merge_cnn','maxpool','bilstm','dropout','dense64','dropout2','output']
 if [l.name for l in original.layers]!=expected or original.count_params()!=244673:raise ValueError('Unexpected model architecture')
 if original.get_layer('char_embedding').mask_zero:raise ValueError('Masked embedding needs a different wrapper')
 x=keras.Input(shape=(200,64),dtype='float32',name='audit_embedding_input')
 z=original.get_layer('embed_dropout')(x,training=False)
 branches=[original.get_layer(n)(z) for n in ['conv_k3','conv_k5']]
 z=original.get_layer('merge_cnn')(branches)
 for name in ['maxpool','bilstm','dropout','dense64','dropout2','output']:z=original.get_layer(name)(z,training=False)
 return keras.Model([x],z,name='saved_layers_attribution_tail')

def make_ig(tail,tf):
 @tf.function(input_signature=[tf.TensorSpec([None,200,64],tf.float32)],reduce_retracing=True)
 def gradients(z):
  with tf.GradientTape() as tape:
   tape.watch(z);p=tail([z],training=False)[:,0]
  result=tape.gradient(p,z)
  return result
 def integrated(x,baseline):
  alpha=np.linspace(0.,1.,IG_STEPS+1,dtype=np.float32)
  path=baseline[None]+alpha[:,None,None]*(x-baseline)[None]
  g=np.concatenate([gradients(tf.constant(path[i:i+16])).numpy() for i in range(0,len(path),16)])
  avg=((g[:-1]+g[1:])*.5).mean(axis=0)
  return ((x-baseline)*avg).sum(axis=-1)
 return integrated

def predict(model,x):
 return np.concatenate([model(np.asarray(x[i:i+128],dtype=np.int32),training=False).numpy().reshape(-1) for i in range(0,len(x),128)])

def save_json(path,obj):
 temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def evaluate_one(seq,shap_values,ig_values,pred,predict_fn,index):
 active=np.flatnonzero(seq!=0);direction=1 if pred>=.5 else -1
 rng=np.random.default_rng(600000+int(index))
 orders={'shap':active[np.argsort(-(shap_values[active]*direction),kind='stable')],
         'integrated_gradients':active[np.argsort(-(ig_values[active]*direction),kind='stable')]}
 for j in range(RANDOM_REPEATS):orders[f'random_{j}']=rng.permutation(active)
 batch=[];keys=[]
 for method,order in orders.items():
  for kind in ['deletion','insertion']:
   for f in FRACTIONS:batch.append(perturb(seq,order,f,kind));keys.append((method,kind,float(f)))
 scores=predict_fn(np.array(batch,dtype=np.int32))
 curves={}
 for (method,kind,f),p in zip(keys,scores):
  curves.setdefault((method,kind),[]).append(float(p if direction==1 else 1-p))
 # All rankings share the expected endpoints.
 for (method,kind),values in curves.items():
  endpoint=values[0] if kind=='deletion' else values[-1]
  if abs(endpoint-(pred if direction==1 else 1-pred))>2e-5:raise ValueError('Perturbation endpoint mismatch')
 return orders,curves

def summarize(out,parts,indices,bg_indices,seqs,probs,pad_p,bg_p,protocol,tf,keras,shap):
 character_rows=[];url_rows=[];curve_rows=[];faith=[]
 for sample_i,index in enumerate(indices):
  cache=out/'cache'/f'url_{int(index):05d}.npz'
  with np.load(cache,allow_pickle=False) as c:
   sv=c['shap'];ig=c['ig'];scores=c['curves'];elapsed=float(c['seconds']);order_s=c['shap_order'];order_i=c['ig_order']
  seq=seqs[sample_i];text=normalize(parts['test'].url.iloc[index]);p=float(probs[sample_i]);label=int(parts['test'].label.iloc[index]);pred=int(p>=.5)
  active=seq!=0
  abs_sv=np.abs(sv);abs_ig=np.abs(ig)
  n_top=min(10,int(active.sum()));ss=set(order_s[:n_top].tolist());ii=set(order_i[:n_top].tolist())
  corr=pd.Series(sv[active]).corr(pd.Series(ig[active]),method='spearman') if active.sum()>1 else float('nan')
  url_rows.append({'sample_position':sample_i,'test_row_index':int(index),'label':label,'predicted_label':pred,'correct':pred==label,'malicious_score':p,
   'encoded_length':int(active.sum()),'normalized_length':len(text),'truncated':len(text)>200,
   'shap_completeness_residual':p-bg_p-float(sv.sum()),'ig_completeness_residual':p-pad_p-float(ig.sum()),
   'shap_absolute_mass_padding_fraction':float(abs_sv[~active].sum()/abs_sv.sum()) if abs_sv.sum() else 0.,
   'ig_absolute_mass_padding_fraction':float(abs_ig[~active].sum()/abs_ig.sum()) if abs_ig.sum() else 0.,
   'signed_spearman_nonpadding':None if not np.isfinite(corr) else float(corr),
   'top10_predicted_direction_jaccard':len(ss&ii)/len(ss|ii) if ss|ii else None,'attribution_seconds':elapsed})
  for pos in range(200):
   token=int(seq[pos]);char=text[pos] if token else '<PAD>'
   character_rows.append({'sample_position':sample_i,'test_row_index':int(index),'label':label,'predicted_label':pred,
    'position_zero_based':pos,'token_id':token,'actual_character':char,'unknown_token':token==96,
    'is_padding':token==0,'structural_region':region(text,pos),'shap_signed':float(sv[pos]),'ig_signed':float(ig[pos])})
  methods=['shap','integrated_gradients']+[f'random_{j}' for j in range(RANDOM_REPEATS)]
  for method_i,method in enumerate(methods):
   for kind_i,kind in enumerate(['deletion','insertion']):
    for frac_i,f in enumerate(FRACTIONS):curve_rows.append({'test_row_index':int(index),'label':label,'predicted_label':pred,'method':method,'perturbation':kind,'fraction':float(f),'original_predicted_class_score':float(scores[method_i,kind_i,frac_i])})
  random_deletion=np.trapz(scores[2:,0,:],FRACTIONS,axis=1).mean();random_insertion=np.trapz(scores[2:,1,:],FRACTIONS,axis=1).mean()
  for method_i,method in enumerate(methods[:2]):
   da=float(np.trapz(scores[method_i,0,:],FRACTIONS));ia=float(np.trapz(scores[method_i,1,:],FRACTIONS))
   faith.append({'test_row_index':int(index),'label':label,'predicted_label':pred,'correct':pred==label,'method':method,'deletion_auc':da,'insertion_auc':ia,
    'random_mean_deletion_auc':float(random_deletion),'random_mean_insertion_auc':float(random_insertion),
    'deletion_advantage_over_random':float(random_deletion-da),'insertion_advantage_over_random':float(ia-random_insertion)})
 chars=pd.DataFrame(character_rows);urls=pd.DataFrame(url_rows);curves=pd.DataFrame(curve_rows);faith=pd.DataFrame(faith)
 chars.to_csv(out/'character_attributions.csv',index=False);urls.to_csv(out/'url_diagnostics.csv',index=False);curves.to_csv(out/'perturbation_curves.csv',index=False);faith.to_csv(out/'faithfulness_per_url.csv',index=False)
 positional=[]
 for pos,g in chars.groupby('position_zero_based'):
  nonpad=g[~g.is_padding];pad=g[g.is_padding]
  counts=nonpad.actual_character.value_counts().head(5).to_dict()
  positional.append({'position_zero_based':int(pos),'sample_urls':len(g),'nonpadding_urls':len(nonpad),'padding_urls':len(pad),
   'nonpadding_fraction':len(nonpad)/len(g),'shap_mean_abs_all':g.shap_signed.abs().mean(),
   'shap_mean_abs_nonpadding':nonpad.shap_signed.abs().mean(),'shap_mean_abs_padding':pad.shap_signed.abs().mean(),
   'ig_mean_abs_nonpadding':nonpad.ig_signed.abs().mean(),'top_actual_characters_counts':json.dumps(counts,ensure_ascii=True)})
 pd.DataFrame(positional).to_csv(out/'position_occupancy_summary.csv',index=False)
 chars.assign(shap_abs=chars.shap_signed.abs(),ig_abs=chars.ig_signed.abs()).groupby('structural_region').agg(positions=('token_id','size'),shap_mean_abs=('shap_abs','mean'),ig_mean_abs=('ig_abs','mean')).to_csv(out/'region_summary.csv')
 summary=[]
 for method,g0 in faith.groupby('method'):
  for group,g in [('all',g0),('benign',g0[g0.label==0]),('malicious',g0[g0.label==1]),('model_correct',g0[g0.correct]),('model_incorrect',g0[~g0.correct])]:
   if not len(g):continue
   for metric in ['deletion_advantage_over_random','insertion_advantage_over_random']:
    summary.append({'method':method,'group':group,'urls':len(g),'metric':metric,'mean':g[metric].mean(),'sample_sd_across_urls':g[metric].std(ddof=1),'fraction_positive':(g[metric]>0).mean()})
 summary=pd.DataFrame(summary);summary.to_csv(out/'faithfulness_summary.csv',index=False)
 report={'complete':True,'protocol':protocol,'sample_rows':len(indices),'sample_benign':int((parts['test'].label.iloc[indices]==0).sum()),'sample_malicious':int((parts['test'].label.iloc[indices]==1).sum()),
  'training_performed':False,'model_modified':False,'background_mean_malicious_score':bg_p,'all_padding_malicious_score':pad_p,
  'mean_abs_shap_completeness_residual':urls.shap_completeness_residual.abs().mean(),'mean_abs_ig_completeness_residual':urls.ig_completeness_residual.abs().mean(),
  'mean_shap_padding_absolute_mass_fraction':urls.shap_absolute_mass_padding_fraction.mean(),
  'mean_signed_spearman_nonpadding':float(urls.signed_spearman_nonpadding.mean()) if urls.signed_spearman_nonpadding.notna().any() else None,'mean_top10_jaccard':urls.top10_predicted_direction_jaccard.mean(),
  'references':['https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html','https://www.tensorflow.org/tutorials/interpretability/integrated_gradients'],
  'versions':{'tensorflow':tf.__version__,'keras':keras.__version__,'shap':shap.__version__,'numpy':np.__version__,'python':platform.python_version()},
  'limitations':['Exploratory attribution evaluation of original seed 42 only; not broader-data models or five-seed attribution stability.',
   'Balanced URL-level sample is not a prevalence estimate; URLs may share domains and summary SD is not a population confidence interval.',
   'SHAP GradientExplainer estimates expected gradients in continuous embedding space, not exact discrete-character Shapley values.',
   'Integrated Gradients is related to expected gradients; agreement is not independent confirmation of correctness.',
   'Background is balanced training-only; alternative background distributions and sample sizes were not evaluated.',
   'Padding is a learned embedding, not a mask; its attribution may reflect different background lengths.',
   'Deletion compacts characters and shifts positions; insertion uses fixed positions with padding holes. Neither preserves URL syntax, label or live functionality.',
   'Insertion/deletion operate on the already truncated 200-token input; deletion does not pull unseen characters beyond the truncation boundary into the sequence.',
   'Random controls match URL, changed-character count and operation; these perturbations can still be out of distribution.',
   'Fraction-zero/all-padding baselines are diagnostic inputs, not representative URLs.',
   'Attribution completeness residuals are recorded, not forced to zero; finite sampling and integration errors remain.',
   'No human analyst usefulness, attacker intent, prediction truth or causal mechanism is established.']}
 save_json(out/'report.json',report)
 # Deterministic examples: first sampled URL in each original confusion-matrix group.
 example_ids=[]
 for label,pred in [(0,0),(0,1),(1,0),(1,1)]:
  group=urls[(urls.label==label)&(urls.predicted_label==pred)]
  if len(group):example_ids.append(int(group.test_row_index.iloc[0]))
 chars[chars.test_row_index.isin(example_ids)].to_csv(out/'worked_character_examples.csv',index=False)
 print('\nFAITHFULNESS SUMMARY (positive advantage means better than matched random order)\n'+summary.to_string(index=False))
 print('\nCOMPLETE. Saved to:',out)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--check-only',action='store_true');group.add_argument('--pilot',action='store_true');group.add_argument('--run',action='store_true')
 args=parser.parse_args();root=Path(__file__).resolve().parent;folder=root/'corrected_results_v3/seed_42'
 model_path=folder/'phishguard_gh_model.keras'
 if sha(model_path)!=MODEL_SHA:raise ValueError('Saved seed-42 model fingerprint differs')
 parts={}
 for name,expected in PART_SHA.items():
  p=root/'dataset_fix_outputs_v3'/f'{name}.csv'
  if sha(p)!=expected:raise ValueError(f'{name}: frozen partition differs')
  parts[name]=pd.read_csv(p,keep_default_na=False)
  if not parts[name].label.isin([0,1]).all():raise ValueError('Invalid partition labels')
 indices=sample_indices(parts['test'].label,N_EXPLAIN,42);bg_indices=sample_indices(parts['train'].label,N_BACKGROUND,1042)
 x=np.array([encode(u) for u in parts['test'].url.iloc[indices]],dtype=np.int32);bg=np.array([encode(u) for u in parts['train'].url.iloc[bg_indices]],dtype=np.int32)
 xt=np.load(folder/'X_test.npy',allow_pickle=False);yt=np.load(folder/'y_test.npy',allow_pickle=False);yp=np.load(folder/'y_prob.npy',allow_pickle=False).reshape(-1)
 if xt.shape!=(7653,200) or yp.shape!=(7653,) or not np.array_equal(yt.reshape(-1),parts['test'].label):raise ValueError('Saved test labels or array dimensions differ')
 # Verify complete test encoding, not just selected rows.
 if not np.array_equal(xt,np.array([encode(u) for u in parts['test'].url],dtype=np.int32)):raise ValueError('Saved test encoding differs')
 if not np.isfinite(yp).all() or ((yp<0)|(yp>1)).any():raise ValueError('Invalid saved scores')
 if not (x!=0).any(axis=1).all():raise ValueError('Empty sampled encoding')
 try:
  import tensorflow as tf
  import keras
  import shap
 except ImportError as error:
  raise SystemExit('Missing dependency: '+str(error)+'. Required: your existing TensorFlow/Keras plus shap. No training has started.')
 keras.utils.set_random_seed(42)
 original=keras.models.load_model(model_path,compile=False);tail=make_tail(original,keras);embedding=original.get_layer('char_embedding')
 emb=embedding(x).numpy().astype(np.float32);bg_emb=embedding(bg).numpy().astype(np.float32)
 actual=predict(original,x);wrapped=tail([emb],training=False).numpy().reshape(-1)
 max_wrapper=float(np.max(np.abs(actual-wrapped)));max_saved=float(np.max(np.abs(actual-yp[indices])))
 if max_wrapper>2e-5 or max_saved>2e-5:raise ValueError(f'Prediction mismatch: wrapper {max_wrapper}, saved {max_saved}')
 pad_seq=np.zeros((1,200),np.int32);pad_emb=embedding(pad_seq).numpy()[0];pad_p=float(predict(original,pad_seq)[0]);bg_p=float(np.mean(predict(original,bg)))
 explainer=shap.GradientExplainer(tail,bg_emb,batch_size=16)
 print('Checking SHAP gradients on one selected URL... ',flush=True)
 smoke=collapse_shap(explainer.shap_values(emb[:1],nsamples=4,rseed=42))
 if not np.isfinite(smoke).all():raise ValueError('Non-finite SHAP gradients')
 ig_fn=make_ig(tail,tf)
 print(f'CHECK PASSED: model identity; full test encoding; {N_EXPLAIN} random test URLs; {N_BACKGROUND} training-only background URLs; wrapper max difference {max_wrapper:.3g}; saved-score max difference {max_saved:.3g}.',flush=True)
 if args.check_only:
  print('Model loading and a small gradient/inference check only. No training, reports or output directories.');return
 file_inputs={'model':MODEL_SHA,'partitions':PART_SHA}
 for name in ['X_test.npy','y_test.npy','y_prob.npy']:file_inputs[name]=sha(folder/name)
 protocol={'software':{'tensorflow':tf.__version__,'keras':keras.__version__,'shap':shap.__version__,'numpy':np.__version__},'seed':42,'script_sha256':sha(Path(__file__)),'inputs':file_inputs,'test_indices':[int(i) for i in indices],'background_train_indices':[int(i) for i in bg_indices],
  'sample_selection':'100 random benign + 100 random composite malicious test rows, default_rng seed42, no correctness filtering; shuffled',
  'background_selection':'16 random benign + 16 random composite malicious training rows, default_rng seed1042; no test/validation background',
  'shap_nsamples':SHAP_SAMPLES,'ig_trapezoid_steps':IG_STEPS,'random_repeats':RANDOM_REPEATS,'fractions':FRACTIONS.tolist(),
  'explained_output':'malicious sigmoid score; signed position attribution is sum over 64 embedding coordinates',
  'ranking':'descending signed attribution toward the original predicted class, nonpadding positions only',
  'padding_baseline':'actual learned embedding for token 0; inference mode, dropout disabled',
  'maximum_wrapper_prediction_difference':max_wrapper,'maximum_saved_prediction_difference':max_saved}
 # Runtime numeric differences do not determine cache identity.
 stable={k:v for k,v in protocol.items() if not k.startswith('maximum_')}
 protocol_id=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()[:12]
 out=root/'attribution_evaluation_v3'/f'seed_42_{protocol_id}';out.mkdir(parents=True,exist_ok=True);(out/'cache').mkdir(exist_ok=True)
 if (out/'protocol.json').exists():
  old=json.loads((out/'protocol.json').read_text());old_stable={k:v for k,v in old.items() if not k.startswith('maximum_')}
  if old_stable!=stable:raise ValueError('Existing protocol differs')
 else:save_json(out/'protocol.json',protocol)
 for name,frame,selected in [('test_sample.csv',parts['test'],indices),('training_background.csv',parts['train'],bg_indices)]:
  selection=frame.iloc[selected].copy();selection.insert(0,'partition_row_index',selected);selection.to_csv(out/name,index=False)
 limit=4 if args.pilot else N_EXPLAIN;start=time.time()
 for j,index in enumerate(indices[:limit]):
  cache=out/'cache'/f'url_{int(index):05d}.npz'
  if cache.exists():
   with np.load(cache,allow_pickle=False) as c:
    if int(c['test_index'])!=int(index) or c['shap'].shape!=(200,) or c['ig'].shape!=(200,) or c['curves'].shape!=(12,2,5) or not all(np.isfinite(c[k]).all() for k in ['shap','ig','curves']):raise ValueError('Invalid checkpoint')
   print(f'Reusing {j+1}/{limit}, test row {index}',flush=True);continue
  tick=time.time();sv=collapse_shap(explainer.shap_values(emb[j:j+1],nsamples=SHAP_SAMPLES,rseed=420000+int(index)))
  ig=ig_fn(emb[j],pad_emb)
  if not np.isfinite(sv).all() or not np.isfinite(ig).all():raise ValueError('Non-finite attribution')
  orders,curves=evaluate_one(x[j],sv,ig,float(actual[j]),lambda z:predict(original,z),int(index))
  names=['shap','integrated_gradients']+[f'random_{r}' for r in range(RANDOM_REPEATS)]
  cv=np.array([[curves[(n,k)] for k in ['deletion','insertion']] for n in names])
  temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,test_index=int(index),shap=sv,ig=ig,curves=cv,shap_order=orders['shap'],ig_order=orders['integrated_gradients'],seconds=time.time()-tick);temp.replace(cache)
  print(f'Completed {j+1}/{limit}, test row {index}; {time.time()-tick:.1f}s; elapsed {(time.time()-start)/60:.1f}m',flush=True)
 if sha(model_path)!=MODEL_SHA:raise ValueError('Source model changed during audit')
 if args.pilot:
  save_json(out/'pilot_report.json',{'pilot_complete':True,'cached_urls':4,'full_sample_urls':N_EXPLAIN,'full_evaluation_complete':False,'output_folder':str(out),'training_performed':False})
  print('\nPILOT COMPLETE: four checkpoints saved. Continue with --run; these four URLs will be reused.\nSaved to:',out);return
 summarize(out,parts,indices,bg_indices,x,actual,pad_p,bg_p,protocol,tf,keras,shap)

if __name__=='__main__':main()
