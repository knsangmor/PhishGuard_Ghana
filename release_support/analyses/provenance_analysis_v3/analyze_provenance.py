import io,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
aug=pd.read_csv(ROOT/'upload/modern_legit_augmentation.csv',keep_default_na=False)
def key(u):return re.sub(r'^https?://','',u.strip()).lower().strip()
keys=set(aug.url.map(key));assert len(keys)==2244 and (aug.label==0).all()
mz=zipfile.ZipFile(ROOT/'upload/corrected_results_v3.zip');bz=zipfile.ZipFile(ROOT/'upload/baseline_results_v3.zip')
selected=json.loads(bz.read('baseline_results_v3/selected_configs.json'))
snapshot=pd.read_csv(io.BytesIO(mz.read('corrected_results_v3/seed_42/dataset_training_snapshot.csv')),keep_default_na=False)
counts=[];start=0
for part,n in [('train',35664),('validation',7668),('test',7653)]:
 d=snapshot.iloc[start:start+n];start+=n;g=d.url.map(key).isin(keys)
 counts.append(dict(partition=part,total=n,generated=int(g.sum()),other=int((~g).sum()),benign=int((d.label==0).sum()),malicious=int((d.label==1).sum())))
assert start==len(snapshot)
pd.DataFrame(counts).to_csv(OUT/'partition_provenance_counts.csv',index=False)
rows=[]
for seed in range(42,47):
 main=pd.read_csv(io.BytesIO(mz.read(f'corrected_results_v3/seed_{seed}/test_predictions.csv')),keep_default_na=False)
 for model in ['PhishGuard_GH']+list(selected):
  d=main if model=='PhishGuard_GH' else pd.read_csv(io.BytesIO(bz.read(f'baseline_results_v3/seed_{seed}/{model}/test_predictions.csv')),keep_default_na=False)
  assert d[['url','true_label']].equals(main[['url','true_label']])
  generated=d.url.map(key).isin(keys).to_numpy();assert generated.sum()==279 and (d.true_label[generated]==0).all()
  for subset,mask in [('generated_assumed_benign',generated),('other_source',~generated)]:
   y=d.true_label.to_numpy()[mask];p=d.probability.to_numpy()[mask];q=p>=.5
   tn=int(((y==0)&~q).sum());fp=int(((y==0)&q).sum());tp=int(((y==1)&q).sum());fn=int(((y==1)&~q).sum())
   rows.append(dict(model=model,seed=seed,subset=subset,rows=len(y),TN=tn,FP=fp,TP=tp,FN=fn,accuracy=accuracy_score(y,q),fpr=fp/(tn+fp),specificity=tn/(tn+fp),precision=precision_score(y,q,zero_division=0) if (y==1).any() else np.nan,recall=recall_score(y,q) if (y==1).any() else np.nan,f1=f1_score(y,q) if (y==1).any() else np.nan,auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan))
a=pd.DataFrame(rows);a.to_csv(OUT/'per_seed_subset_metrics.csv',index=False)
s=a.groupby(['model','subset'])[['accuracy','fpr','specificity','precision','recall','f1','auc']].agg(['mean','std']);s.to_csv(OUT/'subset_summary.csv')
(OUT/'methods.txt').write_text('''Version 3 original/generated evaluation for all nine model pipelines and five seeds. Threshold 0.5. Generated membership uses the previously audited scheme-stripped/lowercased key and the saved 2,244-row augmentation file. This subset is assumed benign synthetic URLs, not independently validated benign web resources. Other-source membership is the complement of that mask; its historical source matching was established in the earlier source audit, not repeated here because the full original corpus is not part of these inputs. Snapshot partition boundaries use the verified version 3 train/validation/test order and sizes.\n\nAll 279 generated test records are benign-labelled; positive-class precision/recall/F1/AUC are deliberately left unavailable for that subset. Report false-positive rate and specificity instead. Means and sample SD are across fitted seeds on the same records. This subset evaluation is not an augmentation ablation and cannot demonstrate augmentation benefit. Full-source label ambiguity and the assumed labels of generated URLs remain relevant limitations.\n''')
print(pd.DataFrame(counts).to_string(index=False))
print(a[a.subset=='generated_assumed_benign'].groupby('model').fpr.agg(['mean','std']).to_string())
