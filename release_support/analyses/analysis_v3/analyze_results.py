import zipfile,json,io,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
z=zipfile.ZipFile(out.parent/'upload/corrected_results_v3.zip')
metrics=[];operating=[];curves=[];ref={};fingerprints=[]
for seed in range(42,47):
 prefix=f'corrected_results_v3/seed_{seed}/'
 m=json.loads(z.read(prefix+'metrics.json'))
 assert m['seed']==seed and m['dataset_version']=='v3'
 fingerprints.append(m['partitions'])
 v=pd.read_csv(io.BytesIO(z.read(prefix+'validation_predictions.csv')),keep_default_na=False)
 d=pd.read_csv(io.BytesIO(z.read(prefix+'test_predictions.csv')),keep_default_na=False)
 for name,frame,n in [('validation',v,7668),('test',d,7653)]:
  assert len(frame)==n and frame.true_label.isin([0,1]).all()
  assert np.isfinite(frame.probability).all() and frame.probability.between(0,1).all()
  identity=frame[['url','true_label']]
  if name in ref: assert identity.equals(ref[name])
  else:ref[name]=identity
 y=d.true_label.to_numpy();p=d.probability.to_numpy();pred=p>=.5
 scores=dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred),recall=recall_score(y,pred),f1_score=f1_score(y,pred),auc_roc=roc_auc_score(y,p),average_precision=average_precision_score(y,p))
 for k in m.keys() & scores.keys():assert abs(m[k]-scores[k])<1e-6,(seed,k)
 metrics.append(dict(seed=seed,**scores))
 fpr,tpr,thresholds=roc_curve(v.true_label,v.probability,drop_intermediate=False)
 choices=[('default_0.5',.5)]
 for target in [.001,.005,.01]:
  possible=np.flatnonzero(fpr<=target+1e-15)
  best=possible[np.argmax(tpr[possible])]
  choices.append((f'validation_FPR_cap_{target}',float(thresholds[best])))
 for policy,threshold in choices:
  yp=p>=threshold;tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
  vt=v.probability.to_numpy()>=threshold;vy=v.true_label.to_numpy()
  fr=fp/(fp+tn);rr=tp/(tp+fn)
  row=dict(seed=seed,policy=policy,threshold=threshold,validation_fpr=float(vt[vy==0].mean()),test_fpr=fr,test_recall=rr,test_precision=tp/(tp+fp) if tp+fp else np.nan,TN=tn,FP=fp,FN=fn,TP=tp)
  for prevalence in [.001,.01]:
   den=rr*prevalence+fr*(1-prevalence)
   row[f'precision_at_prevalence_{prevalence}']=rr*prevalence/den if den else np.nan
  operating.append(row)
 pr,rc,_=precision_recall_curve(y,p);curves.append((seed,rc,pr,scores['average_precision']))
for name in ['train','validation','test']:
 assert len({x[name]['sha256'] for x in fingerprints})==1
md=pd.DataFrame(metrics);op=pd.DataFrame(operating)
md.to_csv(out/'five_seed_metrics.csv',index=False);op.to_csv(out/'operating_points.csv',index=False)
summ=[]
for k in md.columns.drop('seed'):
 a=md[k];mean=a.mean();sd=a.std(ddof=1);half=t.ppf(.975,4)*sd/np.sqrt(5)
 summ.append(dict(metric=k,mean=mean,sample_sd=sd,ci95_low=mean-half,ci95_high=mean+half))
s=pd.DataFrame(summ);s.to_csv(out/'five_seed_summary.csv',index=False)
agg=op.groupby('policy')[[c for c in op.columns if c not in ['seed','policy','threshold','TN','FP','FN','TP']]].agg(['mean','std'])
agg.to_csv(out/'operating_point_summary.csv')
plt.figure(figsize=(7,5))
for seed,rc,pr,ap in curves:plt.plot(rc,pr,label=f'Seed {seed}: AP {ap:.4f}',lw=1.4)
plt.axhline(ref['test'].true_label.mean(),color='gray',ls='--',label='Test prevalence')
plt.xlim(0,1);plt.ylim(0,1.01);plt.xlabel('Recall (malicious class)');plt.ylabel('Precision');plt.title('Version 3 test precision–recall curves');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/'precision_recall.png',dpi=200);plt.close()
report='''Version 3 evaluation: five fixed-partition training seeds\n\nValidation: 7,668 URLs. Test: 7,653 URLs (4,056 benign, 3,597 malicious).\nAll five runs share partition fingerprints and URL/label order. Saved metrics agree with recomputation from CSV probabilities to rounding tolerance.\n\nMean, sample SD and 95% Student-t intervals for the mean across five seeds:\n'''+s.to_string(index=False)+'''\n\nIntervals quantify training-seed uncertainty conditional on these fixed partitions. They do not capture uncertainty over datasets, domain sampling or deployment traffic; five seeds give limited precision.\n\nOperating-point protocol: baseline threshold 0.5 plus validation FPR caps 0.1%, 0.5% and 1%. For each cap, select the ROC threshold with maximum validation recall among feasible thresholds (ties favor the first/higher threshold). Freeze it and evaluate on test data. No threshold is selected from test scores. Test FPR need not meet its validation cap. Validation contains 4,055 benign URLs, limiting low-FPR resolution.\n\nPrecision projections at 0.1% and 1% malicious prevalence are hypothetical prior-shift scenarios, not measured Ghana traffic rates. They assume test TPR/FPR transfer to deployment. The positive label includes phishing, malware and defacement; these are malicious-prevalence projections, not phishing-only estimates. PR curves and average precision reflect the observed test prevalence (~47%).\n\nOperating-point summaries (arithmetic mean and sample SD across seeds):\n'''+agg.to_string()+'''\n\nRemaining reviewer work includes baseline validation tuning and scheme-feature ablation, paired comparisons, shortcut/augmentation ablations, sensitivity and adaptive-input tests, independent external evaluation, CPU/logging overhead benchmarks, and manuscript updates. These files do not close all reviewer points.\n'''
(out/'evaluation_notes.txt').write_text(report)
print(s.to_string(index=False));print(agg.to_string())
