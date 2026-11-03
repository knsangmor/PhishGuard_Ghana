"""Reproduce version 3 baseline analysis from the two supplied archives."""
from pathlib import Path
import zipfile,json,io
import numpy as np,pandas as pd
from scipy.stats import t,binomtest
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
bz=zipfile.ZipFile(ROOT/'upload/baseline_results_v3.zip')
mz=zipfile.ZipFile(ROOT/'upload/corrected_results_v3.zip')
protocol=json.loads(bz.read('baseline_results_v3/protocol.json'))
selected=json.loads(bz.read('baseline_results_v3/selected_configs.json'))
models=list(selected)
all_metrics=[];preds={};ids={};ablation=[];mcn=[]
def read(z,path):return pd.read_csv(io.BytesIO(z.read(path)),keep_default_na=False)
def calc(y,p):
 q=p>=.5
 return dict(accuracy=accuracy_score(y,q),precision=precision_score(y,q,zero_division=0),recall=recall_score(y,q),f1_score=f1_score(y,q),auc_roc=roc_auc_score(y,p),average_precision=average_precision_score(y,p))
for seed in range(42,47):
 mainbase=f'corrected_results_v3/seed_{seed}/'
 mm=json.loads(mz.read(mainbase+'metrics.json'))
 for split in ['validation','test']:
  frame=read(mz,mainbase+split+'_predictions.csv')
  identity=frame[['url','true_label']]
  if split in ids:assert identity.equals(ids[split])
  else:ids[split]=identity
 main=read(mz,mainbase+'test_predictions.csv')
 y=main.true_label.to_numpy();preds[('PhishGuard_GH',seed)]=main.probability.to_numpy()
 all_metrics.append(dict(model='PhishGuard_GH',seed=seed,**calc(y,main.probability)))
 for model in models:
  base=f'baseline_results_v3/seed_{seed}/{model}/'
  met=json.loads(bz.read(base+'metrics.json'))
  assert met['seed']==seed and met['dataset_version']=='v3'
  assert met['partition_hashes']==protocol['partition_hashes']
  assert met['selected_config']==selected[model]
  for n,d in mm['partitions'].items():assert d['sha256']==met['partition_hashes'][n]
  for split in ['validation','test']:
   frame=read(bz,base+split+'_predictions.csv')
   assert frame[['url','true_label']].equals(ids[split]),(seed,model,split)
   assert frame.probability.between(0,1).all() and np.isfinite(frame.probability).all()
   assert np.array_equal(frame.pred_label,(frame.probability>=.5).astype(int))
  p=frame.probability.to_numpy();preds[(model,seed)]=p
  score=calc(y,p)
  for k,v in score.items():assert abs(v-met['test_metrics'][k])<1e-10
  all_metrics.append(dict(model=model,seed=seed,**score))
  a=main.probability.to_numpy()>=.5;b=p>=.5
  main_only=int(((a==y)&(b!=y)).sum());base_only=int(((a!=y)&(b==y)).sum())
  pv=binomtest(main_only,main_only+base_only,.5).pvalue if main_only+base_only else 1.
  mcn.append(dict(model=model,seed=seed,main_only_correct=main_only,baseline_only_correct=base_only,exact_mcnemar_p=pv))
df=pd.DataFrame(all_metrics);df.to_csv(OUT/'per_seed_metrics.csv',index=False)
summary=[]
for model,g in df.groupby('model',sort=False):
 for metric in g.columns.drop(['model','seed']):
  a=g[metric];m=a.mean();sd=a.std(ddof=1);h=t.ppf(.975,4)*sd/np.sqrt(5)
  summary.append(dict(model=model,metric=metric,mean=m,sample_sd=sd,ci95_low=m-h,ci95_high=m+h))
s=pd.DataFrame(summary);s.to_csv(OUT/'five_seed_summary.csv',index=False)
pd.DataFrame(mcn).to_csv(OUT/'mcnemar_per_seed_descriptive.csv',index=False)
for kind in ['RF','XGB']:
 for first,second in [(kind+'_full',kind+'_no_scheme'),(kind+'_no_scheme',kind+'_no_scheme_no_ghana')]:
  for seed in range(42,47):
   one=df[(df.model==first)&(df.seed==seed)].iloc[0];two=df[(df.model==second)&(df.seed==seed)].iloc[0]
   ablation.append(dict(first=first,second=second,seed=seed,accuracy_change=two.accuracy-one.accuracy,f1_change=two.f1_score-one.f1_score,auc_change=two.auc_roc-one.auc_roc,average_precision_change=two.average_precision-one.average_precision))
pd.DataFrame(ablation).to_csv(OUT/'ablation_per_seed.csv',index=False)
# Match the partition's domain grouping: bundled PSL, private domains enabled.
import tldextract,ipaddress
from urllib.parse import urlsplit
ext=tldextract.TLDExtract(suffix_list_urls=(),include_psl_private_domains=True)
def group(url):
 text=url.strip()
 if text.startswith('//'):text='https:'+text
 elif '://' not in text:text='https://'+text
 host=(urlsplit(text).hostname or '').lower().rstrip('.')
 try:return str(ipaddress.ip_address(host))
 except ValueError:return ext(host).top_domain_under_public_suffix or host
labels=ids['test'].true_label.to_numpy();groups=np.array([group(u) for u in ids['test'].url])
unique,inverse=np.unique(groups,return_inverse=True);G=len(unique);N=len(labels)
main_correct=np.mean([((preds[('PhishGuard_GH',seed)]>=.5)==labels) for seed in range(42,47)],axis=0)
D=[]
for model in models:
 base_correct=np.mean([((preds[(model,seed)]>=.5)==labels) for seed in range(42,47)],axis=0)
 D.append(np.bincount(inverse,weights=base_correct-main_correct,minlength=G))
D=np.array(D).T
counts=np.bincount(inverse,minlength=G);obs=D.sum(axis=0)/N
rng=np.random.default_rng(20261017);B=20000;exceed=np.zeros(len(models),int)
for start in range(0,B,200):
 signs=rng.integers(0,2,size=(min(200,B-start),G),dtype=np.int8)*2-1
 stat=signs@D/N
 exceed+=(np.abs(stat)>=np.abs(obs)-1e-15).sum(axis=0)
pvals=(exceed+1)/(B+1)
order=np.argsort(pvals);adjusted=np.empty_like(pvals);running=0
for rank,index in enumerate(order):
 running=max(running,(len(pvals)-rank)*pvals[index]);adjusted[index]=min(1,running)
boot=[]
for start in range(0,3000,100):
 weights=rng.multinomial(G,np.full(G,1/G),size=100)
 boot.append((weights@D)/(weights@counts)[:,None])
boot=np.concatenate(boot);ci=np.quantile(boot,[.025,.975],axis=0)
paired=pd.DataFrame([dict(model=model,baseline_minus_main_accuracy=obs[i],domain_bootstrap_ci95_low=ci[0,i],domain_bootstrap_ci95_high=ci[1,i],domain_signflip_p=pvals[i],holm_adjusted_p=adjusted[i],test_domains=G) for i,model in enumerate(models)])
paired.to_csv(OUT/'paired_domain_comparisons.csv',index=False)
# Summary table for reading.
pretty=[]
for model in ['RF_full','RF_no_scheme','RF_no_scheme_no_ghana','XGB_full','XGB_no_scheme','XGB_no_scheme_no_ghana','CNN','BiLSTM','PhishGuard_GH']:
 row={'model':model}
 for metric in ['accuracy','f1_score','auc_roc']:
  r=s[(s.model==model)&(s.metric==metric)].iloc[0];scale=1 if metric=='auc_roc' else 100
  row[metric]=f'{r["mean"]*scale:.4f} +/- {r["sample_sd"]*scale:.4f}'
 pretty.append(row)
table=pd.DataFrame(pretty);table.to_csv(OUT/'readable_comparison.csv',index=False)
notes='''Five-seed tuned baseline comparison, version 3\n\nVerified all 40 baseline model results against saved CSV probabilities; validated URL/label ordering and partition fingerprints against main-model runs. Test set: 7,653 URLs. Positive class combines phishing, malware and defacement. Threshold 0.5.\n\nSelection: modest predefined candidate grids, validation average precision, tuning seed 42; early stopping uses validation loss. Selected settings frozen across seeds. These are limited searches, not exhaustive optimization. Main-model architecture was fixed and was not tuned by this baseline search.\n\nMean +/- sample SD and Student-t intervals use five training seeds conditional on the fixed split and chosen configurations. Reusing the selected tuning model for seed 42 is disclosed; selection variability is not included. XGBoost uses full row/column sampling, and all five saved predictions for each variant are identical: zero seed SD does not mean zero generalization uncertainty.\n\nPaired inference: baseline-minus-main accuracy averaged over seeds for each URL, then summed within PSL registrable-domain groups. Domain sign-flip test uses 20,000 random sign assignments, swapping the sign of the entire domain difference vector, and Holm correction across eight baseline-versus-main comparisons. This is conditional on trained models and assumes independent, exchangeable/symmetric domain-level differences under the null. 3,000 paired domain-bootstrap replicates resample complete domains, retain all URLs and the same five fitted seeds, and compute percentile intervals for mean-seed accuracy differences. These do not include retraining or tuning uncertainty. Do not pool seeds as independent test URLs. PSL snapshot/version may differ from original Windows environment; report the reconstructed domain count.\n\nExact McNemar per seed is supplied descriptively, but URL independence may fail within domains; domain-aware comparisons are the primary analysis. P-values are not probabilities that a model is better. No significance conclusion for F1 or AUC is made from accuracy tests.\n\nAblations are separately retuned on validation, so differences compare feature-availability pipelines, not a single fixed configuration's isolated feature effect. no_scheme drops is_https only: lexical URL lengths/slashes can still encode scheme information. no_scheme_no_ghana also removes is_trusted_gh_tld, brand_injected and brand_on_free_tld; it does not remove all possible geographic cues. Neural and numeric inputs remain different; same partitions alone do not make representations identical.\n\nResults:\n'''+table.to_string(index=False)+'\n\nPaired domain comparisons:\n'+paired.to_string(index=False)
(OUT/'methods_and_limitations.txt').write_text(notes)
print(table.to_string(index=False));print(paired.to_string(index=False));print('Domain count:',G)
