"""Reviewer 2 category audit. No training; inputs are never modified."""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_SHA = 'd83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'
PART_SHA = {
 'train': '28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
 'validation': '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
 'test': '22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
SEEDS = [42, 43, 44, 45, 46]

def sha(p):
 h = hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
 return h.hexdigest()

def canonical(u):
 u = str(u).strip()
 for prefix in ('https://','http://'):
  if u.lower().startswith(prefix):
   u = u[len(prefix):]; break
 return u.lower().strip()

def encoded(u):
 return tuple([ord(c)-31 if 32 <= ord(c) <= 126 else 96 for c in canonical(u)[:200]])

def read(p, cols):
 d = pd.read_csv(p, keep_default_na=False)
 if not set(cols).issubset(d.columns): raise ValueError(f'{p}: missing {cols}')
 return d

def main():
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--check-only', action='store_true')
 args = ap.parse_args()
 root = Path(__file__).resolve().parent
 source = root/'malicious_phish.csv'
 if sha(source) != SOURCE_SHA: raise ValueError('Original source fingerprint differs')
 parts = {}
 for name, expected in PART_SHA.items():
  path = root/'dataset_fix_outputs_v3'/f'{name}.csv'
  if sha(path) != expected: raise ValueError(f'{name}: frozen fingerprint differs')
  parts[name] = read(path, ['url','label'])
  if not parts[name].label.isin([0,1]).all(): raise ValueError('Invalid partition labels')
 # Full-source category sets, not first-match assignment.
 src = read(source, ['url','type'])
 src['type'] = src['type'].astype(str).str.strip().str.lower()
 if not src['type'].isin(['benign','phishing','malware','defacement']).all():
  raise ValueError('Unrecognised source category')
 source_map = defaultdict(set)
 for u,t in zip(src.url,src['type']): source_map[canonical(u)].add(t)
 # Check all source labels sharing the actual normalized/truncated encoding.
 wanted = {encoded(u) for d in parts.values() for u in d.url}
 encoded_labels = defaultdict(set)
 for u,t in zip(src.url,src['type']):
  k = encoded(u)
  if k in wanted: encoded_labels[k].add(int(t != 'benign'))
 aug = read(root/'modern_legit_augmentation.csv', ['url','label'])
 if not aug.label.eq(0).all(): raise ValueError('Generated labels must be benign')
 generated = {canonical(u) for u in aug.url}
 if generated & source_map.keys(): raise ValueError('Generated/source provenance overlap')
 assigned = []
 for name,d in parts.items():
  cats=[]
  for u,y in zip(d.url,d.label):
   k=canonical(u); labels=encoded_labels.get(encoded(u),set())
   if len(labels)>1 or (labels and int(y) not in labels):
    raise ValueError(f'{name}: conflicting encoded source labels')
   if k in generated:
    if y != 0: raise ValueError('Generated malicious label')
    cat='generated_benign'
   else:
    ts=source_map.get(k,set())
    if not ts: raise ValueError(f'{name}: URL absent from source and augmentation')
    if {int(t!='benign') for t in ts} != {int(y)}:
     raise ValueError(f'{name}: source-label conflict')
    cat=next(iter(ts)) if len(ts)==1 else 'multiple_malicious_categories'
   cats.append(cat)
  d['category']=cats
  for (cat,label),n in d.groupby(['category','label']).size().items():
   assigned.append({'partition':name,'category':cat,'label':int(label),'rows':int(n)})
 test=parts['test']; probs={}; pred_hashes={}
 for seed in SEEDS:
  path=root/'corrected_results_v3'/f'seed_{seed}'/'predictions.csv'
  pred=read(path,['url','true_label','pred_label','probability'])
  # Exact URL membership and labels checked for every full-snapshot row.
  snapshot=pd.concat([parts[n][['url','label']] for n in PART_SHA],ignore_index=True)
  if snapshot.url.duplicated().any() or pred.url.duplicated().any():
   raise ValueError('Duplicate exact URL; alignment needs review')
  if set(pred.url)!=set(snapshot.url): raise ValueError(f'Seed {seed}: snapshot membership differs')
  aligned=pred.set_index('url').loc[snapshot.url]
  if not np.array_equal(aligned.true_label.to_numpy(),snapshot.label.to_numpy()):
   raise ValueError(f'Seed {seed}: full snapshot labels differ')
  allp=pd.to_numeric(aligned.probability,errors='raise').to_numpy(float)
  if not np.isfinite(allp).all() or ((allp<0)|(allp>1)).any(): raise ValueError('Invalid scores')
  if not np.array_equal((allp>=.5).astype(int),aligned.pred_label.to_numpy()):
   raise ValueError(f'Seed {seed}: saved verdicts differ from threshold 0.5')
  p=pred.set_index('url').loc[test.url,'probability'].to_numpy(float)
  saved=np.load(path.parent/'y_prob.npy',allow_pickle=False).reshape(-1)
  labels=np.load(path.parent/'y_test.npy',allow_pickle=False).reshape(-1)
  if not np.array_equal(labels,test.label.to_numpy()) or p.shape!=saved.shape:
   raise ValueError(f'Seed {seed}: saved test alignment differs')
  if not np.isfinite(saved).all() or ((saved < 0) | (saved > 1)).any():
   raise ValueError(f'Seed {seed}: invalid saved test scores')
  if not np.allclose(p, saved, rtol=0, atol=1e-6):
   raise ValueError(f'Seed {seed}: test scores differ beyond numerical tolerance')
  if not np.array_equal(p >= 0.5, saved >= 0.5):
   raise ValueError(f'Seed {seed}: test verdicts differ')
  probs[seed] = saved.astype(float)
  pred_hashes[str(seed)] = sha(path)
 print('CHECK PASSED: frozen partitions, original source, category mapping and all five prediction files.')
 counts=pd.DataFrame(assigned)
 print('\nPARTITION CATEGORY COUNTS\n'+counts.to_string(index=False))
 if args.check_only:
  print('\nNo reports created. No training or inference performed.');return
 rows=[]
 for seed,p in probs.items():
  for cat in sorted(test.category.unique()):
   mask=test.category.eq(cat).to_numpy(); y=test.label.to_numpy()[mask]; pred=p[mask]>=.5
   n=int(mask.sum()); positive=int(y[0]); detected=int(pred.sum())
   rows.append({'seed':seed,'category':cat,'label':positive,'rows':n,
    'predicted_malicious':detected,'predicted_benign':n-detected,
    'measure':'detection_recall' if positive else 'false_positive_rate',
    'rate':detected/n})
 per=pd.DataFrame(rows);summary=[]
 for (cat,measure),g in per.groupby(['category','measure']):
  summary.append({'category':cat,'measure':measure,'test_rows':int(g.rows.iloc[0]),
   'seeds':len(g),'mean_percent':100*g.rate.mean(),
   'sample_sd_percentage_points':100*g.rate.std(ddof=1),
   'min_percent':100*g.rate.min(),'max_percent':100*g.rate.max()})
 summary=pd.DataFrame(summary)
 out=root/'category_evaluation_v3'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
 out.mkdir(parents=True,exist_ok=False)
 counts.to_csv(out/'partition_category_counts.csv',index=False)
 per.to_csv(out/'test_category_per_seed.csv',index=False)
 summary.to_csv(out/'test_category_summary.csv',index=False)
 test[['url','label','category']].to_csv(out/'test_category_membership.csv',index=False)
 report={'source_sha256':SOURCE_SHA,'partition_sha256':PART_SHA,
  'prediction_sha256':pred_hashes,'script_sha256':sha(Path(__file__)),
  'seeds':SEEDS,'threshold':.5,'training_performed':False,
  'score_source':'y_prob.npy, aligned using frozen test labels and URL-matched full predictions',
  'csv_score_absolute_tolerance':1e-6,
  'csv_npy_threshold_verdicts_must_match':True,
  'category_assignment':'all canonical source matches; same-binary-class multiple categories reported separately',
  'limitations':['Inherited corpus categories, not independently verified threats.',
   'Category recall is sensitivity versus benign, not multiclass classification accuracy.',
   'Mean and sample SD describe training seeds on a fixed test set, not population confidence intervals.',
   'Only original v3 models evaluated; broader-data models are a separate experiment.',
   'No AUC is defined for a single-label category subset.']}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('\nFIVE-SEED TEST CATEGORY RESULTS\n'+summary.to_string(index=False))
 print('\nSaved to:',out)
 print('No training or inference performed. Source datasets and results unchanged.')

if __name__=='__main__':main()
