"""Read-only, exploratory M2 audit of saved v3 predictions. No training or inference."""
import argparse,csv,hashlib,ipaddress,json,math
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import tldextract
PART_SHA={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
TRANCO_SHA='3a2f5b2a5eec2049cdf5de297507a6611789c2d0da12d2da6cfbbf461f8a76d8'
SOURCE_SHA='d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'
SEEDS=range(42,47)
LIMITS=['Exploratory analysis after inspecting earlier tests; no confirmatory hypothesis test.',
'Historical Tranco rank is a popularity proxy, not measured traffic, legitimacy or campaign prevalence; historical daily-list ID unavailable.',
'The list also supplied augmentation domains, so provenance and rank are dependent; generated URLs are reported separately.',
'Absent-from-list does not mean unpopular or malicious. No causal popularity effect is established.',
'Length, source category, domain and formatting can remain confounded; joint strata are descriptive and may be sparse.',
'www, slash and appended-path changes may alter the resource; verdict flips are not transformed accuracy or real attack success.',
'Seed SDs concern repeated fits on fixed URLs, not independent URLs/domains or traffic uncertainty.',
'No original dataset, model, feature file or result is modified.']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canonical(u):
 s=str(u).strip()
 for prefix in ['https://','http://']:
  if s.lower().startswith(prefix):s=s[len(prefix):];break
 return s.lower().strip()
def parse(u):
 text=str(u).strip()
 p=urlsplit('https:'+text if text.startswith('//') else text if '://' in text else 'https://'+text)
 host=(p.hostname or '').lower().rstrip('.')
 if not host:raise ValueError('Empty hostname')
 return p,host

def rank_host(host):
 return host[4:] if host.startswith('www.') else host

def rank_band(rank):
 if rank is None:return 'not_listed'
 if rank<=1000:return '1-1000'
 if rank<=10000:return '1001-10000'
 if rank<=100000:return '10001-100000'
 return '100001-1000000'

def length_band(n):return '0-49' if n<50 else '50-99' if n<100 else '100-200' if n<=200 else 'over_200_truncated'
def rates(y,q):
 benign=y==0;mal=y==1
 return {'rows':len(y),'benign':int(benign.sum()),'malicious':int(mal.sum()),
 'false_positives':int(q[benign].sum()),'true_positives':int(q[mal].sum()),
 'fpr':float(q[benign].mean()) if benign.any() else None,
 'recall':float(q[mal].mean()) if mal.any() else None,
 'accuracy':float((q==y).mean())}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
 ap.add_argument('--tranco',type=Path,help='Exact recovered CSV; default ROOT/top-1m.csv. LFS pointer is rejected.')
 mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--check-only',action='store_true');mode.add_argument('--run',action='store_true')
 args=ap.parse_args();root=args.root.resolve();tranco=(args.tranco or root/'top-1m.csv').resolve()
 files={}
 for part,expected in PART_SHA.items():
  p=root/'dataset_fix_outputs_v3'/f'{part}.csv';files[part]=p
  if sha(p)!=expected:raise ValueError('Frozen partition differs: '+part)
 if sha(tranco)!=TRANCO_SHA:raise ValueError('Tranco bytes differ from the recovered historical CSV; do not substitute a fresh list or an LFS pointer.')
 files['tranco']=tranco;source=root/'malicious_phish.csv';files['source']=source
 if sha(source)!=SOURCE_SHA:raise ValueError('Original source corpus differs')
 test=pd.read_csv(files['test'],keep_default_na=False)
 augpath=root/'modern_legit_augmentation.csv';files['augmentation']=augpath
 aug=pd.read_csv(augpath,keep_default_na=False)
 if not aug.label.eq(0).all() or len(aug)!=2244:raise ValueError('Generated provenance differs')
 generated={canonical(u) for u in aug.url};wanted={canonical(u) for u in test.url};cats=defaultdict(set)
 with source.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   k=canonical(row['url'])
   if k in wanted:cats[k].add(row['type'].strip().lower())
 category=[];provenance=[]
 for u,y in zip(test.url,test.label):
  k=canonical(u)
  if k in generated:
   if y!=0 or cats.get(k):raise ValueError('Generated/source label conflict')
   category.append('generated_benign');provenance.append('generated_benign')
  else:
   types=cats.get(k,set())
   if not types or not types<={'benign','phishing','malware','defacement'} or {int(t!='benign') for t in types}!={int(y)}:raise ValueError('Source provenance conflict')
   category.append(next(iter(types)) if len(types)==1 else 'multiple_malicious_categories');provenance.append('original_source')
 test['category']=category;test['provenance']=provenance
 extractor=tldextract.TLDExtract(suffix_list_urls=(),include_psl_private_domains=True,cache_dir=None)
 domains=[];hosts=[];forms=[];www=[];lengths=[]
 for u in test.url:
  p,host=parse(u);hosts.append(host);www.append(host.startswith('www.'));lengths.append(len(canonical(u)))
  try:ipaddress.ip_address(host);domain=host
  except ValueError:
   d=extractor(host);domain='.'.join(x for x in [d.domain,d.suffix] if x)
  domains.append(domain);forms.append('root_empty' if p.path=='' else 'root_slash' if p.path=='/' else 'non_root_path')
 rankmap={};seen_ranks=set()
 with tranco.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.reader(f):
   if len(row)!=2:raise ValueError('Expected rank/domain Tranco CSV')
   rank=int(row[0]);host=rank_host(row[1].strip().lower().rstrip('.'))
   if not 1<=rank<=1000000 or rank in seen_ranks:raise ValueError('Invalid Tranco ranks')
   seen_ranks.add(rank);rankmap[host]=min(rank,rankmap.get(host,rank))
 if len(seen_ranks)!=1000000:raise ValueError('Historical list is not one million rows')
 test['hostname']=hosts;test['domain_group']=domains;test['has_www']=www;test['path_form']=forms
 test['normalized_length']=lengths;test['length_group']=[length_band(x) for x in lengths]
 # Keep two explicitly different proxies; do not inherit a parent's rank for the exact-host view.
 test['exact_host_rank']=[rankmap.get(rank_host(h)) for h in hosts]
 test['domain_rank']=[rankmap.get(rank_host(d)) for d in domains]
 for field in ['exact_host','domain']:
  test[field+'_rank_band']=[rank_band(None if pd.isna(r) else int(r)) for r in test[field+'_rank']]
  test[field+'_listed']=np.where(test[field+'_rank'].isna(),'not_listed','listed')
 predictions={};verification={}
 for seed in SEEDS:
  folder=root/'corrected_results_v3'/f'seed_{seed}';p=folder/'test_predictions.csv';files[f'seed_{seed}_csv']=p
  frame=pd.read_csv(p,keep_default_na=False)
  if not frame[['url','true_label']].rename(columns={'true_label':'label'}).equals(test[['url','label']]):raise ValueError('Prediction alignment differs')
  scores=np.load(folder/'y_prob.npy',allow_pickle=False).reshape(-1);labels=np.load(folder/'y_test.npy',allow_pickle=False).reshape(-1)
  for name in ['y_prob.npy','y_test.npy']:files[f'seed_{seed}_{name}']=folder/name
  if scores.shape!=(len(test),) or not np.array_equal(labels,test.label.to_numpy()) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('Saved score alignment/range differs')
  delta=float(np.abs(scores-frame.probability.to_numpy()).max())
  if delta>1e-6 or not np.array_equal(scores>=.5,frame.pred_label.to_numpy()==1):raise ValueError('Saved CSV/NPY verdicts differ')
  predictions[seed]=scores>=.5;verification[str(seed)]={'csv_npy_max_difference':delta,'verdict_differences':0}
 print(f'CHECK PASSED: {len(test):,} aligned test URLs; five saved runs; exact historical Tranco CSV; generated/source provenance.')
 if args.check_only:print('No model loaded, inference, training or output folders.');return
 out=root/'m2_popularity_audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
 rows=[];joint=[];y=test.label.to_numpy()
 axes=['domain_listed','domain_rank_band','exact_host_listed','exact_host_rank_band','length_group','path_form','has_www']
 for seed,q in predictions.items():
  for prov in ['all','original_source','generated_benign']:
   base=np.ones(len(test),bool) if prov=='all' else test.provenance.eq(prov).to_numpy()
   for axis in axes:
    for value in sorted(test[axis].unique(),key=str):
     mask=base&test[axis].eq(value).to_numpy()
     if not mask.any():continue
     rows.append(dict(seed=seed,provenance=prov,axis=axis,group=str(value),domains=int(test.loc[mask,'domain_group'].nunique()),**rates(y[mask],q[mask])))
   for (length,form,has_www,listed),indices in test.loc[base].groupby(['length_group','path_form','has_www','domain_listed']).groups.items():
    idx=np.array(list(indices));joint.append(dict(seed=seed,provenance=prov,length_group=length,path_form=form,has_www=has_www,domain_listed=listed,domains=int(test.loc[idx,'domain_group'].nunique()),**rates(y[idx],q[idx])))
 per=pd.DataFrame(rows);per.to_csv(out/'per_seed_strata.csv',index=False)
 pd.DataFrame(joint).to_csv(out/'joint_length_format_rank_strata.csv',index=False)
 summaries=[]
 for keys,g in per.groupby(['provenance','axis','group'],dropna=False):
  record=dict(zip(['provenance','axis','group'],keys));record.update({k:int(g[k].iloc[0]) for k in ['rows','domains','benign','malicious']})
  for metric in ['fpr','recall','accuracy']:
   values=g[metric].dropna();record[metric+'_mean_percent']=100*float(values.mean()) if len(values) else None;record[metric+'_sample_sd_pp']=100*float(values.std(ddof=1)) if len(values)>1 else None
  summaries.append(record)
 summary=pd.DataFrame(summaries);summary.to_csv(out/'five_seed_strata_summary.csv',index=False)
 test.to_csv(out/'test_popularity_membership.csv',index=False)
 support=test.groupby(['provenance','category','domain_listed','length_group','path_form','has_www']).agg(rows=('url','size'),domains=('domain_group','nunique')).reset_index()
 support.to_csv(out/'source_category_joint_support.csv',index=False)
 category_rows=[];standardized=[];support_rows=[]
 for seed,q in predictions.items():
  for (category,listed),idx in test.groupby(['category','domain_listed']).groups.items():
   idx=np.array(list(idx));category_rows.append(dict(seed=seed,category=category,domain_listed=listed,**rates(y[idx],q[idx])))
  for label in [0,1]:
   base=test[(test.provenance=='original_source')&(test.label==label)]
   cells=[]
   for keys,frame in base.groupby(['category','length_group','path_form','has_www']):
    if set(frame.domain_listed)!={'listed','not_listed'}:continue
    a=frame[frame.domain_listed=='listed'];b=frame[frame.domain_listed=='not_listed'];n=len(frame)
    cells.append((n,float(q[a.index].mean()),float(q[b.index].mean())))
    if seed==42:support_rows.append(dict(label=label,category=keys[0],length_group=keys[1],path_form=keys[2],has_www=keys[3],listed_rows=len(a),not_listed_rows=len(b),listed_domains=a.domain_group.nunique(),not_listed_domains=b.domain_group.nunique()))
   total=sum(v[0] for v in cells)
   standardized.append({'seed':seed,'label':label,'measure':'FPR' if label==0 else 'recall','common_strata':len(cells),'rows_in_common_strata':total,'original_source_class_rows':len(base),'listed_standardized_percent':100*sum(n*a for n,a,b in cells)/total if total else None,'not_listed_standardized_percent':100*sum(n*b for n,a,b in cells)/total if total else None})
 pd.DataFrame(category_rows).to_csv(out/'per_seed_category_rank.csv',index=False)
 pd.DataFrame(standardized).to_csv(out/'common_support_descriptive_standardization.csv',index=False)
 pd.DataFrame(support_rows).to_csv(out/'common_support_counts.csv',index=False)

 manifest={k:{'path':str(p),'sha256':sha(p)} for k,p in files.items()}
 report={'seeds':list(SEEDS),'threshold':.5,'rows':len(test),'domains':int(test.domain_group.nunique()),'partition_sha256':PART_SHA,
 'rank_definition':{'exact_host':'Case-insensitive hostname, trailing dot stripped, one leading www removed; exact Tranco match only.',
 'domain':'Public-suffix registrable domain, private suffixes included; rank by exact Tranco match of that domain; IP by address.'},
 'common_support_method':'Original-source URLs only, separately by binary class; exact strata for source category, length band, path form and www indicator. Keep strata containing both rank groups, with at least one URL in each. Weight stratum-specific positive prediction rates by pooled stratum URL count. Sparse cells and unmeasured confounding remain; not a causal effect.',
 'tldextract_version':tldextract.__version__,'psl':'bundled offline snapshot; no network access; private suffixes included',
 'inputs':manifest,'saved_score_checks':verification,'script_sha256':sha(Path(__file__)),
 'training_performed':False,'inference_performed':False,'limitations':LIMITS}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 print(summary[(summary.axis=='domain_listed')].to_string(index=False));print('Saved to:',out)
 print('Descriptive associations only. Frozen inputs and results unchanged; no training or inference.')
if __name__=='__main__':main()
