"""Evaluate frozen v3 rule annotations; no training, inference or input edits."""
import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd

PART_SHA = {'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
RULE_SIGNATURE='52660660af6e7208012ff1168c35ead3a2b423764ceb6bdd950a919b6ca3a04d'
RULES=['free_tld','ip_hostname','shortener','brand_on_free_tld','at_symbol','non_https']
SEEDS=[42,43,44,45,46]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def portable_ast(node):
 # Python 3.12 added empty type_params fields absent in Python 3.11.
 # Ignore only that empty metadata; retain every actual rule expression.
 if isinstance(node, ast.AST):
  return [type(node).__name__, [[key, portable_ast(value)]
   for key, value in ast.iter_fields(node)
   if not (key == 'type_params' and value == [])]]
 if isinstance(node, list):return [portable_ast(value) for value in node]
 return node

def rule_digest(module):
 payload=json.dumps(portable_ast(module),ensure_ascii=True,separators=(',',':'))
 return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def load_rules(path):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'))
 nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'_BRANDS','_FREE_TLDS','_SHORTENERS'} for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='build_rationale')]
 module=ast.Module(body=nodes,type_ignores=[])
 signature=rule_digest(module)
 if signature!=RULE_SIGNATURE:raise ValueError(f'Rule code differs from reviewed v3 attachment: {path.name}. Expected {RULE_SIGNATURE}; found {signature}. Do not edit training code; send the current file for review.')
 ns={'re':re,'urlparse':urlparse}
 # Only reviewed literal lists and rationale function; never import/run pipeline.
 exec(compile(module,str(path),'exec'),ns)
 return ns

def indicators(url,ns):
 low=url.lower()
 try:host=urlparse(url).netloc or ''
 except Exception:host=''
 values=[any(host.endswith(t) for t in ns['_FREE_TLDS']),
  bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',host)),
  any(s in low for s in ns['_SHORTENERS']),
  any(b in low for b in ns['_BRANDS']) and any(host.endswith(t) for t in ns['_FREE_TLDS']),
  '@' in url,not low.startswith('https')]
 labels=[]
 if values[0]:labels.append('Free TLD ('+next(t for t in ns['_FREE_TLDS'] if host.endswith(t))+')')
 if values[1]:labels.append('IP address hostname')
 if values[2]:labels.append('Shortener ('+next(s for s in ns['_SHORTENERS'] if s in low)+')')
 if values[3]:labels.append("Brand '"+next(b for b in ns['_BRANDS'] if b in low)+"' on suspicious domain")
 if values[4]:labels.append('@ symbol in URL')
 if values[5]:labels.append('Non-HTTPS')
 displayed=np.zeros(6,dtype=bool)
 displayed[np.flatnonzero(values)[:3]]=True
 secondary='; '.join(labels[:3]) if labels else 'No rule-based indicators'
 actual=ns['build_rationale'](url,'PHISHING',.7)
 if actual.split('[SECONDARY] ',1)[1] != secondary+'.':raise ValueError('Indicator reproduction differs from original function')
 return values,displayed,secondary,not bool(host)

def divide(a,b):return float(a/b) if b else None

def stats(y,hit):
 y=np.asarray(y,dtype=int);hit=np.asarray(hit,dtype=bool)
 tp=int(((y==1)&hit).sum());fp=int(((y==0)&hit).sum())
 fn=int(((y==1)&~hit).sum());tn=int(((y==0)&~hit).sum())
 return {'rows':len(y),'triggered':tp+fp,'coverage':divide(tp+fp,len(y)),
  'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision_on_this_corpus':divide(tp,tp+fp),
  'malicious_trigger_rate_recall':divide(tp,tp+fn),'benign_trigger_rate_fpr':divide(fp,fp+tn)}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
 root=Path(__file__).resolve().parent
 paths=[root/'model_hash_v3.py',root/'model_hash_v3(1).py']
 path=next((p for p in paths if p.exists()),paths[0]);ns=load_rules(path)
 parts={};input_hash={str(path.name):sha(path)}
 for name,digest in PART_SHA.items():
  p=root/'dataset_fix_outputs_v3'/f'{name}.csv'
  if sha(p)!=digest:raise ValueError(f'{name}: frozen partition differs')
  d=pd.read_csv(p,keep_default_na=False)
  if not {'url','label'}.issubset(d.columns) or not d.label.isin([0,1]).all():raise ValueError('Invalid partition schema/labels')
  parts[name]=d
 snapshot=pd.concat([parts[n][['url','label']] for n in PART_SHA],ignore_index=True)
 if snapshot.url.duplicated().any():raise ValueError('Duplicate exact URLs')
 probs={}
 for seed in SEEDS:
  folder=root/'corrected_results_v3'/f'seed_{seed}';p=folder/'predictions.csv'
  pred=pd.read_csv(p,keep_default_na=False)
  if not {'url','true_label','probability','pred_label'}.issubset(pred):raise ValueError('Prediction schema differs')
  if pred.url.duplicated().any() or set(pred.url)!=set(snapshot.url):raise ValueError(f'Seed {seed}: snapshot mismatch')
  aligned=pred.set_index('url').loc[snapshot.url]
  if not np.array_equal(aligned.true_label,snapshot.label):raise ValueError('Snapshot label mismatch')
  allp=pd.to_numeric(aligned.probability,errors='raise').to_numpy(float)
  if not np.isfinite(allp).all() or ((allp<0)|(allp>1)).any():raise ValueError('Invalid prediction scores')
  if not np.array_equal(allp>=.5,aligned.pred_label.to_numpy()):raise ValueError('Saved verdict mismatch')
  scores=pred.set_index('url').loc[parts['test'].url,'probability'].to_numpy(float)
  saved=np.load(folder/'y_prob.npy',allow_pickle=False).reshape(-1)
  labels=np.load(folder/'y_test.npy',allow_pickle=False).reshape(-1)
  if not np.array_equal(labels,parts['test'].label) or scores.shape!=saved.shape:raise ValueError('Test alignment mismatch')
  if not np.isfinite(saved).all() or ((saved<0)|(saved>1)).any():raise ValueError('Invalid saved test scores')
  if not np.allclose(scores,saved,rtol=0,atol=1e-6) or not np.array_equal(scores>=.5,saved>=.5):raise ValueError('Test score/verdict mismatch')
  probs[seed]=saved.astype(float)
  for f in [p,folder/'y_prob.npy',folder/'y_test.npy']:input_hash[str(f.relative_to(root))]=sha(f)
 data={};metric_rows=[];diag=[]
 for name in ['validation','test']:
  d=parts[name];rows=[indicators(u,ns) for u in d.url]
  raw=np.array([r[0] for r in rows],dtype=bool);shown=np.array([r[1] for r in rows],dtype=bool)
  data[name]=(raw,shown,[r[2] for r in rows])
  for mode,mat in [('all_triggered',raw),('displayed_first_three',shown)]:
   for j,rule in enumerate(RULES):metric_rows.append({'partition':name,'mode':mode,'rule':rule,**stats(d.label,mat[:,j])})
   for rule,hit in [('any_rule',mat.any(axis=1)),('any_except_non_https',mat[:,:5].any(axis=1))]:
    metric_rows.append({'partition':name,'mode':mode,'rule':rule,**stats(d.label,hit)})
  schemeless=~d.url.str.match(r'^[A-Za-z][A-Za-z0-9+.-]*://').to_numpy()
  empty=np.array([r[3] for r in rows]);labels=d.label.to_numpy()
  for group,mask in [('all',np.ones(len(d),bool)),('benign',labels==0),('malicious',labels==1),('schemeless',schemeless)]:
   diag.append({'partition':name,'group':group,'rows':int(mask.sum()),'empty_netloc':int((mask&empty).sum()),'non_https_triggered':int((mask&raw[:,5]).sum())})
 print('CHECK PASSED: v3 rule fingerprint, partitions, five prediction runs and exact rationale reproduction.')
 if args.check_only:
  print('No reports created. No training or inference performed.');return
 test=parts['test'];y=test.label.to_numpy();raw,shown,secondary=data['test'];agreement=[];examples=[]
 for mode,mat in [('all_triggered',raw),('displayed_first_three',shown)]:
  hits={r:mat[:,j] for j,r in enumerate(RULES)}
  hits['any_rule']=mat.any(axis=1);hits['any_except_non_https']=mat[:,:5].any(axis=1)
  for rule,hit in hits.items():
   for seed,p in probs.items():
    model=p>=.5
    agreement.append({'seed':seed,'mode':mode,'rule':rule,
      'rule_triggered_model_malicious':int((hit&model).sum()),
      'rule_triggered_model_benign':int((hit&~model).sum()),
      'rule_not_triggered_model_malicious':int((~hit&model).sum()),
      'rule_not_triggered_model_benign':int((~hit&~model).sum()),
      'model_false_negatives_with_rule':int(((y==1)&~model&hit).sum()),
      'model_false_positives_with_rule':int(((y==0)&model&hit).sum())})
   # Deterministic illustrative selection, not evidence of analyst usefulness.
   for outcome,mask in [('TP',(y==1)&hit),('FP',(y==0)&hit),('FN',(y==1)&~hit),('TN',(y==0)&~hit)]:
    indices=np.flatnonzero(mask)
    indices=sorted(indices,key=lambda i:hashlib.sha256(test.url.iloc[i].encode()).hexdigest())[:2]
    for i in indices:
     p=float(probs[42][i]);verdict='PHISHING' if p>=.5 else 'LEGITIMATE'
     examples.append({'mode':mode,'rule':rule,'indicator_outcome':outcome,'test_row_index':int(i),
      'url':test.url.iloc[i],'true_label':int(y[i]),'rule_triggered':bool(hit[i]),
      'seed':42,'model_score_malicious':p,'model_predicted_label':int(p>=.5),
      'model_correct':bool((p>=.5)==y[i]),'displayed_secondary':secondary[i],
      'original_rationale':ns['build_rationale'](test.url.iloc[i],verdict,p)})
 out=root/'rule_evaluation_v3'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
 metrics=pd.DataFrame(metric_rows);metrics.to_csv(out/'rule_metrics.csv',index=False)
 pd.DataFrame(agreement).to_csv(out/'rule_model_agreement_per_seed.csv',index=False)
 pd.DataFrame(diag).to_csv(out/'parsing_diagnostics.csv',index=False)
 pd.DataFrame(examples).to_csv(out/'worked_examples.csv',index=False)
 members=test[['url','label']].copy()
 for j,r in enumerate(RULES):members['triggered_'+r]=raw[:,j];members['displayed_'+r]=shown[:,j]
 members['displayed_secondary']=secondary
 members.to_csv(out/'test_rule_membership.csv',index=False)
 report={'partition_sha256':PART_SHA,'input_sha256':input_hash,'script_sha256':sha(Path(__file__)),
  'rule_ast_sha256':RULE_SIGNATURE,'seeds':SEEDS,'threshold':.5,
  'rules':{k:ns[k] for k in ['_BRANDS','_FREE_TLDS','_SHORTENERS']},
  'test_rows':len(test),'training_performed':False,'inference_performed':False,
  'score_source':'saved y_prob.npy; CSV tolerance 1e-6, verdict agreement required',
  'rules_modified':False,'display_limit':3,
  'limitations':['Rules are independent annotations, not explanations of neural computation.',
   'Treating a trigger as a positive flag is a diagnostic; rules did not determine stored verdicts.',
   'No threshold, combined rule or model was tuned on these results.',
   'All-rule and non-HTTPS-excluded summaries are descriptive, not validated replacement detectors.',
   'No analyst user study, causal explanation or incremental benefit from combining scores and rules was tested.',
   'Shortener and brand matches use substrings; IP regex is not IP validation; non-HTTPS includes missing schemes.',
   'Original urlparse(netloc) behaviour is preserved, including empty hostnames for many schemeless URLs.',
   'Positives mix phishing, malware and defacement. Corpus precision is not deployment precision.',
   'Examples are deterministic hash-selected cases, not representative estimates or live-verified labels.',
   'Original rationale uses PHISHING and probability wording; these labels do not establish phishing-only detection or calibrated confidence.']}
 (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 display=metrics[(metrics.partition=='test')&(metrics['mode']=='all_triggered')].copy()
 cols=['coverage','precision_on_this_corpus','malicious_trigger_rate_recall','benign_trigger_rate_fpr']
 display[cols]=display[cols]*100
 print('\nTEST RULE METRICS (percent; undefined values shown as NaN)')
 print(display[['rule','triggered','tp','fp']+cols].to_string(index=False))
 print('\nPARSING DIAGNOSTICS\n'+pd.DataFrame(diag).to_string(index=False))
 print('\nSaved to:',out)
 print('Original files preserved. No training, model inference or network access.')

if __name__=='__main__':main()
