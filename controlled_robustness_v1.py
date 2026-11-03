"""Local controlled-fixture score-guided robustness benchmark. No training or live-site access.
Synthetic fixture labels are not verified real phishing labels. This is not external-test evidence.
"""
import argparse,ast,csv,hashlib,http.client,json,math,os,random,threading
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit,parse_qs,quote,unquote,urlparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='3';os.environ['TF_ENABLE_ONEDNN_OPTS']='0';os.environ['CUDA_VISIBLE_DEVICES']='-1'
PART_HASH={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
LIMITATIONS=['Constructed .test fixtures and simulated positive/negative behaviours; not verified real phishing pages or natural traffic.','Byte-identical local HTTP responses validate only this explicitly alias-aware, script-free fixture application.','Logical hosts are simulated tenants; this does not demonstrate abuse of an actual legitimate provider.','No live URL or credential-collection functionality is accessed; there is no demonstrated preservation of a real malicious payload.','A four-round best-so-far score-guided search is evaluated under a maximum 16 new queries per case per model; not a complete adaptive-attacker evaluation.','All labels and transformations are fixed before inference. Results must remain separate from corpus test metrics and external validation.','No deployment, local Ghanaian effectiveness, human-user behaviour or universal robustness claim is supported.']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def encode(u):
 t=u.strip()
 for s in ['https://','http://']:
  if t.lower().startswith(s):t=t[len(s):];break
 t=t.lower().strip()[:200];return [ord(c)-31 if 32<=ord(c)<=126 else 96 for c in t]+[0]*(200-len(t))
def fixtures():
 rng=random.Random(7042);out=[]
 for i in range(200):
  host=f'tenant-{i:03d}.example.test'
  for label,leaf in [(0,'index.html'),(1,'login.php')]:
   key=f'{rng.getrandbits(96):024x}';path=f'/news/{key}/index.html' if not label else f'/account/{key}/login.php'
   u=f'http://{host}{path}?case={key}'
   # No executable scripts, external assets, input collection or external form actions.
   body=(f'<!doctype html><html><head><title>Local research fixture</title></head><body>'
         f'<h1>Controlled research fixture</h1><p>Case {key}</p>'
         f'<section data-simulation-class="{label}">'
         + ('<p>Simulated account prompt. No credentials are collected.</p><button disabled>Disabled demonstration</button>' if label else '<p>Simulated informational article.</p>')
         + '</section></body></html>').encode()
   out.append({'case_id':key,'host':host,'label':label,'leaf':leaf,'url':u,'body':body,'body_sha256':hashlib.sha256(body).hexdigest()})
 return out

def router(cases,host,target):
 p=urlsplit(target);q=parse_qs(p.query);ids=q.get('case',[])
 if len(ids)!=1:return 404,b'Unknown case'
 case=cases.get(ids[0])
 if not case or host.lower()!=case['host']:return 404,b'Unknown host'
 # Explicit application policy: arbitrary directory aliases, fixed decoded leaf and case ID.
 # Campaign/theme parameters and fragments have no application semantics in this fixture.
 leaf=unquote(p.path).rsplit('/',1)[-1]
 if leaf!=case['leaf']:return 404,b'Unknown leaf'
 return 200,case['body']
class Lab:
 def __init__(self,items):self.cases={x['case_id']:x for x in items};self.requests=0
 def __enter__(self):
  lab=self
  class Handler(BaseHTTPRequestHandler):
   def do_GET(self):
    lab.requests+=1;status,body=router(lab.cases,self.headers.get('Host',''),self.path)
    self.send_response(status);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
   def log_message(self,*args):pass
  self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.daemon_threads=True
  self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();return self
 def validate(self,case,u):
  p=urlsplit(u)
  if p.scheme!='http' or p.hostname!=case['host'] or p.username or p.password or p.port is not None:return False
  # Physical connection is always loopback: no DNS lookup or connection to logical hosts.
  target=quote(p.path,safe='/%:@-._~')+('?' + quote(p.query,safe='=&%:/?@-._~') if p.query else '')
  conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
  try:
   conn.request('GET',target,headers={'Host':p.netloc});r=conn.getresponse();body=r.read()
   return r.status==200 and r.getheader('Content-Type')=='text/html; charset=utf-8' and hashlib.sha256(body).hexdigest()==case['body_sha256']
  finally:conn.close()
 def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join()

def change(u,kind):
 p=urlsplit(u);path=p.path;query=p.query;fragment=p.fragment
 if kind.startswith('path_'):
  count=int(kind.split('_')[1]);path='/' + ('news/products/support/documentation/'*((count//36)+1))[:count] + '/' + path.lstrip('/')
 elif kind.startswith('pct_'):
  count=int(kind.split('_')[1]);seen=0;parts=[];i=0
  while i<len(path):
   c=path[i]
   if c=='%' and i+2<len(path):parts.append(path[i:i+3]);i+=3;continue
   if c.isascii() and c.isalpha() and seen<count:parts.append(f'%{ord(c):02X}');seen+=1
   else:parts.append(c)
   i+=1
  path=''.join(parts)
 elif kind=='brand_query':query+='&campaign=mtn-momo-paypal-account'
 elif kind=='unicode_query':query+='&theme=m\u03bfm\u03bf-support'
 elif kind=='long_query':query+='&theme='+('news-products-support-'*24)
 elif kind=='fragment':fragment='about-products-contact-support'
 else:raise ValueError(kind)
 v=urlunsplit((p.scheme,p.netloc,path,query,fragment))
 return v if len(v)<=4096 else u
ROUNDS=[['brand_query','pct_1','path_32','fragment'],['path_256','path_512','unicode_query','long_query'],['pct_2','path_96','brand_query','unicode_query'],['path_512','pct_4','long_query','fragment']]

def load_extract(path):
 # Extract definitions only; never execute the generator's CSV-writing main block.
 import ipaddress,tldextract
 tree=ast.parse(path.read_text(encoding='utf-8-sig'))
 domain=tldextract.TLDExtract(suffix_list_urls=(),include_psl_private_domains=True,cache_dir=None)
 scope={'math':math,'ipaddress':ipaddress,'urlsplit':urlsplit,'extract_domain':domain}
 required={'FREE_TLDS','MFS_BRANDS','SHORTENERS','TRUSTED_GH'}
 for node in tree.body:
  if isinstance(node,ast.Assign):
   for t in node.targets:
    if isinstance(t,ast.Name) and t.id in required:scope[t.id]=ast.literal_eval(node.value)
 fn=next((x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='extract'),None)
 if fn is None or not required<=scope.keys():raise ValueError('Expected corrected extract() and indicator constants in generate_corrected_features.py')
 module=ast.Module(body=[fn],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
 scope['extract']._audit_meta={'generator_sha256':sha(path),'tldextract_version':tldextract.__version__,'suffix_list_urls':[],'include_psl_private_domains':True,'numeric_dtype':'float32'}
 return scope['extract']

def paths(root,seed):
 main=root/'corrected_results_v3'/f'seed_{seed}'
 return {'CNN_LSTM':main/'phishguard_gh_model.keras','RF_full':root/'baseline_results_v3'/f'seed_{seed}'/'RF_full'/'model.pkl','XGB_full':root/'baseline_results_v3'/f'seed_{seed}'/'XGB_full'/'model.pkl'}
def preflight(root,seed):
 import numpy as np,pandas as pd
 for name,h in PART_HASH.items():
  if sha(root/'dataset_fix_outputs_v3'/f'{name}.csv')!=h:raise ValueError('Frozen partition changed: '+name)
 test=read(root/'dataset_fix_outputs_v3/test.csv');files=paths(root,seed)
 files['corrected_feature_generator']=root/'generate_corrected_features.py'
 files['test_features']=root/'dataset_fix_outputs_v3/test_features.csv'
 files['baseline_protocol']=root/'baseline_results_v3/protocol.json'
 for name,p in files.items():
  if not p.is_file():raise FileNotFoundError(p)
 recorded=json.loads(files['baseline_protocol'].read_text(encoding='utf-8-sig'))
 if recorded.get('partition_hashes')!=PART_HASH:raise ValueError('Baseline protocol partition hashes differ')
 if sha(files['test_features'])!=recorded['feature_file_hashes']['test']:raise ValueError('Cached test features differ from the trained baseline protocol. Do not regenerate or overwrite them.')
 feat=pd.read_csv(files['test_features'],keep_default_na=False)
 if feat.url.tolist()!=[r['url'] for r in test] or feat.label.tolist()!=[int(r['label']) for r in test]:raise ValueError('Cached feature URL/label alignment differs')
 extract=load_extract(files['corrected_feature_generator']);columns=[c for c in feat.columns if c not in {'url','label'}]
 generated=pd.DataFrame([extract(r['url']) for r in test])
 if len(columns)!=23 or generated.columns.tolist()!=columns:raise ValueError('Corrected feature order differs')
 original=feat[columns].to_numpy(dtype=np.float32);replayed=generated.to_numpy(dtype=np.float32)
 if not np.isfinite(original).all() or not np.isfinite(replayed).all():raise ValueError('Non-finite feature matrix')
 mismatch=(original!=replayed)
 if mismatch.any():
  differing={c:int(mismatch[:,i].sum()) for i,c in enumerate(columns) if mismatch[:,i].any()}
  raise ValueError('Corrected generator does not reproduce cached float32 features: '+json.dumps(differing))
 print(f'FEATURE REPLAY PASSED: {len(test):,} rows x 23 columns; exact float32 match and baseline-recorded file hash.')
 for name in ['CNN_LSTM','RF_full','XGB_full']:
  path=(root/'corrected_results_v3'/f'seed_{seed}'/'test_predictions.csv') if name=='CNN_LSTM' else files[name].parent/'test_predictions.csv'
  saved=read(path)
  if [(x['url'],int(x['true_label'])) for x in saved]!=[(x['url'],int(x['label'])) for x in test]:raise ValueError('Saved test alignment mismatch: '+name)
  files[name+'_test_predictions']=path
  if name!='CNN_LSTM':
   meta=files[name].parent/'metrics.json'
   if json.loads(meta.read_text(encoding='utf-8-sig')).get('input_columns')!=columns:raise ValueError('Saved model feature order differs: '+name)
   files[name+'_metrics']=meta
 manifest={k:{'relative_path':p.relative_to(root).as_posix(),'sha256':sha(p)} for k,p in files.items()}
 return test,files,extract,manifest

class Scorer:
 def __init__(self,name,path,extract):
  import numpy as np
  self.np=np;self.name=name;self.extract=extract
  if name=='CNN_LSTM':
   import keras
   self.model=keras.models.load_model(path,compile=False)
  else:
   import joblib
   self.model=joblib.load(path)
 def key(self,u):
  if self.name=='CNN_LSTM':return bytes(encode(u))
  return tuple(self.extract(u).values())
 def predict(self,urls):
  if self.name=='CNN_LSTM':x=self.np.asarray([encode(u) for u in urls],dtype=self.np.int32);p=self.model.predict(x,batch_size=256,verbose=0).reshape(-1)
  else:
   import pandas as pd
   x=pd.DataFrame([self.extract(u) for u in urls]);names=getattr(self.model,'feature_names_in_',None)
   if names is not None:x=x[list(names)]
   else:x=x.to_numpy(dtype=self.np.float32)
   classes=list(self.model.classes_)
   if 1 not in classes:raise ValueError('No project-positive class in '+self.name)
   p=self.model.predict_proba(x)[:,classes.index(1)]
  if len(p)!=len(urls) or not self.np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid model outputs')
  return [float(x) for x in p]

def verify_saved_models(test,files,extract):
 scorers={};checks={}
 # Verify every model before starting any fixture rounds.
 for name in ['RF_full','XGB_full','CNN_LSTM']:
  print('Loading and verifying original saved predictions:',name,flush=True)
  scorer=Scorer(name,files[name],extract);actual=scorer.predict([r['url'] for r in test]);saved=read(files[name+'_test_predictions']);expected=[float(r['probability']) for r in saved]
  delta=max(abs(a-b) for a,b in zip(actual,expected));flips=sum((a>=.5)!=(b>=.5) for a,b in zip(actual,expected))
  if delta>1e-5 or flips:raise ValueError(f'{name}: original saved predictions do not reproduce; max delta {delta}, verdict differences {flips}. No fixture rounds started.')
  checks[name]={'saved_test_score_max_difference':delta,'saved_test_verdict_differences':flips}
  scorers[name]=scorer;print(f'{name}: saved-score check passed; max difference {delta:.9g}; verdict differences {flips}',flush=True)
 print('ALL THREE MODELS VERIFIED before fixture scoring.',flush=True)
 return scorers,checks

def metrics(labels,probs):
 q=[p>=.5 for p in probs];tp=sum(y==1 and p for y,p in zip(labels,q));tn=sum(y==0 and not p for y,p in zip(labels,q));fp=sum(y==0 and p for y,p in zip(labels,q));fn=sum(y==1 and not p for y,p in zip(labels,q))
 return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'recall':tp/(tp+fn) if tp+fn else None,'false_positive_rate':fp/(fp+tn) if fp+tn else None,'accuracy':(tp+tn)/len(labels)}
def self_test():
 items=fixtures();subset=items[:6]
 with Lab(items) as lab:
  for case in subset:
   assert lab.validate(case,case['url'])
   for kinds in ROUNDS:
    for kind in kinds:assert lab.validate(case,change(case['url'],kind)),kind
   assert not lab.validate(case,case['url'].replace('case=','wrong='))
   assert not lab.validate(case,case['url'].replace(case['leaf'],'wrong.html'))
  print('SELF-TEST PASSED: real loopback HTTP checks, response hashes, route negatives and all transformation families.')
 print('No model loaded, inference, training, source-file changes or live-site requests.')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,choices=range(42,47),default=42);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
 mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--check-only',action='store_true');mode.add_argument('--self-test',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--verify-only',action='store_true');args=ap.parse_args()
 if args.self_test:self_test();return
 root=args.root.resolve();test,files,extract,manifest=preflight(root,args.seed);items=fixtures()
 print(f'CHECK PASSED: seed {args.seed}, frozen partitions, three saved models, saved test alignments and 23-feature extractor.')
 print('Controlled cases: 200 simulated-positive and 200 simulated-negative; 4 rounds x at most 4 new score queries.')
 if args.check_only:print('No model loaded, inference, server, training or output folders created.');return
 if args.verify_only:
  scorers,checks=verify_saved_models(test,files,extract);print(json.dumps(checks,indent=2));print('Verification complete. No fixture rounds, server, training or output folders.');return
 out=root/'controlled_robustness_v1'/('seed_'+str(args.seed)+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));out.mkdir(parents=True,exist_ok=False)
 protocol={'seed':args.seed,'fixture_seed':7042,'fixture_cases':400,'partition_sha256':PART_HASH,'files':manifest,'script_sha256':sha(Path(__file__)),'rounds':ROUNDS,'maximum_new_score_queries_per_case':16,'original_query_charged_separately':True,'threshold':.5,'selection':'Four adaptive rounds; minimize score on simulated positives, maximize on simulated benign false-alarm controls; retain original if no improvement.','input_deduplication':'Per case and model, skip already queried encodings or numeric feature vectors.','functionality_validation':'HTTP 200, exact content type and per-case response-body SHA256 using loopback connection and logical Host header; no redirects or scripts.','fixture_application_policy':'Known tenant and opaque case ID, fixed decoded leaf, arbitrary directory aliases; ignores campaign/theme and fragments.','limitations':LIMITATIONS,'training_performed':False,'external_test':False,'protocol_written_before_inference':True,'scorer_revision':2,'feature_extractor':extract._audit_meta}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 scorers,verification=verify_saved_models(test,files,extract)
 (out/'saved_prediction_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
 rows=[];trace=[];summaries={}
 with Lab(items) as lab:
  for name in ['CNN_LSTM','RF_full','XGB_full']:
   sc=scorers[name];delta=verification[name]['saved_test_score_max_difference'];verdict_diff=verification[name]['saved_test_verdict_differences']
   for x in items:
    if not lab.validate(x,x['url']):raise ValueError('Original fixture failed')
   base=sc.predict([x['url'] for x in items]);best=list(base);selected=[x['url'] for x in items];queries=[0]*len(items);seen=[{sc.key(x['url'])} for x in items];invalid=unchanged=0
   for round_index,kinds in enumerate(ROUNDS,1):
    pending=[]
    for i,case in enumerate(items):
     current=selected[i]
     for kind in kinds:
      v=change(current,kind)
      if not lab.validate(case,v):invalid+=1;continue
      key=sc.key(v)
      if key in seen[i]:unchanged+=1;continue
      seen[i].add(key);pending.append((i,kind,v));queries[i]+=1
    if pending:
     ps=sc.predict([v for i,k,v in pending])
     for (i,kind,v),p in zip(pending,ps):
      trace.append({'model':name,'seed':args.seed,'case_id':items[i]['case_id'],'simulated_label':items[i]['label'],'round':round_index,'family':kind,'url':v,'score':p,'functionality_validated':True,'body_sha256':items[i]['body_sha256']})
      if (items[i]['label']==1 and p<best[i]) or (items[i]['label']==0 and p>best[i]):best[i]=p;selected[i]=v
    print(f'{name}: round {round_index}/4, cumulative new queries {sum(queries):,}',flush=True)
   labels=[x['label'] for x in items];den=sum(y==1 and p>=.5 for y,p in zip(labels,base));evaded=sum(y==1 and p>=.5 and b<.5 for y,p,b in zip(labels,base,best))
   for i,x in enumerate(items):rows.append({'model':name,'seed':args.seed,'case_id':x['case_id'],'simulated_label':x['label'],'original_url':x['url'],'selected_url':selected[i],'original_score':base[i],'selected_score':best[i],'new_queries':queries[i],'originally_detected_simulated_positive':x['label']==1 and base[i]>=.5,'simulated_evasion':x['label']==1 and base[i]>=.5 and best[i]<.5,'body_sha256':x['body_sha256']})
   assert max(queries)<=16
   summaries[name]={'original_fixture_metrics':metrics(labels,base),'worst_case_fixture_metrics':metrics(labels,best),'originally_detected_simulated_positives':den,'simulated_evasions':evaded,'simulated_evasion_rate_among_initially_detected':evaded/den if den else None,'new_score_queries':sum(queries),'maximum_new_queries_for_one_case':max(queries),'invalid_functionality_candidates':invalid,'unchanged_or_already_queried_inputs_skipped':unchanged,'saved_test_score_max_difference':delta,'saved_test_verdict_differences':verdict_diff}
   # Preserve every completed model even if a later fixture round fails.
   for filename,records in [('per_case_results.csv',rows),('candidate_trace.csv',trace)]:
    if records:
     with (out/filename).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
   (out/'partial_results.json').write_text(json.dumps({'complete':False,'results':summaries},indent=2,allow_nan=False),encoding='utf-8')
 protocol['loopback_http_requests']=lab.requests
 for filename,records in [('per_case_results.csv',rows),('candidate_trace.csv',trace)]:
  if records:
   with (out/filename).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 (out/'report.json').write_text(json.dumps({'complete':True,'protocol':protocol,'results':summaries},indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps(summaries,indent=2));print('Saved to:',out);print('No training, external network requests or existing input/model changes. Results are controlled-fixture evidence only.')
if __name__=='__main__':main()
