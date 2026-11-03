"""Offline SHA-256/HMAC logging experiment. Never edits source logs or trains models."""
import argparse
import csv
import hashlib
import hmac
import json
import platform
import random
import secrets
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GENESIS='0'*64
PART_SHA={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f','validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e','test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}

def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def canonical(record):
 return json.dumps({k:v for k,v in record.items() if k!='record_hash'},sort_keys=True,ensure_ascii=True).encode('utf-8')

def digest(record,key=None):
 message=canonical(record)
 return hashlib.sha256(message).hexdigest() if key is None else hmac.new(key,message,hashlib.sha256).hexdigest()

def build(payloads,key=None,log_id=None):
 result=[];prev=GENESIS
 for payload in payloads:
  r={k:v for k,v in payload.items() if k not in {'prev_hash','record_hash'}}
  if log_id is not None:r['log_id']=log_id
  r['prev_hash']=prev;r['record_hash']=digest(r,key);prev=r['record_hash'];result.append(r)
 return result

def verify(records,key=None):
 prev=GENESIS
 for i,r in enumerate(records):
  try:
   if r['prev_hash']!=prev or not hmac.compare_digest(digest(r,key),r['record_hash']):return False,i
   prev=r['record_hash']
  except (KeyError,TypeError,ValueError):return False,i
 return True,None

def checkpoint(records,log_id):
 return {'log_id':log_id,'count':len(records),'final_hash':records[-1]['record_hash'] if records else GENESIS}

def accept(records,key=None,reference=None):
 if not verify(records,key)[0]:return False
 if reference is None:return True
 actual=checkpoint(records,reference['log_id'])
 if actual!=reference:return False
 return all(r.get('log_id')==reference['log_id'] for r in records)

def changed(value):
 if isinstance(value,bool):return not value
 if isinstance(value,int):return value+7
 if isinstance(value,float):return .123456 if value!=.123456 else .654321
 return str(value)+' [edited]'

def scenarios(records):
 n=len(records)
 yield 'untampered_control','none',None,None,'control'
 for pos,label in [(0,'first'),(n//2,'middle'),(n-1,'last')]:
  for field in records[pos]:yield f'edit_{label}_{field}','log_write',pos,field,'edit'
  yield f'delete_{label}','log_write',pos,None,'delete'
 yield 'swap_first_two','log_write',None,None,'swap'
 yield 'delete_tail_1','log_write',1,None,'tail'
 yield 'delete_tail_10','log_write',min(10,n-1),None,'tail'
 yield 'delete_tail_10percent','log_write',max(1,n//10),None,'tail'
 yield 'delete_all','log_write',None,None,'empty'
 yield 'edit_and_rebuild_without_key','log_write',None,None,'rebuild_no_key'
 yield 'edit_and_rebuild_with_stolen_key','log_write_and_key',None,None,'rebuild_key'
 yield 'append_with_stolen_key','log_write_and_key',None,None,'append_key'
 yield 'rebuild_and_replace_checkpoint','log_write_key_and_reference',None,None,'replace_reference'

def mutate(records,key,position,field,kind):
 # Shallow record copies are sufficient: source payload fields are scalars.
 out=[dict(r) for r in records]
 if kind=='edit':out[position][field]=changed(out[position][field])
 elif kind=='delete':del out[position]
 elif kind=='swap':out[0],out[1]=out[1],out[0]
 elif kind=='tail':out=out[:-position]
 elif kind=='empty':out=[]
 elif kind in {'rebuild_no_key','rebuild_key','replace_reference'}:
  out[len(out)//2]['verdict']='LEGITIMATE' if out[len(out)//2]['verdict']=='PHISHING' else 'PHISHING'
  # A no-key attacker recomputes plain hashes, or attempts HMAC using a wrong key.
  attacker_key=(b'not-the-secret-key' if key is not None else None) if kind=='rebuild_no_key' else key
  out=build(out,attacker_key)
 elif kind=='append_key':
  new=dict(out[-1]);new['seq_no']=len(out)+1;out.append(new);out=build(out,key)
 return out

def evaluate_attacks(plain,keyed,key,log_id):
 references={'sha256':checkpoint(plain,log_id),'hmac_sha256':checkpoint(keyed,log_id)}
 rows=[]
 for i,(name,cap,pos,field,kind) in enumerate(scenarios(plain)):
  row={'scenario':name,'attacker_capability':cap,'target_index':pos,'field':field,'kind':kind}
  for alg,base,k in [('sha256',plain,None),('hmac_sha256',keyed,key)]:
   tampered=mutate(base,k,pos,field,kind)
   ref=checkpoint(tampered,log_id) if kind=='replace_reference' else references[alg]
   row[alg+'_internal_accepts']=accept(tampered,k)
   row[alg+'_checkpoint_accepts']=accept(tampered,k,ref)
  if kind=='control':expected=[True]*4
  elif kind in {'tail','empty'} or (kind=='delete' and pos==len(plain)-1):expected=[True,False,True,False]
  elif kind=='rebuild_no_key':expected=[True,False,False,False]
  elif kind in {'rebuild_key','append_key'}:expected=[True,False,True,False]
  elif kind=='replace_reference':expected=[True]*4
  else:expected=[False]*4
  actual=[row[c] for c in ['sha256_internal_accepts','sha256_checkpoint_accepts','hmac_sha256_internal_accepts','hmac_sha256_checkpoint_accepts']]
  if actual!=expected:raise AssertionError(f'Unexpected result for {name}: {actual} vs {expected}')
  row['matches_stated_expectation']=True;rows.append(row)
 return rows,references

def timing(payloads,key,log_id,repeats):
 rng=random.Random(42);rows=[]
 for rep in range(1,repeats+1):
  methods=['sha256','hmac_sha256'];rng.shuffle(methods)
  for order,alg in enumerate(methods):
   k=None if alg=='sha256' else key
   start=time.perf_counter_ns();records=build(payloads,k,log_id);elapsed=time.perf_counter_ns()-start
   rows.append({'repeat':rep,'order_in_pair':order,'algorithm':alg,'stage':'chain_construction_in_memory','records':len(records),'ms_per_record':elapsed/1e6/len(records)})
   start=time.perf_counter_ns();ok=accept(records,k,checkpoint(records,log_id));elapsed=time.perf_counter_ns()-start
   if not ok:raise AssertionError('Timing control failed')
   rows.append({'repeat':rep,'order_in_pair':order,'algorithm':alg,'stage':'verification_with_checkpoint_in_memory','records':len(records),'ms_per_record':elapsed/1e6/len(records)})
 return rows

def write_csv(path,rows):
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--seed',type=int,choices=[42,43,44,45,46],default=42)
 parser.add_argument('--records',type=int,default=5000,help='Prefix sample for attacks/timing; whole original log is always verified first')
 parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
 root=Path(__file__).resolve().parent
 folder=root/'corrected_results_v3'/f'seed_{args.seed}'
 chain_path=folder/'phishguard_evidence_chain.jsonl'
 original_hash=sha_file(chain_path)
 expected_urls=[]
 for name,expected in PART_SHA.items():
  path=root/'dataset_fix_outputs_v3'/f'{name}.csv'
  if sha_file(path)!=expected:raise ValueError(f'{name}: frozen partition fingerprint differs')
  with path.open(encoding='utf-8-sig',newline='') as f:expected_urls.extend(row['url'] for row in csv.DictReader(f))
 records=[]
 with chain_path.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():records.append(json.loads(line))
 if len(records)!=len(expected_urls) or len({r['url'] for r in records})!=len(records) or {r['url'] for r in records}!=set(expected_urls):raise ValueError('Source log membership differs from frozen v3 partitions')
 if any(r.get('seq_no')!=i+1 for i,r in enumerate(records)):raise ValueError('Source sequence numbers differ')
 ok,index=verify(records)
 if not ok:raise ValueError(f'Original chain integrity failed at index {index}')
 if any(any(isinstance(v,(dict,list)) for v in r.values()) for r in records):raise ValueError('Nested record schema needs review')
 if not 20<=args.records<=len(records):raise ValueError(f'--records must be between 20 and {len(records)}')
 print(f'CHECK PASSED: all {len(records):,} original records verified; {args.records:,}-record prefix planned for experiments.')
 if args.check_only:
  print('No reports created, training, inference or source-file changes.');return
 # Ephemeral key: never saved, printed, sent or used to protect a production log.
 key=secrets.token_bytes(32);log_id=secrets.token_hex(16)
 payloads=records[:args.records]
 plain=build(payloads,None,log_id);keyed=build(payloads,key,log_id)
 print('Testing field edits, deletions, truncation, reconstruction and key/reference compromise...')
 attacks,references=evaluate_attacks(plain,keyed,key,log_id)
 print('Timing matched in-memory construction and verification, five paired repeats...')
 timed=timing(payloads,key,log_id,5);summary=[]
 for stage in sorted({r['stage'] for r in timed}):
  for alg in ['sha256','hmac_sha256']:
   vals=[r['ms_per_record'] for r in timed if r['stage']==stage and r['algorithm']==alg]
   summary.append({'stage':stage,'algorithm':alg,'repeats':len(vals),'mean_ms_per_record':statistics.mean(vals),'sample_sd_ms_per_record':statistics.stdev(vals)})
 # Before any output, confirm the original log has not changed.
 if sha_file(chain_path)!=original_hash:raise ValueError('Source log changed during experiment; no report saved')
 out=root/'secure_logging_evaluation_v3'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
 write_csv(out/'attack_results.csv',attacks);write_csv(out/'timing_per_repeat.csv',timed);write_csv(out/'timing_summary.csv',summary)
 report={'seed':args.seed,'original_chain_records':len(records),'tested_prefix_records':len(payloads),
  'attack_scenarios_including_control':len(attacks),'all_scenario_expectations_met':True,
  'original_chain_sha256':original_hash,'partition_sha256':PART_SHA,'script_sha256':sha_file(Path(__file__)),
  'algorithms':['SHA-256 chain','HMAC-SHA-256 chain'],'canonicalization':'sorted-key ensure_ascii JSON, default separators, UTF-8; exclude record_hash',
  'key_bytes':32,'key_storage':'ephemeral memory only, not persisted; fresh per experiment',
  'checkpoint_reference_model':'expected log_id, final count and tip held outside attacker mutations in verifier memory',
  'checkpoints':references,'external_anchor_deployed':False,'training_performed':False,'inference_performed':False,
  'runtime':{'python':sys.version,'platform':platform.platform(),'processor':platform.processor()},
  'references':['https://www.rfc-editor.org/rfc/rfc2104','https://docs.python.org/3/library/hmac.html','https://www.rfc-editor.org/rfc/rfc5848'],
  'limitations':['HMAC is a standard authentication mechanism, not a new cryptographic contribution.',
   'A log-write attacker is assumed unable to read the HMAC key unless the scenario explicitly grants key compromise.',
   'Same-process memory is not demonstrated isolation from a compromised host; OS, hardware and remote key protection are not evaluated.',
   'Any holder of the shared key can forge records; no non-repudiation or forward security is established.',
   'A valid prefix, including empty input, passes internal-only SHA and HMAC checks; known count/tip freshness is required to detect rollback.',
   'The checkpoint is a simulated protected reference, not an externally anchored or independently timestamped service.',
   'A compromised key permits full HMAC reconstruction; replacing the reference as well defeats all simulated checks.',
   'New keyed records authenticate a retrospective copy of prior records, not their historical origin or acquisition time.',
   'Timing is in-memory construction and verification only, not disk logging, network anchoring or end-to-end detector latency.',
   'Attack counts are deterministic scenario outcomes, not population detection rates or a security proof.',
   'No classifier truth, legal admissibility, key lifecycle or authenticated operator identity was evaluated.']}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('\nATTACK SUMMARY: True means the altered log was accepted (undetected).')
 for row in attacks:
  if row['kind'] not in ['edit','delete']:
   print(row['scenario'],{k:v for k,v in row.items() if k.endswith('_accepts')})
 print('\nTIMING SUMMARY (ms/record)')
 for row in summary:print(row)
 print('\nSaved to:',out)
 print('Original log preserved. No training or model inference. Ephemeral test key was not saved.')

if __name__=='__main__':main()
