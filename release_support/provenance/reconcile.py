import csv,io,re,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import pandas as pd
W=Path(__file__).resolve().parent;B=W/'collected'
def shab(b):return hashlib.sha256(b).hexdigest()
def read(n):
 with (B/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pairs(rs):return Counter((r['url'],int(r['label'])) for r in rs)
def canon(u):return re.sub(r'^https?://','',u.strip()).lower().strip()
def enc(u):
 t=u.strip()
 for p in ['https://','http://']:
  if t.lower().startswith(p):t=t[len(p):];break
 t=t.lower().strip()[:200];return bytes([ord(c)-31 if 32<=ord(c)<=126 else 96 for c in t]+[0]*(200-len(t)))
def csv_bytes(rs):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=['url','label']);w.writeheader();w.writerows(rs);return f.getvalue().encode()
report={'checks':{},'limitations':[]};c=report['checks']
src=read('malicious_phish.csv');a=read('phishguard_gh_dataset_augmented.csv');g=read('modern_legit_augmentation.csv');r=read('phishguard_gh_dataset_recall.csv')
c['original_source_rows']=len(src);c['source_categories']=dict(Counter(x['type'] for x in src));assert len(src)==651191
assert shab((B/'malicious_phish.csv').read_bytes())=='d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'
base=pairs(a)-pairs(g);assert sum(base.values())==50000
# Replay original pandas sampling with source ordering and seed unchanged.
df=pd.read_csv(B/'malicious_phish.csv');df['label']=df.type.str.lower().str.strip().ne('benign').astype(int)
sampled=pd.concat([df[df.label==1].sample(25000,random_state=42),df[df.label==0].sample(25000,random_state=42)],ignore_index=True)[['url','label']].dropna().sample(frac=1,random_state=42).reset_index(drop=True)
assert Counter(zip(sampled.url,sampled.label))==base;c['base_sampling_membership_replayed']=True
for line in ['\n','\r\n']:
 b=sampled.to_csv(index=False,lineterminator=line).encode()
 if shab(b)=='79c4ace036c6f0cf64fd0447a1a8b24bd95f136b4bfb502f1c38a6f8462f1736':
  (W/'base_recovered_exact.csv').write_bytes(b);c['base_original_lfs_target_recovered_exactly']=True;c['base_sha256']=shab(b);break
if not c.get('base_original_lfs_target_recovered_exactly'):
 c['base_original_lfs_target_recovered_exactly']=False
 sampled.to_csv(W/'base_sampling_replayed.csv',index=False)
# Independent cleaning replay.
labels=defaultdict(set)
for x in a:labels[canon(x['url'])].add(x['label'])
conflict={k for k,v in labels.items() if len(v)>1};seen=set();clean=[];bad=[];dup=[]
for x in sorted(a,key=lambda x:(canon(x['url']),x['url'],x['label'])):
 k=canon(x['url'])
 if k in conflict:bad.append(x)
 elif k in seen:dup.append(x)
 else:seen.add(k);clean.append(x)
for name,rs in [('dataset_cleaned.csv',clean),('conflicting_records.csv',bad),('removed_duplicates.csv',dup)]:
 assert rs==read('dataset_fix_outputs/'+name)
 assert csv_bytes(rs)==(B/'dataset_fix_outputs'/name).read_bytes()
c['cleaning_replayed']={'conflicting_keys':len(conflict),'conflicting_rows':len(bad),'duplicates':len(dup),'cleaned_rows':len(clean),'all_three_outputs_byte_identical':True}
invalid=read('dataset_fix_outputs/invalid_hostname_records.csv');initial={p:read('dataset_fix_outputs/'+p+'.csv') for p in ['train','validation','test']};assert pairs(clean)==sum((pairs(x) for x in initial.values()),Counter())+pairs(invalid)
c['initial_partition_row_accounting_exact']=True;c['invalid_hostname_rows']=len(invalid)
# Full-source encoding labels; include saved generated benign rows exactly as the historical source audit.
labels_enc=defaultdict(set);source_canon=defaultdict(set)
for x in src:
 y=int(x['type'].strip().lower()!='benign');labels_enc[enc(x['url'])].add(y);source_canon[canon(x['url'])].add(y)
for x in g:labels_enc[enc(x['url'])].add(int(x['label']))
ambiguous={k for k,v in labels_enc.items() if len(v)>1};v2={};v3={};removals={};finalprovenance=[]
gkeys={canon(x['url']) for x in g};assert not gkeys&source_canon.keys()
for p,rs in initial.items():
 kept=[x for x in rs if enc(x['url']) not in ambiguous];quar=[x for x in rs if enc(x['url']) in ambiguous]
 assert kept==read('dataset_fix_outputs_v2/'+p+'.csv');assert quar==read('dataset_fix_outputs_v2/'+p+'_quarantined.csv')
 assert csv_bytes(kept)==(B/'dataset_fix_outputs_v2'/f'{p}.csv').read_bytes();assert csv_bytes(quar)==(B/'dataset_fix_outputs_v2'/f'{p}_quarantined.csv').read_bytes()
 v2[p]=kept;v3[p]=read('dataset_fix_outputs_v3/'+p+'.csv');removals[p]=len(quar)
trcodes={enc(x['url']) for x in v2['train']};removed=[x for x in v2['validation'] if enc(x['url']) in trcodes];kept=[x for x in v2['validation'] if enc(x['url']) not in trcodes]
assert kept==v3['validation'];assert removed==read('dataset_fix_outputs_v3/validation_encoding_overlap_removed.csv');assert len(removed)==1
assert v2['train']==v3['train'] and v2['test']==v3['test']
for p,rs in v3.items():
 assert csv_bytes(rs)==(B/'dataset_fix_outputs_v3'/f'{p}.csv').read_bytes()
 for x in rs:
  k=canon(x['url']);y=int(x['label'])
  assert labels_enc[enc(x['url'])]=={y}
  if k not in gkeys:assert source_canon[k]=={y}
 finalprovenance.append({'partition':p,'rows':len(rs),'generated_benign':sum(canon(x['url']) in gkeys for x in rs),'source_rows':sum(canon(x['url']) not in gkeys for x in rs),'sha256':shab((B/'dataset_fix_outputs_v3'/f'{p}.csv').read_bytes())})
c['source_wide_quarantine_replayed']=removals;c['source_wide_quarantined_total']=sum(removals.values());c['v2_outputs_byte_identical']=True;c['v3_overlap_removal_replayed']=1;c['v3_outputs_byte_identical']=True;c['final_source_and_label_checks_passed']=True;c['partition_provenance']=finalprovenance
extra=pairs(r)-pairs(a);assert not pairs(a)-pairs(r);assert sum(extra.values())==4000;final=sum((pairs(x) for x in v3.values()),Counter());assert not extra&final
c['counterfactual_rows']=4000;c['counterfactual_exact_rows_retained']=0;c['counterfactual_new_canonical_keys']=len({canon(u) for u,y in extra}-{canon(x['url']) for x in a})
pointer=(B/'top-1m.csv').read_bytes();c['tranco_pointer_sha256']=shab(pointer);c['tranco_target_sha256']=re.search(rb'oid sha256:([a-f0-9]+)',pointer)[1].decode();c['tranco_target_size']=int(re.search(rb'size (\d+)',pointer)[1]);c['tranco_hash_issue']='The previously reported 6b... hash identifies the pointer text; 3a... identifies its data target, not a conflicting downloaded list.'
report['limitations']=['Historical Tranco daily-list identifier and original console log unavailable.','Initial domain fold assignment is preserved and exactly accounted for, not regenerated.','Inherited source labels and assumed-benign generated labels are not live verified.']
c['base_sampling_byte_identity_to_project_dataset_pointer_target']=c.pop('base_original_lfs_target_recovered_exactly')
c['project_dataset_pointer_target_role']='Legacy 51,638-row project dataset; not the 50,000-row base sample.'
c['tranco_target_recovered_and_verified']=(W/'top-1m_recovered.csv').is_file() and shab((W/'top-1m_recovered.csv').read_bytes())==c['tranco_target_sha256']
(W/'reconciliation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
