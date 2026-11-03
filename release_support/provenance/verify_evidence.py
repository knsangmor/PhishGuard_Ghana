"""Read-only file, membership and row-count checks; no training or network access."""
import csv,hashlib,json,re
from collections import Counter
from pathlib import Path
W=Path(__file__).resolve().parent;B=W/'collected'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((W/'manifest.json').read_text())
for name,h in manifest['files_sha256'].items():
 if sha(W/name)!=h:raise ValueError('Fingerprint mismatch: '+name)
def rows(n):
 with (B/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pairs(rs):return Counter((x['url'],int(x['label'])) for x in rs)
def canon(u):return re.sub(r'^https?://','',u.strip()).lower().strip()
a=rows('phishguard_gh_dataset_augmented.csv');g=rows('modern_legit_augmentation.csv');recall=rows('phishguard_gh_dataset_recall.csv');clean=rows('dataset_fix_outputs/dataset_cleaned.csv');bad=rows('dataset_fix_outputs/conflicting_records.csv');dup=rows('dataset_fix_outputs/removed_duplicates.csv');invalid=rows('dataset_fix_outputs/invalid_hostname_records.csv')
assert len(a)==52244 and len(g)==2244 and len(recall)==56244
assert len(clean)==51618 and len(bad)==40 and len(dup)==586 and len(invalid)==1
assert pairs(a)==pairs(clean)+pairs(bad)+pairs(dup)
initial={p:rows('dataset_fix_outputs/'+p+'.csv') for p in ['train','validation','test']};v2={p:rows('dataset_fix_outputs_v2/'+p+'.csv') for p in initial};v3={p:rows('dataset_fix_outputs_v3/'+p+'.csv') for p in initial}
assert pairs(clean)==sum((pairs(x) for x in initial.values()),Counter())+pairs(invalid)
for p,n in [('train',480),('validation',68),('test',83)]:
 q=rows('dataset_fix_outputs_v2/'+p+'_quarantined.csv');assert len(q)==n;assert pairs(initial[p])==pairs(v2[p])+pairs(q)
removed=rows('dataset_fix_outputs_v3/validation_encoding_overlap_removed.csv');assert len(removed)==1;assert pairs(v2['validation'])==pairs(v3['validation'])+pairs(removed);assert v2['train']==v3['train'] and v2['test']==v3['test']
extra=pairs(recall)-pairs(a);assert not pairs(a)-pairs(recall);assert sum(extra.values())==4000;assert not extra&sum((pairs(x) for x in v3.values()),Counter())
keys={canon(x['url']) for x in g};assert len(keys)==2244
assert [sum(canon(x['url']) in keys for x in v3[p]) for p in initial]==[1636,329,279]
assert [len(v3[p]) for p in initial]==[35664,7668,7653]
replay=json.loads((W/'augmentation_replay_report.json').read_text());assert replay['membership_multiset_matches_saved'];assert replay['candidate_rows']==replay['retained_rows']==2244
print('CHECK PASSED: all fingerprints, cleaning row accounting, 631 source-audit removals, final overlap removal, generated membership and counterfactual disposition.')
print('Full source-label and generation replays are documented separately. No files changed, training, inference or network access.')
