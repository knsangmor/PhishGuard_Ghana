from pathlib import Path
import runpy,random,csv,json,hashlib
from collections import Counter
W=Path(__file__).resolve().parent;B=W/'collected';ns=runpy.run_path(str(B/'build_modern_legit_augmentation.py'),run_name='provenance_replay_only')
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
norm=ns['normalize_for_dedup'];rng=random.Random(42);domains=ns['load_tranco'](str(W/'top-1m_recovered.csv'));sampled=ns['stratified_sample'](domains,1500,rng);candidates=ns['build_augmentation'](sampled,2,rng)
base=rows(W/'base_sampling_replayed.csv');existing={norm(x['url']) for x in base};seen=set();kept=[];reject_existing=reject_repeat=0
for u in candidates:
 k=norm(u)
 if k in existing:reject_existing+=1;continue
 if k in seen:reject_repeat+=1;continue
 seen.add(k);kept.append({'url':u,'label':'0'})
saved=rows(B/'modern_legit_augmentation.csv');pairs=lambda rs:Counter((x['url'],int(x['label'])) for x in rs)
r={'tranco_sha256':hashlib.sha256((W/'top-1m_recovered.csv').read_bytes()).hexdigest(),'tranco_bytes':(W/'top-1m_recovered.csv').stat().st_size,'tranco_rows':len(domains),'sampled_domains':len(sampled),'seed':42,'paths_per_domain':2,'candidate_rows':len(candidates),'rejected_existing_key':reject_existing,'rejected_batch_key':reject_repeat,'retained_rows':len(kept),'saved_rows':len(saved),'membership_multiset_matches_saved':pairs(kept)==pairs(saved),'historical_execution_proven':False,'note':'Replay against reconstructed source-sampled base membership and exact Tranco LFS target; output membership can match while row order differs because the generator iterates sets.'}
rng2=random.Random(42)
again=ns['stratified_sample'](domains,1500,rng2);drawn=collapsed=0
for rank,domain in again:
 k=rng2.randint(1,2);paths=rng2.sample(ns['PATH_TEMPLATES'],min(k,len(ns['PATH_TEMPLATES'])));drawn+=len(paths);collapsed+=len(paths)-len(set(paths));rng2.random()
r.update(template_entries_drawn=drawn,within_domain_repeated_paths_collapsed=collapsed,sampled_unique_domains=len({d for _,d in sampled}))
(W/'augmentation_replay_report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
