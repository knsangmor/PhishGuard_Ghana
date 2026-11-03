"""Collect PhishGuard provenance evidence without modifying inputs or starting training.
Run from the project folder: python collect_provenance_point1.py
"""
import argparse,csv,hashlib,io,json,random,runpy,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pairs(rs):return Counter((r['url'],int(r['label'])) for r in rs)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path.cwd());args=ap.parse_args();root=args.root.resolve()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out=root/'provenance_point1_collection'/stamp;out.mkdir(parents=True,exist_ok=False)
 names=['malicious_phish.csv','top-1m.csv','phishguard_gh_dataset.csv','phishguard_gh_dataset_augmented.csv','phishguard_gh_dataset_recall.csv','modern_legit_augmentation.csv','build_dataset.py','build_modern_legit_augmentation.py','recover_phishing_recall.py','clean_dataset.py','split_dataset.py','check_augmentation_provenance.py','create_v3_partitions.py','verify_v2_partitions.py','pg_fix.py']
 selected=[root/n for n in names if (root/n).is_file()]
 for folder in ['dataset_fix_outputs','dataset_fix_outputs_v2','dataset_fix_outputs_v3']:
  p=root/folder
  if p.is_dir():
   for f in p.iterdir():
    if f.is_file() and (f.name in ['train.csv','validation.csv','test.csv','dataset_cleaned.csv','conflicting_records.csv','removed_duplicates.csv','invalid_hostname_records.csv'] or f.suffix.lower() in ['.json','.txt'] or any(s in f.name.lower() for s in ['quarantin','exclu','ambig','overlap'])):selected.append(f)
 manifest=[]
 for p in selected:
  item={'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
  if p.stat().st_size<512:
   item['git_lfs_pointer']=p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1')
  manifest.append(item)
 report={'root_folder_name':root.name,'inputs_modified':False,'training_started':False,'files':manifest,'missing_root_files':[n for n in names if not (root/n).is_file()]}
 # Replay only the supplied data-generation functions, using their local saved source and inputs.
 # run_name does not execute the script's main; no detector or training module is imported.
 required=['top-1m.csv','phishguard_gh_dataset.csv','modern_legit_augmentation.csv','build_modern_legit_augmentation.py']
 if all((root/n).is_file() for n in required):
  try:
   if (root/'top-1m.csv').read_bytes()[:80].startswith(b'version https://git-lfs'):raise ValueError('Tranco input is only an LFS pointer')
   ns=runpy.run_path(str(root/'build_modern_legit_augmentation.py'),run_name='provenance_function_audit')
   rng=random.Random(42);domains=ns['load_tranco'](str(root/'top-1m.csv'));sampled=ns['stratified_sample'](domains,1500,rng);candidates=ns['build_augmentation'](sampled,2,rng)
   original=rows(root/'phishguard_gh_dataset.csv');saved=rows(root/'modern_legit_augmentation.csv');norm=ns['normalize_for_dedup'];existing={norm(r['url']) for r in original};seen=set();kept=[];rejected_existing=0;rejected_batch=0
   for u in candidates:
    k=norm(u)
    if k in existing:rejected_existing+=1;continue
    if k in seen:rejected_batch+=1;continue
    seen.add(k);kept.append({'url':u,'label':'0'})
   equality=pairs(kept)==pairs(saved)
   replay={'parameters':{'seed':42,'n_domains':1500,'paths_per_domain':2},'domains_sampled':len(sampled),'candidate_rows':len(candidates),'rejected_existing_key':rejected_existing,'rejected_batch_key':rejected_batch,'retained_rows':len(kept),'saved_rows':len(saved),'saved_membership_multiset_matches_replay':equality,'historical_candidate_counts_validated':False,'interpretation':'A membership match supports replay with these inputs but does not prove historical execution; mismatch means do not use these replay counts as historical counts.'}
   report['augmentation_replay']=replay
  except Exception as e:report['augmentation_replay']={'completed':False,'error':str(e),'historical_candidate_counts_validated':False}
 (out/'collection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 archive=out/'provenance_point1_evidence.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(out/'collection_report.json','collection_report.json')
  for p in selected:z.write(p,'project/'+p.relative_to(root).as_posix())
 print('PROVENANCE COLLECTION COMPLETE')
 print('Files collected:',len(selected));print('Report:',out/'collection_report.json');print('Upload this ZIP:',archive)
 if 'augmentation_replay' in report:print(json.dumps(report['augmentation_replay'],indent=2))
 print('No input files changed. No training, inference or network access performed.')
if __name__=='__main__':main()
