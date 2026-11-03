"""Build a private, explicitly scoped release candidate. No training, inference or publishing.
Use --check-only first. Original project files are read only.
"""
import argparse,csv,hashlib,importlib.metadata,io,json,os,platform,shutil,sys,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

SEEDS=range(42,47)
PARTS={'train':(35664,'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f'),'validation':(7668,'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e'),'test':(7653,'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f')}
BASELINES=['RF_full','RF_no_scheme','RF_no_scheme_no_ghana','XGB_full','XGB_no_scheme','XGB_no_scheme_no_ghana','CNN','BiLSTM']
ROOT_REQUIRED=['model_hash_v3.py','baseline_comparison_v3.py','generate_corrected_features.py']
# Deliberately selected research directories. No home directory, virtual environment,
# browser profile, git credentials or unlisted project directory is scanned.
MODULES={
 'dataset_fix_outputs_v3':'frozen partitions and numeric features',
 'corrected_results_v3':'original five main-model runs',
 'baseline_results_v3':'original tuned baseline family',
 'category_evaluation_v3':'source-category audit',
 'rule_evaluation_v3':'rule audit and worked examples',
 'shortcut_checks_v3':'format probes',
 'controlled_experiments_v3':'single-seed sensitivity experiments',
 'cpu_benchmarks_v3':'stage timing and plain-logging control',
 'adaptive_evasion_v3':'offline synthetic probes',
 'external_phiusiil_v3':'initial external/development corpus evaluation',
 'coverage_frozen_v1':'broader-data frozen model family',
 'original_coverage_frozen_v1':'original frozen threshold family',
 'reserved_holdout_results_v1':'separate Mendeley holdout evaluation',
 'transformer_baseline_v3':'compact Transformer runs and timing',
 'published_baseline_results':'published encoder protocol/checkpoint and evaluation',
 'published_cpu_comparison_v3':'matched published-model CPU timing',
 'controlled_robustness_v1':'controlled-fixture stress tests',
 'workflow_preservation_audit':'retrospective workflow replay',
 'masked_bilstm_control_v1':'five masked BiLSTM controls and evaluation',
}
EXTRA_DIRS=['root_normalization_preflight','root_normalized_pilot_v3','formatting_diagnosis_v3','coverage_validation_comparison','model_threshold_comparison','coverage_threshold_analysis','experiment_preflight_v3_reports','tamper_tests_v3','tamper_tests_v3_results','category_evaluation_v3','attribution_audit_v3','shap_audit_v3','keyed_logging_v3','hmac_evaluation_v3','provenance_point1_collection']
EXT={'.py','.ipynb','.csv','.json','.jsonl','.txt','.md','.npy','.npz','.keras','.h5','.pkl','.joblib','.pt','.safetensors','.png','.jpg','.pdf','.yaml','.yml','.toml'}
# Large original corpora and provider lists should be supplied under their own terms.
SKIP_NAMES={'malicious_phish.csv','top-1m.csv','top-1m_recovered.csv','PhiUSIIL_Phishing_URL_Dataset.csv'}


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def read_json(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csv_rows(p):
 with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def flag(s):
 if str(s).lower() in ('1','true'):return 1
 if str(s).lower() in ('0','false'):return 0
 raise ValueError('Invalid binary verdict')

def score(rows):
 pairs=[(float(r['probability']),int(r['true_label'])) for r in rows]
 if any(not 0<=p<=1 or y not in (0,1) for p,y in pairs):raise ValueError('Invalid score or label')
 if any(flag(r['pred_label'])!=int(float(r['probability'])>=.5) for r in rows):raise ValueError('Verdict differs from threshold .5')
 tp=sum(p>=.5 and y==1 for p,y in pairs);fp=sum(p>=.5 and y==0 for p,y in pairs);pos=sum(y for _,y in pairs);neg=len(pairs)-pos
 if not pos or not neg:raise ValueError('Both test classes required')
 precision=tp/(tp+fp) if tp+fp else 0;recall=tp/pos
 # Group tied scores. Trapezoidal ROC and non-interpolated AP match standard definitions.
 groups={}
 for p,y in pairs:
  a=groups.setdefault(p,[0,0]);a[y]+=1
 ct=cf=0;roc=ap=0.0
 for p,(n0,n1) in sorted(groups.items(),reverse=True):
  oldt,oldf=ct,cf;ct+=n1;cf+=n0
  roc+=(cf-oldf)/neg*((ct+oldt)/pos)/2
  ap+=n1/pos*ct/(ct+cf)
 return {'accuracy':(tp+neg-fp)/len(pairs),'precision':precision,'recall':recall,'f1_score':2*precision*recall/(precision+recall) if precision+recall else 0,'auc_roc':roc,'average_precision':ap,'false_positive_rate':fp/neg}

def aligned(rows,part):
 if len(rows)!=len(part) or any(r['url']!=d['url'] or int(r['true_label'])!=int(d['label']) for r,d in zip(rows,part)):raise ValueError('Prediction URL/label order differs from frozen partition')

def inspect(root):
 errors=[];warnings=[];frames={};checks=[]
 report={'scope':'Release candidate preparation, not a public release or DOI. Validation covers original core metrics and stated file checks; presence-only inventories do not prove reproduction of every later experiment.','root':str(root),'core_errors':errors,'gaps':warnings,'checks':checks,'modules':{},'publication_ready':False}
 def require(p):
  if not p.is_file():raise FileNotFoundError(str(p.relative_to(root)))
  if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('Symlink or out-of-root file refused')
 for n in ROOT_REQUIRED:
  try:require(root/n)
  except Exception as e:errors.append(str(e))
 for n,(size,h) in PARTS.items():
  try:
   p=root/'dataset_fix_outputs_v3'/f'{n}.csv';require(p)
   if digest(p)!=h:raise ValueError('Frozen partition fingerprint differs: '+n)
   rows=csv_rows(p)
   if len(rows)!=size or any(r['label'] not in ('0','1') for r in rows) or len({r['url'] for r in rows})!=size:raise ValueError('Partition size/labels/unique URLs differ: '+n)
   frames[n]=rows
  except Exception as e:errors.append(str(e))
 if len(frames)==3:checks.append('Exact frozen partition fingerprints, counts, labels and unique URLs passed')
 try:
  protocol=read_json(root/'baseline_results_v3/protocol.json')
  if digest(root/'baseline_comparison_v3.py')!=protocol['script_sha256']:raise ValueError('Baseline source differs from recorded protocol')
  if protocol['partition_hashes']!={n:h for n,(_,h) in PARTS.items()}:raise ValueError('Baseline partition protocol differs')
  for n in PARTS:
   p=root/'dataset_fix_outputs_v3'/f'{n}_features.csv';require(p)
   if digest(p)!=protocol['feature_file_hashes'][n]:raise ValueError('Recorded feature hash differs: '+n)
  require(root/'baseline_results_v3/selected_configs.json');checks.append('Baseline source and all recorded feature hashes passed')
 except Exception as e:errors.append(str(e))
 for seed in SEEDS:
  try:
   run=root/'corrected_results_v3'/f'seed_{seed}';m=read_json(run/'metrics.json')
   if m['seed']!=seed or m['dataset_version']!='v3':raise ValueError('Wrong main-model identity')
   for n,(_,h) in PARTS.items():
    if m['partitions'][n]['sha256']!=h:raise ValueError('Main-model partition identity differs')
   for name in ['phishguard_gh_model.keras','phishguard_gh_model.h5','model_fingerprint.txt','predictions_hash.txt','predictions.csv'] :require(run/name)
   for filename,record in [('phishguard_gh_model.h5','model_fingerprint.txt'),('predictions.csv','predictions_hash.txt')]:
    if digest(run/filename)!=read_json(run/record)['sha256']:raise ValueError('Recorded artifact fingerprint differs: '+filename)
   full=csv_rows(run/'predictions.csv');membership={r['url']:int(r['label']) for d in frames.values() for r in d}
   if len(full)!=len(membership) or len({r['url'] for r in full})!=len(membership) or any(r['url'] not in membership or int(r['true_label'])!=membership[r['url']] for r in full):raise ValueError('Full prediction snapshot membership differs')
   index={r['url']:r for r in full}
   for n in ('validation','test'):
    rows=csv_rows(run/f'{n}_predictions.csv');aligned(rows,frames[n])
    if any(abs(float(r['probability'])-float(index[r['url']]['probability']))>1e-6 or flag(r['pred_label'])!=flag(index[r['url']]['pred_label']) for r in rows):raise ValueError('Full/split snapshot score or verdict differs')
   vals=score(csv_rows(run/'test_predictions.csv'))
   for k in ('accuracy','precision','recall','f1_score','auc_roc'):
    if abs(vals[k]-float(m[k]))>1e-5:raise ValueError('Main metric differs: '+k)
   checks.append(f'Main seed {seed}: recorded H5/full-prediction hashes, full/split alignment and five metrics reproduced')
  except Exception as e:errors.append(f'Main seed {seed}: {e}')
  for model in BASELINES:
   try:
    run=root/'baseline_results_v3'/f'seed_{seed}'/model;require(run/('model.keras' if model in ('CNN','BiLSTM') else 'model.pkl'))
    m=read_json(run/'metrics.json')
    for n in ('validation','test'):aligned(csv_rows(run/f'{n}_predictions.csv'),frames[n])
    vals=score(csv_rows(run/'test_predictions.csv'))
    for k,v in m['test_metrics'].items():
     if k in vals and abs(vals[k]-float(v))>1e-8:raise ValueError('Baseline metric differs: '+k)
   except Exception as e:errors.append(f'Baseline {seed}/{model}: {e}')
 if not errors:checks.append('All forty original baseline runs: required checkpoints, prediction alignment and recorded metrics passed')
 for folder,label in MODULES.items():
  p=root/folder
  report['modules'][folder]={'purpose':label,'present':p.is_dir(),'validation':'presence only, except original core checks listed above'}
  if not p.is_dir():warnings.append('Missing study directory: '+folder)
 # Masked completion files carry hashes. Preserve every attempt, but require one completed run per seed.
 for seed in SEEDS:
  completed=list((root/'masked_bilstm_control_v1').glob(f'seed_{seed}_*/training_complete.json'))
  if len(completed)!=1:warnings.append(f'Masked seed {seed}: expected one completed attempt, found {len(completed)}');continue
  try:
   p=completed[0];m=read_json(p)
   for name,key in [('model.keras','model_sha256'),('validation_predictions.csv','validation_predictions_sha256')]:
    if digest(p.parent/name)!=m[key]:raise ValueError(name+' hash differs')
  except Exception as e:warnings.append(f'Masked seed {seed}: {e}')
 # Colab-trained published models may still be only in Drive.
 for seed in SEEDS:
  hits=[p for p in (root/'published_baseline_results').rglob('best_model.pt') if f'seed_{seed}' in p.parts] if (root/'published_baseline_results').is_dir() else []
  if not hits:warnings.append(f'Published baseline seed {seed}: seed-labelled best_model.pt missing from published_baseline_results (a root best_model.pt cannot establish its seed)')
 warnings.extend(['Confirm code/model redistribution licences and source-specific data terms before public deposit. No licence is assigned by this builder.','Documentary outputs stored under unlisted folder names must be added explicitly with --extra-dir; presence-only inventories require a final manuscript-to-assets review.','Checkpoints are hashed, not executed; fresh inference and full retraining reproduction are not performed by this builder.','A public repository tag/DOI, public accessibility and author confirmation remain pending.'])
 report['core_passed']=not errors
 return report

def collect(root,extra):
 paths=[]
 # Only root-level Python research scripts and explicitly named documentation.
 for p in sorted(root.glob('*.py')):
  if not p.is_symlink():paths.append(p)
 for n in ['README.md','LICENSE','LICENSE.txt','NOTICE','CITATION.cff','requirements.txt']:
  p=root/n
  if p.is_file():paths.append(p)
 for folder in dict.fromkeys(list(MODULES)+EXTRA_DIRS+extra):
  d=(root/folder).resolve()
  if not d.is_relative_to(root) or d==root:raise ValueError('Extra directory must be a named directory inside the project')
  if not d.is_dir():continue
  for parent,dirs,files in os.walk(d,followlinks=False):
   dirs[:]=sorted(n for n in dirs if not n.startswith('.') and n!='__pycache__' and not (Path(parent)/n).is_symlink())
   for name in sorted(files):
    p=Path(parent)/name
    if p.suffix.lower() in EXT and name not in SKIP_NAMES and not p.is_symlink() and p.resolve().is_relative_to(root):paths.append(p)
 return sorted(set(paths),key=lambda p:p.relative_to(root).as_posix())

def build(root,out,report,extra):
 if report['core_errors']:raise ValueError('Core checks failed; no candidate archive created')
 files=collect(root,extra);total=sum(p.stat().st_size for p in files)
 if shutil.disk_usage(out.parent if out.parent.exists() else root).free<total+104857600:raise ValueError('Insufficient free space for conservative uncompressed-size estimate')
 out.mkdir(parents=True,exist_ok=False)
 manifest=[]
 for i,p in enumerate(files,1):
  manifest.append({'path':p.relative_to(root).as_posix(),'size_bytes':p.stat().st_size,'sha256':digest(p)})
  if i%100==0:print(f'Hashed {i}/{len(files)} files',flush=True)
 env={}
 for name in ['numpy','pandas','tensorflow','keras','scikit-learn','xgboost','tldextract','shap','torch','transformers','tokenizers','safetensors']:
  try:env[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:env[name]=None
 release={'candidate_created_utc':datetime.now(timezone.utc).isoformat(),'public_release':False,'doi':None,'release_tag':None,'scope_report':report,'preparation_environment':{'python':sys.version,'platform':platform.platform(),'packages':env},'environment_note':'Installed packages at preparation time; not proof of every historical training environment.','files':manifest}
 readme='''# PhishGuard revised artifact release candidate\n\nThis is a private release candidate, not an issued journal release. No public tag or DOI is assigned. Study scope and missing assets are listed in RELEASE_MANIFEST.json. Read that report before making any completeness claim.\n\n## Verification without training\n\nRun `python prepare_release_v1.py --verify PATH_TO_CANDIDATE_ZIP` to validate every archived file against the enclosed manifest. An unkeyed manifest verifies consistency with that manifest; retain the outer ZIP checksum through a trusted release page for a separate reference.\n\nThe builder already checked exact frozen v3 partition hashes, recorded feature hashes, original saved-prediction alignment and original core metrics. It does not execute pickle/checkpoint files, reproduce every later experiment, or promise bitwise-identical retraining. No listed malicious URL needs to be visited.\n\n## Reproducing the original core\n\nExtract into a NEW project folder. Keep frozen inputs and results intact. The original main model is a CNN-LSTM with a bidirectional recurrent block, 244,673 parameters. Original main and baseline runs use seeds 42–46. Use saved predictions first. `model_hash_v3.py --check-only` and `baseline_comparison_v3.py --check-only` perform their original environment-dependent preflights. Their training modes create outputs; use a separate working copy for any rerun. All later studies retain their own protocols. The masked BiLSTM is a later exploratory control, not a replacement of the original BiLSTM.\n\n## Data and software terms\n\nThe base corpus is Siddharth, Malicious URLs Dataset (Kaggle, 2021). PhiUSIIL is Prasad and Chandra, Computers & Security (2023), DOI 10.1016/j.cose.2023.103545; its UCI dataset record and the separate Mendeley holdout have their own attribution terms. Tranco supplied the historical domain list. Preserve source notices; no single blanket licence is assigned here. Root raw corpus/provider-list files are excluded, while frozen partitions and selected research outputs are included for author review. Before public deposit, verify rights for each distributed component and choose an author-approved licence for original code and models.\n\n## Remaining release steps\n\nResolve every asset gap in the manifest; verify the manuscript-to-assets mapping and licences; add seed-labelled published-model checkpoints and their completion/protocol records where missing; confirm author/ethics declarations; deposit the final assets, record their checksums and actual public tag or DOI, and then update manuscript availability statements. Do not substitute a candidate filename for a real public release identifier.\n'''
 (out/'RELEASE_MANIFEST.json').write_text(json.dumps(release,indent=2,allow_nan=False),encoding='utf-8')
 archive=out/'PhishGuard_Revision_Release_Candidate.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p,entry in zip(files,manifest):
   if digest(p)!=entry['sha256']:raise ValueError('Input changed while packaging: '+entry['path'])
   z.write(p,entry['path'])
  z.writestr('RELEASE_MANIFEST.json',json.dumps(release,indent=2,allow_nan=False));z.writestr('RELEASE_README.md',readme)
 (out/'ZIP_SHA256.txt').write_text(digest(archive)+'  '+archive.name+'\n',encoding='ascii')
 verify(archive);print('CANDIDATE BUILT:',archive);print('Publication gaps:',len(report['gaps']),'No files published; original project unchanged.')

def verify(path):
 with zipfile.ZipFile(path) as z:
  manifest=json.loads(z.read('RELEASE_MANIFEST.json'));names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP paths')
  expected={r['path'] for r in manifest['files']}|{'RELEASE_MANIFEST.json','RELEASE_README.md'}
  if set(names)!=expected:raise ValueError('Archive contains unmanifested or missing files')
  for r in manifest['files']:
   p=r['path']
   if p.startswith('/') or '..' in Path(p).parts:raise ValueError('Unsafe archive path')
   b=z.read(p)
   if len(b)!=r['size_bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('Archive fingerprint mismatch: '+p)
 print('ARCHIVE VERIFICATION PASSED:',len(manifest['files']),'files. Consistency check only; public-release status remains pending.')

def main():
 ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--check-only',action='store_true');mode.add_argument('--build',action='store_true');mode.add_argument('--verify',type=Path);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path);ap.add_argument('--extra-dir',action='append',default=[]);a=ap.parse_args()
 if a.verify:verify(a.verify);return
 root=a.root.resolve();report=inspect(root);files=collect(root,a.extra_dir);report['candidate_file_count']=len(files);report['candidate_uncompressed_bytes']=sum(p.stat().st_size for p in files)
 print(json.dumps(report,indent=2,allow_nan=False));print('CORE CHECK PASSED' if report['core_passed'] else 'CORE CHECK FAILED: resolve core_errors before building')
 if a.check_only:print('READ-ONLY CHECK COMPLETE. No output folders, training, inference or publication.');return
 out=(a.output or root/('release_candidate_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
 if out==root or any(out==root/n or (root/n) in out.parents for n in list(MODULES)+EXTRA_DIRS):raise ValueError('Output must be a NEW separate release directory')
 build(root,out,report,a.extra_dir)
if __name__=='__main__':main()
