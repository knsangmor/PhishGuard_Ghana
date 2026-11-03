import zipfile,json,io
from pathlib import Path
import pandas as pd,numpy as np
from urllib.parse import urlsplit
OUT=Path(__file__).resolve().parent
z=zipfile.ZipFile(OUT.parent/'upload/shortcut_checks_v3.zip');main=zipfile.ZipFile(OUT.parent/'upload/corrected_results_v3.zip')
allrows=[];lengths=[];seeds=[]
def bare(u):
 u=u.strip()
 for p in ['https://','http://']:
  if u.lower().startswith(p):return u[len(p):]
 return u
for name in z.namelist():
 if not name.endswith('shortcut_report.json'):continue
 r=json.loads(z.read(name));seed=r['seed'];seeds.append(seed);prefix=name.rsplit('/',1)[0]+'/'
 assert r['original_verdict_agreement_with_saved']==1
 assert all(v['encoding_mismatches']==0 for v in r['scheme_encoding_checks'].values())
 saved=pd.read_csv(io.BytesIO(main.read(f'corrected_results_v3/seed_{seed}/test_predictions.csv')),keep_default_na=False)
 summary=pd.read_csv(io.BytesIO(z.read(prefix+'perturbation_summary.csv')));summary['seed']=seed
 for kind in ['toggle_www','toggle_root_slash','append_path_256_chars']:
  d=pd.read_csv(io.BytesIO(z.read(prefix+kind+'_details.csv')),keep_default_na=False)
  assert np.array_equal(d.url.to_numpy(),saved.iloc[d.test_row].url.to_numpy())
  assert np.max(abs(d.original_probability.to_numpy()-saved.iloc[d.test_row].probability.to_numpy()))<1e-5
  assert np.array_equal((d.original_probability>=.5)!=(d.perturbed_probability>=.5),d.verdict_flipped)
  for row in d.itertuples():
   a=urlsplit('https://'+bare(row.url).removeprefix('//'));b=urlsplit('https://'+row.perturbed_url)
   if kind=='toggle_www':
    netloc=a.netloc[4:] if a.netloc.lower().startswith('www.') else 'www.'+a.netloc
    assert b.netloc==netloc and (a.path,a.query,a.fragment)==(b.path,b.query,b.fragment)
   else:
    path=('/' if a.path=='' else '') if kind=='toggle_root_slash' else a.path.rstrip('/')+'/'+'a'*256
    assert b.path==path and (a.netloc,a.query,a.fragment)==(b.netloc,b.query,b.fragment)
  for group,mask in [('all',np.ones(len(d),bool)),('original_benign',d.original_label==0),('original_malicious',d.original_label==1)]:
   line=summary[(summary.probe==kind)&(summary.original_class_group==group)].iloc[0]
   assert line.eligible_urls==sum(mask) and line.verdict_flips==d.loc[mask,'verdict_flipped'].sum()
 allrows.append(summary)
 length=pd.read_csv(io.BytesIO(z.read(prefix+'original_performance_by_length.csv')));length['seed']=seed;lengths.append(length)
assert sorted(seeds)==list(range(42,47))
a=pd.concat(allrows);a.to_csv(OUT/'per_seed_shortcut_results.csv',index=False)
g=a.groupby(['probe','original_class_group']).agg(eligible_urls=('eligible_urls','first'),mean_flip_rate=('flip_rate','mean'),sd_flip_rate=('flip_rate','std'),min_flip_rate=('flip_rate','min'),max_flip_rate=('flip_rate','max'),mean_probability_change=('mean_absolute_probability_change','mean'))
g.to_csv(OUT/'shortcut_summary.csv')
l=pd.concat(lengths);l.to_csv(OUT/'per_seed_length_results.csv',index=False)
l.groupby('length_group').agg(urls=('rows','first'),mean_accuracy=('accuracy','mean'),sd_accuracy=('accuracy','std'),mean_recall=('recall','mean'),mean_fpr=('fpr','mean')).to_csv(OUT/'length_summary.csv')
(OUT/'interpretation.txt').write_text('''Five saved main-model seeds, 42–46. All 7,653 original test predictions in every run agree in verdict with the saved results; probability differences below 1e-5. Details reproduce summary counts; transformations preserve all parsed components except the intended component. HTTP, HTTPS and schemeless variants have identical encodings for every test URL.\n\nwww and root slash sensitivity is substantial across all five seeds. Flip rates describe changes in predicted verdict, not changes in measured accuracy or verified evasion. Perturbed URL resources and labels were not independently verified. Non-applicable URLs are excluded from each probe. Mean and SD describe seed variation on the same eligible URLs, not independent samples of deployment traffic.\n\nThe encoding-level scheme fix is demonstrated, but no causal claim about generalization improvement or the augmentation's benefit follows. Before/after correction, separate training-fix ablations and validated adaptive attacks remain outstanding. Length subgroup associations do not establish causal effects. Keep the frozen dataset and model outputs; do not silently retrofit normalization to the evaluated models.\n''')
print(g.to_string())
