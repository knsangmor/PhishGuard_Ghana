"""Descriptive paired error analysis of already-scored holdout predictions.
No model loading, inference, training, threshold changes or URL network access.
"""
import argparse,json,re,hashlib
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def features(url):
    text=str(url).strip();rest=re.sub(r'^https?://','',text,flags=re.I)
    length=len(rest.lower().strip())
    try:
        p=urlsplit('https:'+rest if rest.startswith('//') else 'https://'+rest)
        host=(p.hostname or '').lower();path=p.path
        valid=bool(host)
    except ValueError:host='';path='';valid=False
    length_bin='<=50' if length<=50 else '51-100' if length<=100 else '101-200' if length<=200 else '>200'
    return {'format':('www' if host.startswith('www.') else 'no_www')+'_'+('empty_path' if path=='' else 'root_slash' if path=='/' else 'other_path') if valid else 'unparsed',
            'length_bin':length_bin,'scheme': 'https' if text.lower().startswith('https://') else 'http' if text.lower().startswith('http://') else 'other_or_absent'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true');mode.add_argument('--analyze',action='store_true')
    mode.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:
        assert features('https://www.example.com')['format']=='www_empty_path'
        assert features('http://example.com/')['format']=='no_www_root_slash'
        assert features('example.com/a')['format']=='no_www_other_path'
        print('SELF-TEST PASSED: URL format grouping.');return
    source=ROOT/'reserved_holdout_results_v1';report=json.loads((source/'report.json').read_text())
    if report['seeds']!=[42,43,44,45,46] or report['holdout_rows']!=7659:raise ValueError('Unexpected evaluation')
    if not (source/'COMPLETE.txt').is_file():raise ValueError('Evaluation incomplete')
    frames={};reference=None
    for family in ['original','broader_data']:
        for seed in report['seeds']:
            key=f'{family}_seed_{seed}';path=source/f'{key}_predictions.csv'
            if sha(path)!=report['prediction_sha256'][key]:raise ValueError('Prediction hash differs: '+key)
            f=pd.read_csv(path,keep_default_na=False)
            if reference is None:reference=f[['url','true_label']].copy()
            if not f[['url','true_label']].equals(reference):raise ValueError('Prediction alignment differs')
            p=f.probability.to_numpy(dtype=float)
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
            frames[(family,seed)]=p
    print('CHECK PASSED: all ten saved prediction hashes and paired row alignment.')
    if a.check_only:print('No analysis output folders created.');return
    y=reference.true_label.to_numpy(dtype=int)
    feat=pd.DataFrame([features(u) for u in reference.url]);paired=[];groups=[]
    per_url=reference.copy()
    for op in ['default_0.5','fpr_cap_0.01','fpr_cap_0.005']:
        counts={family:np.zeros(len(y),dtype=int) for family in ['original','broader_data']}
        for seed in report['seeds']:
            correct={}
            for family in ['original','broader_data']:
                threshold=report['models'][f'{family}_{seed}']['thresholds'][op]
                pred=(frames[(family,seed)]>=threshold).astype(int)
                correct[family]=pred==y;counts[family]+=~correct[family]
                for dimension in feat.columns:
                    for group in sorted(feat[dimension].unique()):
                        mask=feat[dimension].eq(group).to_numpy();benign=mask&(y==0);malicious=mask&(y==1)
                        fp=int(np.sum(pred[benign]==1));fn=int(np.sum(pred[malicious]==0));nb=int(benign.sum());nm=int(malicious.sum())
                        groups.append(dict(family=family,seed=seed,operating_point=op,dimension=dimension,group=group,rows=int(mask.sum()),benign=nb,malicious=nm,false_positives=fp,false_negatives=fn,false_positive_rate=fp/nb if nb else None,recall=1-fn/nm if nm else None))
            old=correct['original'];new=correct['broader_data']
            for label,name in [(None,'all'),(0,'benign'),(1,'phishing')]:
                mask=np.ones(len(y),dtype=bool) if label is None else y==label
                paired.append(dict(seed=seed,operating_point=op,class_group=name,rows=int(mask.sum()),both_correct=int(np.sum(mask&old&new)),corrected_by_broader=int(np.sum(mask&~old&new)),introduced_by_broader=int(np.sum(mask&old&~new)),both_wrong=int(np.sum(mask&~old&~new))))
        for family in counts:per_url[f'{family}_{op}_errors_out_of_5']=counts[family]
    out=ROOT/'holdout_error_analysis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True,exist_ok=False)
    pair=pd.DataFrame(paired);group=pd.DataFrame(groups)
    pair.to_csv(out/'paired_errors_per_seed.csv',index=False);group.to_csv(out/'format_errors_per_seed.csv',index=False)
    per_url=pd.concat([per_url,feat],axis=1);per_url.to_csv(out/'url_error_counts.csv',index=False)
    group_summary=group.groupby(['family','operating_point','dimension','group'],sort=False).agg(rows=('rows','first'),benign=('benign','first'),malicious=('malicious','first'),mean_false_positive_rate=('false_positive_rate','mean'),sd_false_positive_rate=('false_positive_rate','std'),mean_recall=('recall','mean'),sd_recall=('recall','std')).reset_index()
    group_summary.to_csv(out/'format_error_summary.csv',index=False)
    mean_pair=pair.groupby(['operating_point','class_group'])[['both_correct','corrected_by_broader','introduced_by_broader','both_wrong']].mean()
    mean_pair.to_csv(out/'paired_error_summary.csv')
    notes={'purpose':'Post-evaluation descriptive diagnosis; no model or threshold selection.',
           'input_report_sha256':sha(source/'report.json'),'holdout_sha256':report['holdout_sha256'],
           'limitations':['URL groups overlap across dimensions; do not sum across dimensions.','Formatting associations do not establish causes or validate source labels.','Small groups can have unstable rates.','The same URLs recur across seeds; these are not independent additional test examples.','No statistical significance claim or holdout tuning performed.','Future models informed by this diagnosis need a new untouched confirmation set.']}
    (out/'report.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print('\nPAIRED ERRORS: MEAN COUNTS PER SEED')
    print(mean_pair.to_string(float_format='%.2f'))
    print('\nSaved:',out)
    print('No training, inference, threshold changes or network requests.')

if __name__=='__main__':main()
