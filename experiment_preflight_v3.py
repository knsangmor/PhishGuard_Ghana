"""Audit proposed experiments; writes reports only, never datasets or models."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HASHES = {
    'train': '28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
    'validation': '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
    'test': '22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f',
}
SOURCE_HASH = 'd83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'
SETTINGS = [('strip_100',100,True), ('strip_200',200,True), ('strip_300',300,True), ('retain_scheme_200',200,False)]
MALICIOUS = {'phishing','malware','defacement','malicious','1'}
BENIGN = {'benign','0'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as file:
        reader=csv.DictReader(file)
        if not {'url','label'}.issubset(reader.fieldnames or []):raise ValueError(f'Missing columns: {path}')
        rows=list(reader)
    for row in rows:
        if not row['url'].strip() or row['label'].strip() not in {'0','1'}:raise ValueError(f'Invalid row: {path}')
    return rows

def canonical(url):
    text=url.strip()
    # Match the previously used cleaning key exactly.
    if text.startswith('https://'):text=text[8:]
    elif text.startswith('http://'):text=text[7:]
    return text.lower().strip()

def code(url,length,strip):
    text=url.strip()
    if strip:
        for prefix in ('https://','http://'):
            if text.lower().startswith(prefix):
                text=text[len(prefix):];break
    text=text.lower().strip()[:length]
    tokens=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in text]
    tokens.extend([0]*(length-len(tokens)))
    return hashlib.sha256(bytes(tokens)).digest()

def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Check file identities only; no report directories')
    args=parser.parse_args()
    folder=ROOT/'dataset_fix_outputs_v3'
    source=ROOT/'malicious_phish.csv'
    augmentation_path=ROOT/'modern_legit_augmentation.csv'
    for name,expected in HASHES.items():
        if sha(folder/f'{name}.csv')!=expected:raise ValueError(f'Version 3 fingerprint mismatch: {name}')
    if sha(source)!=SOURCE_HASH:raise ValueError('Original corpus hash mismatch (or Git LFS pointer instead of CSV)')
    augmentation=read(augmentation_path)
    generated={canonical(row['url']) for row in augmentation}
    if len(augmentation)!=2244 or len(generated)!=2244 or any(row['label'].strip()!='0' for row in augmentation):
        raise ValueError('Expected 2,244 unique benign-labelled additions')
    parts={name:read(folder/f'{name}.csv') for name in HASHES}
    sizes={'train':35664,'validation':7668,'test':7653}
    if any(len(parts[name])!=sizes[name] for name in sizes):raise ValueError('Unexpected partition size')
    feature_hashes={}
    for name in parts:
        features=read(folder/f'{name}_features.csv')
        if [(r['url'],r['label']) for r in parts[name]] != [(r['url'],r['label']) for r in features]:
            raise ValueError(f'Feature URL/label alignment failed: {name}')
        numeric=[k for k in features[0] if k not in {'url','label'}]
        if len(numeric)!=23:raise ValueError('Expected 23 features')
        import math
        if any(not math.isfinite(float(r[k])) for r in features for k in numeric):raise ValueError('Invalid numeric features')
        feature_hashes[name]=sha(folder/f'{name}_features.csv')
    print('INPUT CHECK PASSED: version 3 partitions, features, augmentation and original corpus.',flush=True)
    if args.check_only:
        print('No reports created and no training started.');return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/'experiment_preflight_v3_reports'/stamp
    output.mkdir(parents=True,exist_ok=False)
    details=[];summary=[];proposed_exclusions={name:set() for name in parts}
    for setting,length,strip in SETTINGS:
        print(f'\nAuditing {setting} against all original source labels...',flush=True)
        records={};masks={}
        for part,rows in parts.items():
            for index,row in enumerate(rows):
                encoded=code(row['url'],length,strip)
                records.setdefault(encoded,[]).append((part,index,row))
                masks[encoded]=0
        total_source=0
        with source.open(encoding='utf-8-sig',newline='') as file:
            for row in csv.DictReader(file):
                category=row['type'].strip().lower()
                if category in MALICIOUS:label=1
                elif category in BENIGN:label=0
                else:raise ValueError(f'Unrecognized source category: {category!r}')
                encoded=code(row['url'],length,strip)
                if encoded in masks:masks[encoded]|=1<<label
                total_source+=1
        if total_source!=651191:raise ValueError('Unexpected source row count')
        for row in augmentation:
            encoded=code(row['url'],length,strip)
            if encoded in masks:masks[encoded]|=1
        totals={part:Counter() for part in parts}
        for encoded,items in records.items():
            across=len({part for part,_,_ in items})>1
            ambiguous=masks[encoded]==3
            for part,index,row in items:
                expected_mask=1<<int(row['label'])
                mismatch=masks[encoded]!=3 and masks[encoded]!=expected_mask
                if across:totals[part]['cross_partition_encoding']+=1
                if ambiguous:totals[part]['opposite_source_labels']+=1
                if mismatch:totals[part]['source_label_mismatch_or_missing']+=1
                if across or ambiguous or mismatch:
                    proposed_exclusions[part].add(index)
                    details.append({'setting':setting,'partition':part,'row_index':index,'url':row['url'],'label':row['label'],
                        'opposite_source_labels':ambiguous,'cross_partition_encoding':across,'source_label_mismatch_or_missing':mismatch,
                        'encoded_sha256':encoded.hex()})
        for part in parts:
            entry={'setting':setting,'partition':part,'rows':len(parts[part]),**{k:totals[part][k] for k in ['opposite_source_labels','cross_partition_encoding','source_label_mismatch_or_missing']}}
            summary.append(entry);print(entry,flush=True)
    common=[]
    for part,rows in parts.items():
        indices=proposed_exclusions[part]
        generated_before=sum(canonical(r['url']) in generated for r in rows)
        generated_after=sum(canonical(r['url']) in generated for i,r in enumerate(rows) if i not in indices)
        common.append({'partition':part,'original_rows':len(rows),'proposed_exclusions':len(indices),
            'proposed_retained_rows':len(rows)-len(indices),'generated_before':generated_before,'generated_after':generated_after,
            'proposed_non_generated_training_rows':sum(canonical(r['url']) not in generated for i,r in enumerate(rows) if i not in indices) if part=='train' else None})
    save_csv(output/'encoding_audit.csv',summary,list(summary[0]))
    detail_fields=['setting','partition','row_index','url','label','opposite_source_labels','cross_partition_encoding','source_label_mismatch_or_missing','encoded_sha256']
    save_csv(output/'affected_records.csv',details,detail_fields)
    save_csv(output/'proposed_common_subset_counts.csv',common,list(common[0]))
    report={'source_sha256':SOURCE_HASH,'augmentation_sha256':sha(augmentation_path),'partition_sha256':HASHES,'feature_sha256':feature_hashes,
        'settings':[{'name':n,'max_length':l,'strip_scheme':s} for n,l,s in SETTINGS],
        'summary':summary,'proposed_common_subset':common,
        'policy':'Report only. No exclusions applied. Proposed common subset conservatively removes every affected record from every partition across all four encodings. Existing version 3 models were trained on different rows and cannot serve as exact matched controls if this subset is adopted.',
        'caution':'Source labels are used across the corpus solely for predeclared ambiguity screening, not model selection. Any resulting dataset must be disclosed. Shorter/longer encodings can create or resolve representation ambiguity. No claim that nine runs suffice if a new matched reference is needed. Labels and source dates may have other unresolved errors.'}
    (output/'preflight_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nPREFLIGHT COMPLETE. Dataset files unchanged; no training started.')
    for entry in common:print(entry)
    print('Saved reports to:',output)
if __name__=='__main__':main()
