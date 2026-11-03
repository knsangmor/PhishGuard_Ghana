"""Build a filtered PRIVATE review candidate from the verified historical ZIP.
Standard library only. No extraction, training, inference, network or publication.
"""
import argparse
import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_ZIP = '39e873d3fb40867b0812c34a4d3891f394701668304c60f8616434248a763417'
EXPECTED_MANIFEST = '88713361831f846ed1f0ee9d3a20b593775ddeda7712bbd4150e6661ccf06a5f'
EXPECTED_COUNT = 1184
EXPECTED_BYTES = 4793226851
GENERATED = {'RELEASE_MANIFEST.json', 'RELEASE_SCOPE.json', 'VERIFY_RELEASE.py', 'README_RELEASE.md', 'RELEASE_README.md'}
BINARY = {'.pt', '.pth', '.bin', '.safetensors', '.ckpt', '.onnx'}
ASSETS = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'vocab.json', 'merges.txt'}
SCRIPTS = {'published_baseline_preflight.py', 'published_cpu_comparison_v3.py'}


def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def safe_name(name):
    if not name or '\\' in name or name.startswith('/') or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    if any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe archive path: ' + name)
    return name


def exclusion(name):
    p = Path(name)
    low = name.lower()
    if p.suffix.lower() in BINARY:
        return 'Checkpoint/weight format withheld pending third-party permission review'
    if p.suffix.lower() == '.ipynb':
        return 'Historical notebook withheld pending adapted-code and output review'
    if any(x in low.split('/') for x in ('encoder_assets', 'published_encoder')) or p.name.lower() in ASSETS:
        return 'Copied upstream encoder/tokenizer asset withheld'
    if p.name in SCRIPTS:
        return 'Published-model integration code withheld pending upstream-code review'
    if p.name.lower().startswith('top-1m') or p.name.lower() in ('tranco.csv', 'tranco.zip'):
        return 'Historical ranking source withheld pending source-specific terms review'
    if p.name == 'DomURLs_REPRODUCTION.md':
        return 'Historical instructions assume withheld components; replaced by release scope'
    if p.name in ('candidate_1_manifest.json', 'candidate_1_ZIP_SHA256.txt'):
        return 'Private candidate inventory omitted to avoid confusing release scopes'
    return None


def inspect(archive, manifest):
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST:
        raise ValueError('Manifest is not the inspected 29 October candidate; stop for review')
    m = json.loads(raw)
    entries = m['files']
    expected = {}
    for row in entries:
        name = safe_name(row['path'])
        if name in expected:
            raise ValueError('Duplicate manifest path')
        expected[name] = row
    if len(entries) != EXPECTED_COUNT or sum(x['size_bytes'] for x in entries) != EXPECTED_BYTES:
        raise ValueError('Unexpected candidate inventory')
    print('Checking source ZIP SHA256 (this may take a few minutes)...', flush=True)
    if digest(archive) != EXPECTED_ZIP:
        raise ValueError('Source ZIP checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [x.filename for x in infos]
        if len(names) != len(set(names)) or len(names) != len({s.casefold() for s in names}):
            raise ValueError('Duplicate or case-colliding archive paths')
        for info in infos:
            if info.is_dir():
                continue
            safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Archive symlink rejected')
        files = {x.filename: x for x in infos if not x.is_dir()}
        if not set(expected).issubset(files):
            raise ValueError('Manifest files absent from source ZIP')
        extras = set(files) - set(expected)
        if extras - GENERATED:
            raise ValueError('Unexpected unmanifested source files: ' + str(sorted(extras)))
        included, withheld = [], []
        for name, row in expected.items():
            if files[name].file_size != row['size_bytes']:
                raise ValueError('Size mismatch: ' + name)
            reason = exclusion(name)
            if reason:
                withheld.append({**row, 'reason': reason})
            else:
                included.append(row)
    return included, withheld


VERIFIER = '''"""Verify this filtered review archive. Does not execute research code."""
import hashlib, json, sys, zipfile
from pathlib import Path
p = Path(sys.argv[1])
with zipfile.ZipFile(p) as z:
    names = z.namelist()
    assert len(names) == len(set(names)), 'Duplicate ZIP entries'
    m = json.loads(z.read('REVIEW_MANIFEST.json'))
    assert set(names) == {r['path'] for r in m['files']} | {'REVIEW_MANIFEST.json'}, 'Membership mismatch'
    for row in m['files']:
        h = hashlib.sha256()
        with z.open(row['path']) as stream:
            for b in iter(lambda: stream.read(1048576), b''):
                h.update(b)
        assert h.hexdigest() == row['sha256'], row['path']
print('Filtered review archive hashes PASSED. This is not licence clearance or scientific reproduction.')
'''

README = '''# Filtered research review candidate — NOT PUBLISHED

This package is a private review draft, not an approved public release. No DOI or release tag has been issued by this builder. Do not upload it publicly until the outstanding rights and manuscript-to-assets reviews are complete.

`research/` retains included historical files at their original relative paths. Historical READMEs and reports describe their original experiment scope; they do not override this notice. Run research scripts from that directory only after reviewing their requirements. Some now require withheld assets and cannot reproduce the complete study from this package alone.

DomURLs full fine-tuned/base weights, resume checkpoints, tokenizer/config assets, historical notebooks and identified published-model integration scripts are withheld pending permission/adapted-code review. Results, protocols and fingerprints are retained where selected. Full historical artifacts remain in the researcher's unchanged private candidate. This package does not include every trained model and must not be described that way.

The historical Tranco source terms, author-code licence, adapted upstream code, other dependency obligations, manuscript cross-check and public accessibility remain review items. File selection is not a full scan for embedded third-party material, secrets or private records. Source-specific licences are described in THIRD_PARTY_NOTICES.md; no blanket licence is granted by this builder.

`WITHHELD_FILES.json` lists omissions and reasons. `REVIEW_MANIFEST.json` lists actual retained/generated bytes and hashes. `SOURCE_PROVENANCE.json` identifies the input archive and manifest. Source experiment metric checks are not rerun; hashing establishes consistency, not scientific or legal validity.

To check archive consistency, save VERIFY_REVIEW.py separately and run:

    python VERIFY_REVIEW.py PhishGuard_Filtered_Review_Candidate.zip

Keep excluded artifacts privately; do not delete or rewrite historical records to suppress warnings.
'''


def build(archive, manifest, notice, parent):
    included, withheld = inspect(archive, manifest)
    if not notice.is_file():
        raise ValueError('THIRD_PARTY_NOTICES.md is required')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = parent / ('public_review_candidate_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    target = out / 'PhishGuard_Filtered_Review_Candidate.zip'
    temp = target.with_suffix('.partial')
    records = []
    try:
        with zipfile.ZipFile(archive) as src, zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as dst:
            for index, row in enumerate(included, 1):
                name = 'research/' + row['path']
                h = hashlib.sha256()
                with src.open(row['path']) as inp, dst.open(name, 'w', force_zip64=True) as output:
                    for b in iter(lambda: inp.read(1024 * 1024), b''):
                        h.update(b)
                        output.write(b)
                if h.hexdigest() != row['sha256']:
                    raise ValueError('Source content mismatch: ' + row['path'])
                records.append({'path': name, 'size_bytes': row['size_bytes'], 'sha256': h.hexdigest()})
                if index % 100 == 0:
                    print(f'Copied and verified {index}/{len(included)} files', flush=True)
            generated = {
                'README_RELEASE.md': README.encode(),
                'VERIFY_REVIEW.py': VERIFIER.encode(),
                'THIRD_PARTY_NOTICES.md': notice.read_bytes(),
                'WITHHELD_FILES.json': json.dumps(withheld, indent=2).encode(),
                'SOURCE_PROVENANCE.json': json.dumps({
                    'source_zip_sha256': EXPECTED_ZIP,
                    'source_manifest_sha256': EXPECTED_MANIFEST,
                    'builder_sha256': digest(Path(__file__)),
                    'created_utc': stamp,
                    'public_release': False,
                    'publication_ready': False,
                    'licence_clearance': False,
                    'fresh_inference_or_training': False,
                    'included_source_files': len(included),
                    'withheld_source_files': len(withheld),
                    'source_files_unchanged': True
                }, indent=2).encode()
            }
            for name, raw in generated.items():
                dst.writestr(name, raw)
                records.append({'path': name, 'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            draft = {'public_release': False, 'publication_ready': False, 'files': records}
            dst.writestr('REVIEW_MANIFEST.json', json.dumps(draft, indent=2))
        with zipfile.ZipFile(temp) as z:
            if set(z.namelist()) != {r['path'] for r in records} | {'REVIEW_MANIFEST.json'}:
                raise ValueError('Output membership mismatch')
            for row in records:
                with z.open(row['path']) as stream:
                    if digest_stream(stream) != row['sha256']:
                        raise ValueError('Output verification failed: ' + row['path'])
        temp.rename(target)
        (out / 'REVIEW_MANIFEST.json').write_text(json.dumps(draft, indent=2), encoding='utf-8')
        for name, raw in generated.items():
            (out / name).write_bytes(raw)
        (out / 'ZIP_SHA256.txt').write_text(digest(target) + '  ' + target.name + '\n', encoding='utf-8')
    except Exception:
        print('BUILD FAILED. Any partial output must not be released.', flush=True)
        raise
    print('FILTERED REVIEW ARCHIVE VERIFIED:', target)
    print('Not published. Rights review remains pending. Original ZIP unchanged.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', required=True, type=Path, help='Existing 29 October 13:03 candidate directory')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--build', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    source = args.source_dir.resolve()
    archive = source / 'PhishGuard_Revision_Release_Candidate.zip'
    manifest = source / 'RELEASE_MANIFEST.json'
    notice = root / 'THIRD_PARTY_NOTICES.md'
    if args.check_only:
        included, withheld = inspect(archive, manifest)
        print(json.dumps({'included_files': len(included), 'withheld_files': len(withheld),
            'withheld_reasons': dict(Counter(r['reason'] for r in withheld)),
            'notice_present': notice.is_file(), 'publication_ready': False,
            'scope': 'Pinned source archive/hash and selection check; source payloads are checked during build'}, indent=2))
        print('READ-ONLY CHECK COMPLETE. No outputs, training or publication.')
    else:
        build(archive, manifest, notice, root)


if __name__ == '__main__':
    main()
