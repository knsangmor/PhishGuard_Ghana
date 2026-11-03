"""Bounded offline score-guided evasion probes; inherited labels are assumptions.
No training or network requests. Does NOT validate live page functionality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit, unquote

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
HASHES = {
    'train': '28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
    'validation': '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
    'test': '22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f',
}
LIMITATION = ('Synthetic offline score-guided probes, with inherited malicious labels. '
              'Fragments may affect client-side routing; percent-encoding may affect server handling. '
              'No live malicious functionality or label preservation was validated. '
              'This is not a complete adaptive-attacker evaluation or external test.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def bare(url):
    u = str(url).strip()
    for prefix in ('https://', 'http://'):
        if u.lower().startswith(prefix):
            return u[len(prefix):]
    return u


def encode(url):
    chars = bare(url).lower().strip()[:200]
    seq = [ord(c) - 31 if 32 <= ord(c) <= 126 else 96 for c in chars]
    return seq + [0] * (200 - len(seq))


def request_parts(url):
    # Do not silently repair quotes, credentials, whitespace or malformed ports.
    u = str(url).strip()
    if not u or u[0] in "'\"" or any(c.isspace() for c in u):
        return None
    try:
        p = urlsplit(u if u.lower().startswith(('http://', 'https://')) else 'http://' + u)
        if p.scheme.lower() not in ('http', 'https') or not p.hostname or p.username is not None:
            return None
        _ = p.port
        return p
    except ValueError:
        return None


def candidates(url):
    p = request_parts(url)
    if p is None:
        return []
    u = str(url).strip()
    result = []
    seen = {tuple(encode(u))}
    suffixes = ('about', 'home', 'news', 'products', 'support', 'documentation')
    for suffix in suffixes:
        v = u + ('-' if '#' in u else '#') + suffix
        q = request_parts(v)
        assert q is not None and q[:4] == p[:4]
        key = tuple(encode(v))
        if key not in seen:
            seen.add(key)
            result.append(('fragment_' + suffix, v))
    # Encode one/two literal ASCII letters in path only, preserving decoded path.
    # Existing percent triplets, host, query and fragment remain untouched.
    prefix_end = len(u.split('#', 1)[0].split('?', 1)[0]) - len(p.path)
    offsets = []
    i = 0
    while i < len(p.path):
        if p.path[i] == '%' and i + 2 < len(p.path):
            i += 3
            continue
        if p.path[i].isascii() and p.path[i].isalpha():
            offsets.append(i)
        i += 1
    for count in (1, 2):
        if len(offsets) < count:
            continue
        path = p.path
        for offset in reversed(offsets[:count]):
            path = path[:offset] + f'%{ord(path[offset]):02X}' + path[offset + 1:]
        v = u[:prefix_end] + path + u[prefix_end + len(p.path):]
        q = request_parts(v)
        assert q is not None and (q.scheme, q.netloc, q.query, q.fragment) == (p.scheme, p.netloc, p.query, p.fragment)
        assert unquote(q.path) == unquote(p.path)
        key = tuple(encode(v))
        if key not in seen:
            seen.add(key)
            result.append(('percent_path_' + str(count), v))
    assert len(result) <= 8
    return result


def self_test():
    for u in ('https://example.com/login?x=1', 'example.com/a%20b#route', 'http://example.com:80/abc'):
        assert 1 <= len(candidates(u)) <= 8
    assert candidates("'example.com/login'") == []
    assert encode('http://EXAMPLE.com/a') == encode('https://example.com/a')
    assert all(kind.startswith('percent_path_') for kind, _ in candidates('example.com/' + 'a' * 220))
    print('SELF-TEST PASSED: transformations, candidate budget and encoding.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, choices=range(42, 47), default=42)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    import numpy as np
    import pandas as pd
    root = Path(__file__).resolve().parent
    data = root / 'dataset_fix_outputs_v3'
    run = root / 'corrected_results_v3' / f'seed_{args.seed}'
    for name, expected in HASHES.items():
        if sha(data / f'{name}.csv') != expected:
            raise ValueError(f'Frozen partition mismatch: {name}')
    frame = pd.read_csv(data / 'test.csv', keep_default_na=False)
    saved = pd.read_csv(run / 'test_predictions.csv', keep_default_na=False)
    if not frame[['url', 'label']].equals(saved[['url', 'true_label']].rename(columns={'true_label': 'label'})):
        raise ValueError('Saved test URL/label alignment failed')
    x = np.asarray([encode(u) for u in frame.url], dtype=np.int32)
    if not np.array_equal(x, np.load(run / 'X_test.npy', allow_pickle=False)):
        raise ValueError('Encoding differs from main-model run')
    model_path = run / 'phishguard_gh_model.keras'
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    variants = [(i, kind, v) for i, row in frame.iterrows() if row.label == 1
                for kind, v in candidates(row.url)]
    eligible = {i for i, _, _ in variants}
    print(f'CHECK PASSED: seed {args.seed}, {len(frame):,} test rows; '
          f'{len(eligible):,} malicious URLs with changed encodings; '
          f'{len(variants):,} candidate queries (maximum 8 per URL).')
    print(LIMITATION)
    if args.check_only:
        print('No inference, training or output folders created.')
        return
    import keras
    model = keras.models.load_model(model_path, compile=False)
    print('Checking original test predictions...', flush=True)
    base = model.predict(x, batch_size=256, verbose=0).reshape(-1)
    if not np.allclose(base, saved.probability.to_numpy(), atol=1e-5, rtol=1e-5):
        raise ValueError('Saved model probabilities differ; no report written')
    best = base.copy()
    best_url = frame.url.tolist()
    best_kind = ['original'] * len(frame)
    counts = [0] * len(frame)
    records = []
    for start in range(0, len(variants), 1024):
        batch = variants[start:start + 1024]
        xp = np.asarray([encode(v) for _, _, v in batch], dtype=np.int32)
        probs = model.predict(xp, batch_size=256, verbose=0).reshape(-1)
        for (i, kind, v), prob in zip(batch, probs):
            prob = float(prob)
            counts[i] += 1
            records.append({'test_index': i, 'family': kind, 'url': v, 'probability': prob})
            if prob < best[i]:
                best[i], best_url[i], best_kind[i] = prob, v, kind
        print(f'Candidate queries: {min(start + 1024, len(variants)):,}/{len(variants):,}', flush=True)
    malicious = frame.label.to_numpy() == 1
    detected = malicious & (base >= .5)
    eligible_mask = np.asarray([i in eligible for i in range(len(frame))])
    attacked = detected & eligible_mask
    success = attacked & (best < .5)
    summary = {
        'seed': args.seed, 'threshold': .5, 'maximum_candidate_queries_per_url': 8,
        'selection': 'minimum probability, original retained if all candidates score higher; non-sequential score-guided search',
        'partition_sha256': HASHES, 'model_sha256': sha(model_path),
        'test_rows': len(frame), 'malicious_test_rows': int(malicious.sum()),
        'eligible_malicious': len(eligible), 'candidate_queries': len(variants),
        'originally_detected_malicious': int(detected.sum()),
        'originally_detected_and_eligible': int(attacked.sum()),
        'synthetic_evasions': int(success.sum()),
        'synthetic_evasion_rate_among_detected_eligible': float(success.sum() / attacked.sum()) if attacked.any() else None,
        'original_recall': float(detected.sum() / malicious.sum()),
        'worst_case_synthetic_recall_all_malicious': float((best[malicious] >= .5).mean()),
        'limitation': LIMITATION,
    }
    output = root / 'adaptive_evasion_v3' / (f'seed_{args.seed}_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    result = frame.copy()
    result['original_probability'] = base
    result['minimum_probability'] = best
    result['selected_variant'] = best_kind
    result['selected_url'] = best_url
    result['candidate_queries'] = counts
    result['synthetic_evasion'] = success
    result.to_csv(output / 'per_url_results.csv', index=False)
    pd.DataFrame(records).to_csv(output / 'candidate_predictions.csv', index=False)
    (output / 'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved to:', output)
    print('No training, network access or source-file changes performed.')


if __name__ == '__main__':
    main()
