"""Fixed-partition baseline tuning, feature ablations and seeded evaluation.
Run --check-only, then --tune, then --seed 42 through --seed 46.
No main-model retraining. Test results never select baseline settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score)
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset_fix_outputs_v3'
OUTPUT = ROOT / 'baseline_results_v3'
SIZES = {'train': 35664, 'validation': 7668, 'test': 7653}
HASHES = {
    'train': '28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
    'validation': '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
    'test': '22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f',
}
VARIANTS = {
    'full': set(),
    'no_scheme': {'is_https'},
    'no_scheme_no_ghana': {'is_https', 'is_trusted_gh_tld', 'brand_injected', 'brand_on_free_tld'},
}
GRID = {
    'RF': [{'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1},
           {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 2},
           {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 4}],
    'XGB': [{'n_estimators': 200, 'max_depth': 3, 'learning_rate': .1},
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': .1},
            {'n_estimators': 300, 'max_depth': 3, 'learning_rate': .05}],
    'CNN': [{'filters': 64}, {'filters': 128}],
    'BiLSTM': [{'units': 64}, {'units': 128}],
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def encode(url):
    text = str(url).strip()
    for prefix in ('https://', 'http://'):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    tokens = [ord(c) - 31 if 32 <= ord(c) <= 126 else 96 for c in text.lower().strip()[:200]]
    return tokens + [0] * (200 - len(tokens))

def preflight():
    import importlib.util
    missing = [name for name in ['tensorflow', 'keras', 'xgboost', 'joblib'] if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError('Missing packages: ' + ', '.join(missing))
    frames, numeric, encoded = {}, {}, {}
    feature_names = None
    for name, size in SIZES.items():
        path = DATA / f'{name}.csv'
        if digest(path) != HASHES[name]:
            raise ValueError(f'Partition fingerprint mismatch: {name}')
        frame = pd.read_csv(path, keep_default_na=False)
        feature_path = DATA / f'{name}_features.csv'
        feat = pd.read_csv(feature_path, keep_default_na=False)
        if len(frame) != size or not frame[['url', 'label']].equals(feat[['url', 'label']]):
            raise ValueError(f'Feature alignment/size failure: {name}')
        if not frame.label.isin([0, 1]).all():
            raise ValueError(f'Invalid labels: {name}')
        columns = [c for c in feat if c not in {'url', 'label'}]
        if len(columns) != 23 or (feature_names is not None and columns != feature_names):
            raise ValueError('Expected same 23 features in every partition')
        feature_names = columns
        if not set.union(*VARIANTS.values()).issubset(columns):
            raise ValueError('Missing ablation columns')
        values = feat[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'Invalid numeric features: {name}')
        frames[name], numeric[name] = frame, values
        encoded[name] = np.array([encode(u) for u in frame.url], dtype=np.int32)
        print(f'{name}: {size:,} rows; URL/label/features aligned')
    for seed in range(42, 47):
        run = ROOT / 'corrected_results_v3' / f'seed_{seed}'
        m = json.loads((run / 'metrics.json').read_text())
        if m['seed'] != seed or m['dataset_version'] != 'v3':
            raise ValueError(f'Main-model run identity mismatch: {seed}')
        for name in SIZES:
            if m['partitions'][name]['sha256'] != HASHES[name]:
                raise ValueError(f'Main-model fingerprint mismatch: {seed}/{name}')
        for name in ['validation', 'test']:
            saved = pd.read_csv(run / f'{name}_predictions.csv', keep_default_na=False)
            if not saved[['url', 'true_label']].rename(columns={'true_label': 'label'}).equals(frames[name][['url', 'label']]):
                raise ValueError(f'Main prediction alignment mismatch: {seed}/{name}')
            if not np.isfinite(saved.probability).all() or not saved.probability.between(0, 1).all():
                raise ValueError('Invalid saved main-model probabilities')
    print('CHECK PASSED: final partitions, features and all five main-model runs.')
    return frames, numeric, encoded, feature_names

def tasks():
    return [(f'{kind}_{variant}', kind, variant) for kind in ['RF', 'XGB'] for variant in VARIANTS] + [('CNN', 'CNN', None), ('BiLSTM', 'BiLSTM', None)]

def get_x(kind, variant, numeric, encoded, columns, name):
    if kind in ['CNN', 'BiLSTM']:
        return encoded[name]
    indices = [i for i, c in enumerate(columns) if c not in VARIANTS[variant]]
    return numeric[name][:, indices]

def fit_model(kind, config, seed, x, y, vx, vy):
    if kind == 'RF':
        model = RandomForestClassifier(**config, random_state=seed, n_jobs=-1)
        model.fit(x, y)
        return model, None
    if kind == 'XGB':
        from xgboost import XGBClassifier
        model = XGBClassifier(**config, random_state=seed, n_jobs=-1, eval_metric='logloss', subsample=1.0, colsample_bytree=1.0)
        model.fit(x, y)
        return model, None
    import keras
    from keras import layers, callbacks
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(200,), dtype='int32')
    h = layers.Embedding(97, 64)(inputs)
    if kind == 'CNN':
        h = layers.Conv1D(config['filters'], 3, activation='relu', padding='same')(h)
        h = layers.GlobalMaxPooling1D()(h)
    else:
        h = layers.Bidirectional(layers.LSTM(config['units']))(h)
        h = layers.Dropout(.4)(h)
    h = layers.Dense(64, activation='relu')(h)
    output = layers.Dense(1, activation='sigmoid')(h)
    model = keras.Model(inputs, output)
    model.compile(optimizer=keras.optimizers.Adam(.001), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x, y, validation_data=(vx, vy), epochs=50, batch_size=256,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
                   callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=.5, min_lr=1e-5, verbose=1)], verbose=1)
    return model, history.history

def probabilities(model, kind, x):
    return model.predict(x, verbose=0).reshape(-1) if kind in ['CNN', 'BiLSTM'] else model.predict_proba(x)[:, list(model.classes_).index(1)]

def save_model(model, kind, folder):
    if kind in ['CNN', 'BiLSTM']:
        model.save(folder / 'model.keras')
    else:
        joblib.dump(model, folder / 'model.pkl')

def load_model(kind, folder):
    if kind in ['CNN', 'BiLSTM']:
        import keras
        return keras.models.load_model(folder / 'model.keras', compile=False)
    return joblib.load(folder / 'model.pkl')

def scores(y, p):
    pred = p >= .5
    return {k: float(v) for k, v in dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred), f1_score=f1_score(y, pred), auc_roc=roc_auc_score(y, p), average_precision=average_precision_score(y, p)).items()}

def write_predictions(folder, name, frame, p):
    pd.DataFrame({'url': frame.url, 'true_label': frame.label, 'probability': p, 'pred_label': (p >= .5).astype(int)}).to_csv(folder / f'{name}_predictions.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--tune', action='store_true')
    mode.add_argument('--seed', type=int, choices=range(42, 47))
    args = parser.parse_args()
    frames, numeric, encoded, columns = preflight()
    if args.check_only:
        print('No models trained and no output directories created.')
        return
    protocol = {'dataset_version': 'v3', 'partition_hashes': HASHES,
        'feature_file_hashes': {n: digest(DATA / f'{n}_features.csv') for n in SIZES},
        'script_sha256': digest(__file__), 'selection': 'maximum validation average precision; first candidate wins ties',
        'tuning_seed': 42, 'grids': GRID, 'feature_variants': {k: sorted(v) for k, v in VARIANTS.items()},
        'neural_budget': {'epochs': 50, 'batch_size': 256, 'early_stopping_patience': 12, 'monitor': 'val_loss'},
        'note': 'Ghana ablation removes explicit brand and trusted Ghana suffix flags. It is not a guarantee of removing all geographic information. Main model settings are fixed, not newly tuned here.'}
    OUTPUT.mkdir(exist_ok=True)
    lock = OUTPUT / 'protocol.json'
    if lock.exists():
        if json.loads(lock.read_text()) != protocol:
            raise ValueError('Protocol differs from existing experiment. Preserve outputs; do not edit script or data mid-experiment.')
    else:
        save_json(lock, protocol)
    selected_path = OUTPUT / 'selected_configs.json'
    if args.tune:
        if selected_path.exists():
            print('Tuning already complete; selected configurations preserved. Next: --seed 42')
            return
        selected = {}
        for task, kind, variant in tasks():
            summaries = []
            for index, config in enumerate(GRID[kind]):
                folder = OUTPUT / 'tuning' / task / f'candidate_{index}'
                folder.mkdir(parents=True, exist_ok=True)
                checkpoint = folder / 'validation_metrics.json'
                if checkpoint.exists():
                    summary = json.loads(checkpoint.read_text())
                    print(f'Reusing completed tuning candidate: {task}/{index}')
                else:
                    print(f'\nTUNING {task}, candidate {index + 1}/{len(GRID[kind])}: {config}')
                    model, history = fit_model(kind, config, 42,
                        get_x(kind, variant, numeric, encoded, columns, 'train'), frames['train'].label.to_numpy(),
                        get_x(kind, variant, numeric, encoded, columns, 'validation'), frames['validation'].label.to_numpy())
                    p = probabilities(model, kind, get_x(kind, variant, numeric, encoded, columns, 'validation'))
                    save_model(model, kind, folder)
                    write_predictions(folder, 'validation', frames['validation'], p)
                    if history is not None:
                        save_json(folder / 'history.json', history)
                    summary = {'candidate': index, 'config': config, 'validation_average_precision': float(average_precision_score(frames['validation'].label, p))}
                    save_json(checkpoint, summary)
                    del model
                summaries.append(summary)
            winner = max(summaries, key=lambda r: r['validation_average_precision'])
            selected[task] = dict(winner, kind=kind, variant=variant)
            print(f'SELECTED {task}: {winner}')
        save_json(selected_path, selected)
        print('\nTUNING COMPLETE. No test performance used for configuration selection.')
        print('Next: python baseline_comparison_v3.py --seed 42')
        return
    if not selected_path.exists():
        raise FileNotFoundError('Run --tune first; no configurations selected yet.')
    selected = json.loads(selected_path.read_text())
    seed = args.seed
    results = {}
    for task, kind, variant in tasks():
        folder = OUTPUT / f'seed_{seed}' / task
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / 'metrics.json'
        if checkpoint.exists():
            results[task] = json.loads(checkpoint.read_text())['test_metrics']
            print(f'Reusing completed model: {task}, seed {seed}')
            continue
        winner = selected[task]
        if seed == 42:
            candidate_folder = OUTPUT / 'tuning' / task / f"candidate_{winner['candidate']}"
            model = load_model(kind, candidate_folder)
            print(f'Reusing selected seed 42 tuning model: {task}; no retraining.')
        else:
            print(f'\nTRAINING {task}, seed {seed}')
            model, history = fit_model(kind, winner['config'], seed,
                get_x(kind, variant, numeric, encoded, columns, 'train'), frames['train'].label.to_numpy(),
                get_x(kind, variant, numeric, encoded, columns, 'validation'), frames['validation'].label.to_numpy())
            if history is not None:
                save_json(folder / 'history.json', history)
        save_model(model, kind, folder)
        for name in ['validation', 'test']:
            p = probabilities(model, kind, get_x(kind, variant, numeric, encoded, columns, name))
            write_predictions(folder, name, frames[name], p)
            if name == 'test':
                result = scores(frames[name].label.to_numpy(), p)
        save_json(checkpoint, {'seed': seed, 'task': task, 'dataset_version': 'v3', 'partition_hashes': HASHES,
            'selected_config': winner, 'threshold': .5, 'test_metrics': result,
            'input_columns': [c for c in columns if c not in VARIANTS[variant]] if variant else 'scheme-stripped 200-character sequence'})
        results[task] = result
        print(task, result)
        del model
    # Reuse existing main-model predictions: no main-model inference or training.
    main_folder = ROOT / 'corrected_results_v3' / f'seed_{seed}'
    d = pd.read_csv(main_folder / 'test_predictions.csv')
    results['PhishGuard_GH'] = scores(d.true_label.to_numpy(), d.probability.to_numpy())
    table = pd.DataFrame.from_dict(results, orient='index')
    table.index.name = 'model'
    table.to_csv(OUTPUT / f'seed_{seed}' / 'comparison.csv')
    print('\nFINAL TEST COMPARISON (proportions, threshold 0.5)')
    print(table.to_string(float_format=lambda x: f'{x:.4f}'))
    print('Saved to:', OUTPUT / f'seed_{seed}')

if __name__ == '__main__':
    main()
