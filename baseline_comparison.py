
# PhishGuard-GH Baseline Model Comparison 



import os, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import layers, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, joblib



print("  PhishGuard-GH Baseline Model Comparison — unified split & encoding")

MAX_LEN = 200; VOCAB_SIZE = 97
CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}


def strip_scheme(url: str) -> str:
    u = str(url).strip()
    for prefix in ("https://", "http://"):
        if u.lower().startswith(prefix):
            return u[len(prefix):]
    return u


def url_to_seq(url):
    # scheme stripped, exactly matching model_hash.py's encoding for PhishGuardGH
    url = strip_scheme(url).lower().strip()
    seq = [CHAR_TO_INT.get(c, VOCAB_SIZE - 1) for c in url[:MAX_LEN]]
    return seq + [0] * (MAX_LEN - len(seq))


def evaluate(yt, yp, ypr):
    return {"accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "auc": roc_auc_score(yt, ypr)}


from pathlib import Path
import json

SEED = 44
keras.utils.set_random_seed(SEED)

project_folder = Path(__file__).resolve().parent
split_folder = project_folder / "dataset_fix_outputs"
main_run_folder = project_folder / "corrected_results" / f"seed_{SEED}"

print("\nLoading corrected partitions and features...")

parts = {}
features = {}
feature_columns = None

expected_sizes = {
    "train": 36144,
    "validation": 7737,
    "test": 7736,
}

for name, expected in expected_sizes.items():
    part = pd.read_csv(
        split_folder / f"{name}.csv",
        keep_default_na=False,
    )
    feat = pd.read_csv(
        split_folder / f"{name}_features.csv",
        keep_default_na=False,
    )

    if len(part) != expected or len(feat) != expected:
        raise ValueError(f"Unexpected row count in {name}")

    if not part["label"].isin([0, 1]).all():
        raise ValueError(f"Invalid labels in {name}")

    if part["url"].tolist() != feat["url"].tolist():
        raise ValueError(f"URL alignment failed in {name}")

    if part["label"].tolist() != feat["label"].tolist():
        raise ValueError(f"Label alignment failed in {name}")

    columns = [
        column for column in feat.columns
        if column not in {"url", "label"}
    ]

    if len(columns) != 23:
        raise ValueError(f"Expected 23 features in {name}")

    if feature_columns is None:
        feature_columns = columns
    elif columns != feature_columns:
        raise ValueError(f"Feature columns differ in {name}")

    numeric = feat[columns].apply(pd.to_numeric, errors="raise")
    matrix = numeric.to_numpy(dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError(f"Missing or infinite features in {name}")

    parts[name] = part
    features[name] = matrix

Xf_tr = features["train"]
Xf_va = features["validation"]
Xf_te = features["test"]

yf_tr = parts["train"]["label"].to_numpy(dtype=np.int32)
yf_va = parts["validation"]["label"].to_numpy(dtype=np.int32)
yf_te = parts["test"]["label"].to_numpy(dtype=np.int32)

def encode_partition(name):
    return np.array(
        [url_to_seq(url) for url in parts[name]["url"]],
        dtype=np.int32,
    )

Xs_tr = encode_partition("train")
Xs_va = encode_partition("validation")
Xs_te = encode_partition("test")

ys_tr, ys_va, ys_te = yf_tr, yf_va, yf_te

# Check encoding and test labels against the saved main-model run.
if not np.array_equal(
    Xs_te, np.load(main_run_folder / "X_test.npy")
):
    raise ValueError("Test encoding differs from the main-model run")

if not np.array_equal(
    ys_te, np.load(main_run_folder / "y_test.npy")
):
    raise ValueError("Test labels differ from the main-model run")

print(
    f"Train: {len(Xs_tr):,}  "
    f"Validation: {len(Xs_va):,}  "
    f"Test: {len(Xs_te):,}"
)
print("Feature alignment and main-model test encoding checks passed.")

results_folder = (
    project_folder / "corrected_baseline_results" / f"seed_{SEED}"
)
results_folder.mkdir(parents=True, exist_ok=False)
os.chdir(results_folder)
print("Results will save to:", results_folder)

RESULTS = {}; ROC = {}
def store(name, yt, yp, ypr):
    RESULTS[name] = evaluate(yt, yp, ypr)
    fpr, tpr, _ = roc_curve(yt, ypr); ROC[name] = (fpr, tpr)
    r = RESULTS[name]
    print(f"  {name:<25} Acc={r['accuracy']*100:.2f}%  F1={r['f1']*100:.2f}%  AUC={r['auc']:.4f}")


print("\nTraining Random Forest (shared split)...")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
rf.fit(Xf_tr, yf_tr); joblib.dump(rf, "model_rf.pkl")
store("Random Forest", yf_te, rf.predict(Xf_te), rf.predict_proba(Xf_te)[:, 1])

print("Training XGBoost (shared split)...")
xgb = XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED)
xgb.fit(Xf_tr, yf_tr, verbose=False); joblib.dump(xgb, "model_xgb.pkl")
store("XGBoost", yf_te, xgb.predict(Xf_te), xgb.predict_proba(Xf_te)[:, 1])


print("Training Standalone CNN (saved partitions)...")

keras.utils.set_random_seed(SEED)
cnn_es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1,
)

i1 = keras.Input(shape=(MAX_LEN,))
x1 = layers.Embedding(VOCAB_SIZE, 64)(i1)
x1 = layers.Conv1D(
    128, 3, activation="relu", padding="same"
)(x1)
x1 = layers.GlobalMaxPooling1D()(x1)
x1 = layers.Dense(64, activation="relu")(x1)
o1 = layers.Dense(1, activation="sigmoid")(x1)
cnn = keras.Model(i1, o1)

cnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
cnn.fit(
    Xs_tr,
    ys_tr,
    validation_data=(Xs_va, ys_va),
    epochs=30,
    batch_size=256,
    callbacks=[cnn_es],
    verbose=1,
)
cnn.save("model_cnn.keras")

pc = cnn.predict(Xs_te, verbose=0).flatten()
store("Standalone CNN", ys_te, (pc >= 0.5).astype(int), pc)


print("Training Standalone BiLSTM (saved partitions)...")

keras.utils.set_random_seed(SEED)
bilstm_es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1,
)

i2 = keras.Input(shape=(MAX_LEN,))
x2 = layers.Embedding(VOCAB_SIZE, 64)(i2)
x2 = layers.Bidirectional(layers.LSTM(128))(x2)
x2 = layers.Dropout(0.4)(x2)
x2 = layers.Dense(64, activation="relu")(x2)
o2 = layers.Dense(1, activation="sigmoid")(x2)
bl = keras.Model(i2, o2)

bl.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
bl.fit(
    Xs_tr,
    ys_tr,
    validation_data=(Xs_va, ys_va),
    epochs=30,
    batch_size=256,
    callbacks=[bilstm_es],
    verbose=1,
)
bl.save("model_bilstm.keras")

pb2 = bl.predict(Xs_te, verbose=0).flatten()
store("Standalone BiLSTM", ys_te, (pb2 >= 0.5).astype(int), pb2)

print("Loading PhishGuard-GH (already trained on this identical split by model_hash.py)...")
mm = keras.models.load_model(
    main_run_folder / "phishguard_gh_model.keras",
    compile=False,
)
ppg = mm.predict(Xs_te, verbose=0).flatten()  # NOTE: evaluated on Xs_te computed here, same te_idx as model_hash.py used
store("PhishGuard-GH", ys_te, (ppg >= 0.5).astype(int), ppg)

print(f"\n  {'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
for name, r in RESULTS.items():
    star = " \u2605" if name == "PhishGuard-GH" else ""
    print(f"  {name+star:<24} {r['accuracy']*100:>6.2f}% {r['precision']*100:>6.2f}% "
          f"{r['recall']*100:>6.2f}% {r['f1']*100:>6.2f}% {r['auc']:>6.4f}")

with open("baseline_metrics.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "seed": SEED,
            "test_size": len(ys_te),
            "threshold": 0.5,
            "results": RESULTS,
        },
        file,
        indent=2,
    )


# Charts (same as original script)
names = list(RESULTS.keys()); accs = [RESULTS[n]["accuracy"] * 100 for n in names]
f1s = [RESULTS[n]["f1"] * 100 for n in names]
colors = ["steelblue", "steelblue", "steelblue", "steelblue", "#e74c3c"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5)); x = np.arange(len(names))
for ax, vals, lbl in [(axes[0], accs, "Accuracy (%)"), (axes[1], f1s, "F1-Score (%)")]:
    ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(lbl); ax.set_ylim(0, 100)
    ax.set_title(f"{lbl} Comparison (Unified Split & Encoding)\nPhishGuard-GH", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight"); plt.close()

lc = ["#95a5a6", "#7f8c8d", "#3498db", "#2ecc71", "#e74c3c"]; lw = [1.5, 1.5, 1.5, 1.5, 3.0]
plt.figure(figsize=(8, 6))
for i, (name, (fpr, tpr)) in enumerate(ROC.items()):
    plt.plot(fpr, tpr, color=lc[i], lw=lw[i], label=f"{name} (AUC={RESULTS[name]['auc']:.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Comparison (Unified Split & Encoding)\nPhishGuard-GH", fontsize=12, fontweight="bold")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=300, bbox_inches="tight"); plt.close()

print("\n  Saved: model_comparison.png  roc_comparison.png")
