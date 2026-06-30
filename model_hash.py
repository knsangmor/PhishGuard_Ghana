# PhishGuard-GH  |  train_hash.py

# STAGE SEQUENCE (runs top-to-bottom in one process, model loaded ONCE):
#
#   ① SHA256(phishguard_gh_dataset.csv)       → dataset_hash.txt
#   ② Train CNN-LSTM on X_tr / y_tr
#   ③ Evaluate on X_te / y_te                 → metrics.json
#   ④ Save model files → SHA256(model.h5)     → model_fingerprint.txt
#   ⑤ Full-dataset inference (in-memory, no reload)
#   ⑥ Save predictions.csv
#   ⑦ SHA256(predictions.csv)                 → predictions_hash.txt
#   ⑧ Save X_test.npy / y_test.npy /
#         X_train_background.npy / y_prob.npy
#
# Immediately after ⑧ the hash-chain pipeline begins :
#   ① Load predictions.csv
#   ② Verify SHA256(predictions.csv) vs predictions_hash.txt
#   ③ Verify SHA256(model.h5)        vs model_fingerprint.txt
#   ④ Construct hash chain from locked predictions
#   ⑤ Save evidence_chain.jsonl / hash_registry.csv /
#         verification_report.txt / hashchain_results.png


import os, hashlib, json, time, csv, re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import keras
from keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# GLOBAL CONSTANTS

MAX_LEN      = 200
VOCAB_SIZE   = 97
EMBED_DIM    = 64
FILTERS      = 128
LSTM_UNITS   = 128
DROPOUT      = 0.4
BATCH_SIZE   = 256
EPOCHS       = 50
THRESHOLD    = 0.50
CHAR_TO_INT  = {chr(i): i - 31 for i in range(32, 127)}

# Output artefact paths
DATASET_CSV       = "phishguard_gh_dataset.csv"
MODEL_KERAS       = "phishguard_gh_model.keras"
MODEL_H5          = "phishguard_gh_model.h5"
PREDICTIONS_CSV   = "predictions.csv"
DATASET_HASH_FILE = "dataset_hash.txt"
MODEL_FP_FILE     = "model_fingerprint.txt"
PRED_HASH_FILE    = "predictions_hash.txt"
METRICS_JSON      = "metrics.json"
CHAIN_JSONL       = "phishguard_evidence_chain.jsonl"
HASH_REGISTRY_CSV = "phishguard_hash_registry.csv"
VERIFY_TXT        = "phishguard_chain_verification.txt"



# UTILITY: SHA-256


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Stream-hash a file in 1 MB chunks — safe for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_hash_record(filepath: str, digest: str, label: str) -> None:
    """Persist a labelled hash record as JSON."""
    record = {
        "label":      label,
        "file":       os.path.basename(filepath),
        "sha256":     digest,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "size_bytes": os.path.getsize(filepath),
    }
    with open(filepath.replace(os.path.basename(filepath),
              label.lower().replace(" ", "_") + ".txt"), "w") as f:
        # written to named hash file, not over the source
        pass
    # Use the caller supplied output path instead
    return record


def save_hash_txt(out_path: str, record: dict) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"    Saved: {out_path}")



# UTILITY: URL encoder


def strip_scheme(url: str) -> str:
    # """Remove a leading http:// or https:// before character encoding.


    u = str(url).strip()
    for prefix in ("https://", "http://"):
        if u.lower().startswith(prefix):
            return u[len(prefix):]
    return u


def url_to_seq(url: str) -> list:
    url = strip_scheme(url).lower().strip()
    seq = [CHAR_TO_INT.get(c, VOCAB_SIZE - 1) for c in url[:MAX_LEN]]
    return seq + [0] * (MAX_LEN - len(seq))


# HASH-CHAIN CLASSES


class PhishGuardHashChain:
    """
    Forensic hash-chain logger.
    Accepts pre-computed (url, verdict, confidence, rationale) rows.
   """
    GENESIS_HASH = "0" * 64
    VERSION      = "PhishGuard-GH vCNN-LSTM "
    LEGAL        = ("Act 772 Ss.7-11 [Electronic Records]; "
                    "NRCD 323 S.135 [Evidence]; Act 1038 S.4(c) ")
    CSV_HEADERS  = ["seq_no", "timestamp_utc", "url_preview", "verdict",
                    "confidence_pct", "record_hash", "prev_hash",
                    "chain_link_valid"]

    def __init__(self):
        self.chain        = []
        self.last_hash    = self.GENESIS_HASH
        self.record_count = 0
        # Always start fresh — training just produced new predictions
        with open(HASH_REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.CSV_HEADERS)
        # Clear previous chain file
        if os.path.exists(CHAIN_JSONL):
            os.remove(CHAIN_JSONL)
        print("    New evidence chain initialised.")

    # internal hash (excludes record_hash field from its own input)
    def _hash_record(self, data: dict) -> str:
        d = {k: v for k, v in data.items() if k != "record_hash"}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()

    # log one prediction row
    def log(self, url: str, verdict: str, confidence: float,
            rationale: str = "") -> dict:
        self.record_count += 1
        record = {
            "seq_no":     self.record_count,
            "url":        url,
            "verdict":    verdict.upper(),
            "confidence": round(float(confidence), 6),
            "rationale":  rationale,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "model":      self.VERSION,
            "legal":      self.LEGAL,
            "prev_hash":  self.last_hash,
        }
        record["record_hash"] = self._hash_record(record)
        self.last_hash = record["record_hash"]
        self.chain.append(record)

        # FILE 1: append full JSON record
        with open(CHAIN_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # FILE 2: append CSV registry row
        prev = (self.chain[-2]["record_hash"]
                if len(self.chain) > 1 else self.GENESIS_HASH)
        valid_flag = "VALID" if record["prev_hash"] == prev else "BROKEN"
        url_preview = (url[:50] + "..." if len(url) > 50 else url)
        with open(HASH_REGISTRY_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                record["seq_no"],
                record["timestamp"],
                url_preview,
                record["verdict"],
                f"{record['confidence'] * 100:.2f}%",
                record["record_hash"],
                record["prev_hash"],
                valid_flag,
            ])
        return record

    # full-chain cryptographic verification
    def verify(self) -> tuple:
        """Returns (True, None) if intact, (False, broken_index)."""
        prev = self.GENESIS_HASH
        for i, r in enumerate(self.chain):
            if r["prev_hash"] != prev or self._hash_record(r) != r["record_hash"]:
                return False, i
            prev = r["record_hash"]
        return True, None

    def summary(self) -> dict:
        verdicts = [r["verdict"] for r in self.chain]
        ok, broken = self.verify()
        return {
            "total":      len(self.chain),
            "phishing":   verdicts.count("PHISHING"),
            "legitimate": verdicts.count("LEGITIMATE"),
            "valid":      ok,
            "broken":     broken,
        }

    # ─verification report
    def save_report(self, dataset_hash: str, model_hash: str,
                    pred_hash: str) -> bool:
        ok, broken = self.verify()
        s = self.summary()
        ts = datetime.now(timezone.utc).isoformat()
        lines = [
            "  PHISHGUARD-GH HASH-CHAIN VERIFICATION REPORT",
            "=" * 65,
            f"  Generated  : {ts}",
            f"  System     : {self.VERSION}",
            f"  Legal      : {self.LEGAL}",
            "",
            "  INTEGRITY ANCHORS",
            f"  Dataset SHA256     : {dataset_hash[:48]}...",
            f"  Model SHA256       : {model_hash[:48]}...",
            f"  Predictions SHA256 : {pred_hash[:48]}...",
            "",
            "  CHAIN STATISTICS",
            f"  Total records  : {s['total']:,}",
            f"  Phishing       : {s['phishing']:,}",
            f"  Legitimate     : {s['legitimate']:,}",
            "",
            f"  Genesis hash   : {self.GENESIS_HASH[:32]}...",
        ]
        if self.chain:
            lines.append(
                f"  Final hash     : {self.chain[-1]['record_hash'][:32]}...")
        lines += [
            f"  Chain status   : "
            f"{'VERIFIED  ALL LINKS INTACT' if ok else f'BROKEN at record {broken}'}",

        ]
        with open(VERIFY_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return ok

    def forensic_cert(self, seq_num: int) -> str:
        if seq_num < 1 or seq_num > len(self.chain):
            return f"ERROR: Record #{seq_num} does not exist."
        r  = self.chain[seq_num - 1]
        ok, _ = self.verify()
        return (
            f"PHISHGUARD-GH FORENSIC CERTIFICATE\n{'='*40}\n"
            f"Record  : {r['seq_no']}\n"
            f"URL     : {r['url']}\n"
            f"Verdict : {r['verdict']}\n"
            f"Conf    : {r['confidence']*100:.2f}%\n"
            f"Time    : {r['timestamp']}\n\n"
            f"Rationale:\n{r.get('rationale','N/A')}\n\n"
            f"Hash    : {r['record_hash']}\n"
            f"Prev    : {r['prev_hash']}\n"
            f"Chain   : {'VERIFIED' if ok else 'BROKEN'}"
        )



# RATIONALE BUILDER  (rule based secondary indicators)


_BRANDS     = ["mtn", "vodafone", "airtel", "ecobank", "gcb", "absa", "bog"]
_FREE_TLDS  = [".ml", ".tk", ".cf", ".ga", ".gq", ".xyz", ".top"]
_SHORTENERS = ["bit.ly", "tinyurl.com", "ow.ly", "rb.gy", "is.gd"]


def build_rationale(url: str, verdict: str, prob: float) -> str:
    url_l = url.lower()
    indicators = []
    try:
        hostname = urlparse(url).netloc or ""
    except Exception:
        hostname = ""
    for tld in _FREE_TLDS:
        if hostname.endswith(tld):
            indicators.append(f"Free TLD ({tld})")
            break
    if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", hostname):
        indicators.append("IP address hostname")
    for s in _SHORTENERS:
        if s in url_l:
            indicators.append(f"Shortener ({s})")
            break
    for b in _BRANDS:
        if b in url_l and any(hostname.endswith(t) for t in _FREE_TLDS):
            indicators.append(f"Brand '{b}' on suspicious domain")
            break
    if "@" in url:
        indicators.append("@ symbol in URL")
    if not url_l.startswith("https"):
        indicators.append("Non-HTTPS")
    ind = "; ".join(indicators[:3]) if indicators else "No rule-based indicators"
    return (f"[PRIMARY] CNN-LSTM: {prob*100:.2f}% {verdict}. "
            f"[SECONDARY] {ind}.")


# PIPELINE

if __name__ == "__main__":


    print("  PhishGuard-GH , Full Integrity Pipeline")
    print(f"  TensorFlow: {tf.__version__}   Keras: {keras.__version__}")



    # STAGE 1  Hash the raw dataset

    print("\nHashing raw dataset...")
    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")

    dataset_digest = sha256_file(DATASET_CSV)
    dataset_hash_record = {
        "label":      "DATASET_SNAPSHOT",
        "file":       DATASET_CSV,
        "sha256":     dataset_digest,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "size_bytes": os.path.getsize(DATASET_CSV),
        "note":       "Hashed before any preprocessing or splitting",
    }
    save_hash_txt(DATASET_HASH_FILE, dataset_hash_record)
    print(f"    SHA256: {dataset_digest[:48]}...")


    # STAGE 2  Load + preprocess dataset

    print("\nLoading and encoding dataset...")
    df = pd.read_csv(DATASET_CSV)
    all_urls   = df["url"].tolist()
    all_labels = df["label"].tolist()
    total      = len(df)

    X = np.array([url_to_seq(u) for u in all_urls], dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    print(f"    Total: {total:,}  "
          f"Phishing: {int((y==1).sum()):,}  "
          f"Legitimate: {int((y==0).sum()):,}")


    # STAGE 3  Train / val / test split

    print("\nSplitting dataset...")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    print(f"    Train: {len(X_tr):,}   Val: {len(X_val):,}   Test: {len(X_te):,}")


    # STAGE 4  Build model

    print("\nBuilding CNN-LSTM model...")
    inp    = keras.Input(shape=(MAX_LEN,), name="url_input")
    emb    = layers.Embedding(VOCAB_SIZE, EMBED_DIM,
                              name="char_embedding")(inp)
    c3     = layers.Conv1D(FILTERS, 3, activation="relu",
                           padding="same", name="conv_k3")(emb)
    c5     = layers.Conv1D(FILTERS, 5, activation="relu",
                           padding="same", name="conv_k5")(emb)
    merged = layers.Concatenate(name="merge_cnn")([c3, c5])
    pooled = layers.MaxPooling1D(2, name="maxpool")(merged)
    lstm   = layers.Bidirectional(
                 layers.LSTM(LSTM_UNITS, name="lstm"), name="bilstm")(pooled)
    drop   = layers.Dropout(DROPOUT, name="dropout")(lstm)
    dense  = layers.Dense(64, activation="relu", name="dense64")(drop)
    output = layers.Dense(1, activation="sigmoid", name="output")(dense)
    model  = keras.Model(inputs=inp, outputs=output, name="PhishGuard_GH")
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()


    # STAGE 5 Train

    print("\n[5/13] Training...")
    es  = callbacks.EarlyStopping(
              monitor="val_loss", patience=7,
              restore_best_weights=True, verbose=1)
    rlr = callbacks.ReduceLROnPlateau(
              monitor="val_loss", factor=0.5, patience=3, verbose=1)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlr],
        verbose=1,
    )


    # STAGE 6  Evaluate on held-out test set
    print("\nEvaluating on test set...")
    y_prob_test = model.predict(X_te, verbose=0).flatten()
    y_pred_test = (y_prob_test >= THRESHOLD).astype(int)

    acc  = accuracy_score(y_te, y_pred_test)
    prec = precision_score(y_te, y_pred_test, zero_division=0)
    rec  = recall_score(y_te, y_pred_test, zero_division=0)
    f1   = f1_score(y_te, y_pred_test, zero_division=0)
    auc  = roc_auc_score(y_te, y_prob_test)

    print(f"\n EVALUATION RESULTS")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%")
    print(f"    F1-Score  : {f1*100:.2f}%")
    print(f"    AUC-ROC   : {auc:.4f}")

    metrics = {
        "accuracy":       round(acc,  6),
        "precision":      round(prec, 6),
        "recall":         round(rec,  6),
        "f1_score":       round(f1,   6),
        "auc_roc":        round(auc,  6),
        "threshold":      THRESHOLD,
        "test_size":      len(X_te),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_digest,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {METRICS_JSON}")


    # STAGE 7  Save model → fingerprint

    print("\nSaving model and computing fingerprint...")
    model.save(MODEL_KERAS)
    model.save(MODEL_H5)

    # Fingerprint the .h5 (binary-stable; .keras is a zip and can vary)
    model_digest = sha256_file(MODEL_H5)
    model_fp_record = {
        "label":          "MODEL_FINGERPRINT",
        "file":           MODEL_H5,
        "sha256":         model_digest,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "size_bytes":     os.path.getsize(MODEL_H5),
        "architecture":   "CNN-LSTM (BiLSTM, dual Conv1D k3/k5)",
        "dataset_sha256": dataset_digest,
        "auc_roc":        round(auc, 6),
    }
    save_hash_txt(MODEL_FP_FILE, model_fp_record)
    print(f"    Model SHA256: {model_digest[:48]}...")


    # Training history + evaluation charts

    print("\n  Saving training charts...")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(history.history["accuracy"],     label="Train",  color="steelblue")
    a1.plot(history.history["val_accuracy"], label="Val",    color="orange")
    a1.set_title("Model Accuracy\nPhishGuard-GH", fontsize=13, fontweight="bold")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(history.history["loss"],     label="Train", color="steelblue")
    a2.plot(history.history["val_loss"], label="Val",   color="orange")
    a2.set_title("Model Loss\nPhishGuard-GH", fontsize=13, fontweight="bold")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Loss")
    a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_te, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title("Confusion Matrix — PhishGuard-GH", fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_te, y_prob_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"PhishGuard-GH (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve — PhishGuard-GH", fontsize=13, fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    Saved: training_history.png  confusion_matrix.png  roc_curve.png")


    # STAGE 8 Full-dataset inference

    print(f"\nFull dataset inference ({total:,} URLs, model in memory)...")
    t0 = time.time()
    INFER_BATCH = 512
    all_probs = []
    for i in range(0, total, INFER_BATCH):
        batch      = X[i: i + INFER_BATCH]
        probs_batch = model.predict(batch, verbose=0).flatten()
        all_probs.extend(probs_batch.tolist())
        done    = min(i + INFER_BATCH, total)
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 1
        print(f"    {done:>6,}/{total:,} | "
              f"{elapsed/60:.1f}m elapsed | "
              f"~{(total - done)/rate/60:.1f}m left | "
              f"{rate:.0f} URLs/sec", end="\r")

    print(f"\nInference done in {(time.time()-t0)/60:.1f}m")
    all_probs   = np.array(all_probs, dtype=np.float32)
    all_preds   = (all_probs >= THRESHOLD).astype(int)
    full_acc    = accuracy_score(all_labels, all_preds)
    print(f"    Full-dataset accuracy: {full_acc*100:.2f}%  "
          f"Phishing detected: {all_preds.sum():,}")


    # STAGE 9  Save predictions.csv

    print("\n saving predictions.csv...")
    with open(PREDICTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "true_label", "pred_label", "probability"])
        for url, tl, pl, pb in zip(all_urls, all_labels, all_preds, all_probs):
            writer.writerow([url, int(tl), int(pl), round(float(pb), 8)])
    print(f"    Saved: {PREDICTIONS_CSV}  ({os.path.getsize(PREDICTIONS_CSV)/1024/1024:.1f} MB)")


    # STAGE 10  Hash predictions.csv  → predictions_hash.txt

    print("\nLocking prediction artefact...")
    pred_digest = sha256_file(PREDICTIONS_CSV)
    pred_hash_record = {
        "label":          "PREDICTION_ARTIFACT",
        "file":           PREDICTIONS_CSV,
        "sha256":         pred_digest,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "size_bytes":     os.path.getsize(PREDICTIONS_CSV),
        "total_urls":     total,
        "phishing_count": int(all_preds.sum()),
        "legit_count":    int(total - all_preds.sum()),
        "dataset_sha256": dataset_digest,
        "model_sha256":   model_digest,
        "note": ("Hashed immediately after model.predict — "
                 "hash chain reads this file only, never calls model.predict"),
    }
    save_hash_txt(PRED_HASH_FILE, pred_hash_record)
    print(f"    Predictions SHA256: {pred_digest[:48]}...")


    # STAGE 11  Save .npy artefacts

    print("\nSaving SHAP artefacts (.npy files)...")
    np.save("X_test.npy",  X_te)
    np.save("y_test.npy",  y_te)
    np.save("y_prob.npy",  y_prob_test)  # test-set probs only (not full dataset)
    bg_idx = np.random.choice(len(X_tr), 500, replace=False)
    np.save("X_train_background.npy", X_tr[bg_idx])
    print("    Saved: X_test.npy  y_test.npy  y_prob.npy  X_train_background.npy")
    print("    (These are the test-split artefacts — shap_explainability.py "
          "is unaffected by the hash chain.)")


    #  HASH-CHAIN PIPELINE


    print(" HASH CHAIN PIPELINE ")

    # STAGE 12  Load predictions.csv

    print("\n[12a/13] Loading predictions.csv...")
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    loaded_total = len(pred_df)
    print(f"    Rows loaded: {loaded_total:,}")

    # Verify SHA256(predictions.csv)

    print("\nVerifying predictions.csv integrity...")
    verify_pred_digest = sha256_file(PREDICTIONS_CSV)
    if verify_pred_digest != pred_digest:
        raise RuntimeError(
            f"INTEGRITY FAILURE: predictions.csv has been modified!\n"
            f"  Expected : {pred_digest}\n"
            f"  Got      : {verify_pred_digest}\n"
            f"  The hash chain cannot be constructed from a tampered artefact."
        )
    print(f"    predictions.csv  ✓  VERIFIED  ({verify_pred_digest[:48]}...)")


    # STAGE 12  Verify SHA256(model.h5) matches model_fingerprint.txt
    print("\nVerifying model.h5 integrity...")
    verify_model_digest = sha256_file(MODEL_H5)
    if verify_model_digest != model_digest:
        raise RuntimeError(
            f"INTEGRITY FAILURE: model.h5 has been modified since saving!\n"
            f"  Expected : {model_digest}\n"
            f"  Got      : {verify_model_digest}"
        )
    print(f"    model.h5         ✓  VERIFIED  ({verify_model_digest[:48]}...)")


    #  Construct hash chain from locked predictions

    print(f"\nBuilding evidence hash chain ({loaded_total:,} records)...")
    chain     = PhishGuardHashChain()
    latencies = []
    t0        = time.time()
    LOG_EVERY = 1000

    for i, row in pred_df.iterrows():
        url      = str(row["url"])
        prob     = float(row["probability"])
        verdict  = "PHISHING" if int(row["pred_label"]) == 1 else "LEGITIMATE"
        rationale = build_rationale(url, verdict, prob)

        ts = time.perf_counter()
        chain.log(url, verdict, prob, rationale)
        latencies.append((time.perf_counter() - ts) * 1000)

        n = i + 1
        if n % LOG_EVERY == 0 or n == loaded_total:
            elapsed = time.time() - t0
            rate    = n / elapsed if elapsed > 0 else 1
            print(f"    {n:>6,}/{loaded_total:,} logged | "
                  f"{elapsed/60:.1f}m elapsed | "
                  f"~{(loaded_total - n)/rate/60:.1f}m left | "
                  f"avg {np.mean(latencies):.3f}ms/rec")

    lat = np.array(latencies)
    print(f"\n    Chain construction done in {(time.time()-t0)/60:.1f}m")


    # STAGE 13 Verify chain + save all outputs

    print("\nVerifying chain and saving forensic outputs...")
    chain_ok = chain.save_report(
        dataset_hash=dataset_digest,
        model_hash=model_digest,
        pred_hash=pred_digest,
    )
    s = chain.summary()

    print(f"    Chain status : {'✓ VERIFIED — ALL LINKS INTACT' if chain_ok else '✗ BROKEN'}")
    print(f"    Records      : {s['total']:,}")
    print(f"    Phishing     : {s['phishing']:,}")
    print(f"    Legitimate   : {s['legitimate']:,}")

    # Forensic certificate for record #1
    print("\n    Forensic Certificate (Record #1):")
    print(chain.forensic_cert(1))

    # Hash-chain result charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(lat, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(lat.mean(), color="red",    linestyle="--", lw=2,
                    label=f"Mean: {lat.mean():.3f}ms")
    axes[0].axvline(np.percentile(lat, 95), color="orange", linestyle="--", lw=2,
                    label=f"95th: {np.percentile(lat,95):.3f}ms")
    axes[0].set_xlabel("Latency (ms)"); axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Logging Latency\n({loaded_total:,} URLs)",
                      fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(np.cumsum(lat) / 1000, range(1, loaded_total + 1),
                 color="steelblue", lw=1.5)
    axes[1].set_xlabel("Time (sec)"); axes[1].set_ylabel("Records Logged")
    axes[1].set_title(f"Cumulative Throughput\n({loaded_total:,} URLs)",
                      fontweight="bold")
    axes[1].grid(alpha=0.3)

    axes[2].pie(
        [s["phishing"], s["legitimate"]],
        labels=[f"PHISHING\n{s['phishing']:,}", f"LEGITIMATE\n{s['legitimate']:,}"],
        colors=["#e74c3c", "#2ecc71"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes[2].set_title(f"Verdict Distribution\n({loaded_total:,} URLs)",
                      fontweight="bold")
    plt.tight_layout()
    plt.savefig("hashchain_results.png", dpi=300, bbox_inches="tight")
    plt.close()


    # FINAL SUMMARY

    print("\n" + "=" * 65)
    print("  OUTPUT FILES")
    print("=" * 65)
    output_files = [
        # Training artefacts
        ("dataset_hash.txt",           "SHA256 of raw dataset (pre-split)"),
        (METRICS_JSON,                 "Test-set evaluation metrics"),
        (MODEL_KERAS,                  "Trained model (.keras)"),
        (MODEL_H5,                     "Trained model (.h5)"),
        (MODEL_FP_FILE,                "Model SHA256 fingerprint"),
        (PREDICTIONS_CSV,              "Full-dataset predictions (locked)"),
        (PRED_HASH_FILE,               "SHA256 of predictions.csv"),
        ("X_test.npy",                 "Test split features (for SHAP)"),
        ("y_test.npy",                 "Test split labels  (for SHAP)"),
        ("y_prob.npy",                 "Test split probs   (for SHAP)"),
        ("X_train_background.npy",     "SHAP background sample"),
        ("training_history.png",       "Accuracy/loss curves"),
        ("confusion_matrix.png",       "Confusion matrix"),
        ("roc_curve.png",              "ROC curve"),
        # Hash-chain artefacts
        (CHAIN_JSONL,                  "Full forensic evidence chain"),
        (HASH_REGISTRY_CSV,            "Hash registry CSV"),
        (VERIFY_TXT,                   "Chain verification report"),
        ("hashchain_results.png",      "Latency / throughput / distribution"),
    ]
    for fname, desc in output_files:
        if os.path.exists(fname):
            size_mb = os.path.getsize(fname) / 1024 / 1024
            print(f"  {fname:<42} {size_mb:>6.1f} MB   {desc}")
        else:
            print(f"  {fname:<42}  (not produced)")

    print(f"\n  Integrity anchors:")
    print(f"    Dataset SHA256     : {dataset_digest[:48]}...")
    print(f"    Model SHA256       : {model_digest[:48]}...")
    print(f"    Predictions SHA256 : {pred_digest[:48]}...")
    print(f"    Chain              : {'VERIFIED' if chain_ok else 'BROKEN'}")
    print(f"\n  Mean log latency   : {lat.mean():.3f} ms")
    print(f"  95th pct latency   : {np.percentile(lat, 95):.3f} ms")
    log_s = lat.sum() / 1000.0
    print(f"  Throughput         : {loaded_total/log_s:.0f} records/sec")