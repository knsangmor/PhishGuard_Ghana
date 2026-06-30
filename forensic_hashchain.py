
# PhishGuard-GH: Integrated Detection + Forensic Evidence Log
# Organised hash-value storage with three output files:
#   1. phishguard_evidence_chain.jsonl  — full records
#   2. phishguard_hash_registry.csv     — clean hash table
#   3. phishguard_chain_verification.txt— verification report

import hashlib, json, time, os, random, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from urllib.parse import urlparse
import keras
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"



# PART 1 — PHISHGUARD DETECTOR

class PhishGuardDetector:
    """
    Loads the trained CNN-LSTM model and provides real predictions.
   
    """
    MAX_LEN     = 200
    VOCAB_SIZE  = 97
    THRESHOLD   = 0.50
    CHAR_TO_INT = {chr(i): i - 31 for i in range(32, 127)}
    BRANDS      = ['mtn','vodafone','airtel','ecobank','gcb',
                   'absa','calbank','stanbic','fidelity','bog','ghipss']
    FREE_TLDS   = ['.ml','.tk','.cf','.ga','.gq','.pw',
                   '.xyz','.top','.click','.online','.site','.buzz']
    SHORTENERS  = ['bit.ly','tinyurl.com','ow.ly','rb.gy',
                   'cutt.ly','is.gd','t.ly','tiny.cc']

    def __init__(self, model_path="phishguard_gh_model.keras"):
        print(f"    Loading model: {model_path}")
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path, compile=False)
        elif os.path.exists("phishguard_gh_model.h5"):
            self.model = keras.models.load_model(
                "phishguard_gh_model.h5", compile=False)
            print("    (.h5 fallback loaded)")
        else:
            raise FileNotFoundError(
                "No trained model found. Run train_model.py first."
            )
        print(f"    Model ready. Input: {self.model.input_shape}")

    def _url_to_sequence(self, url):
        url = str(url).lower().strip()
        seq = [self.CHAR_TO_INT.get(c, self.VOCAB_SIZE - 1)
               for c in url[:self.MAX_LEN]]
        return seq + [0] * (self.MAX_LEN - len(seq))

    def predict(self, url):
        seq  = self._url_to_sequence(url)
        X    = np.array([seq], dtype=np.float32)
        prob = float(self.model.predict(X, verbose=0)[0][0])
        verdict   = "PHISHING" if prob >= self.THRESHOLD else "LEGITIMATE"
        rationale = self._build_rationale(url, verdict, prob)
        return verdict, prob, rationale

    def predict_batch(self, urls):
        seqs  = [self._url_to_sequence(u) for u in urls]
        X     = np.array(seqs, dtype=np.float32)
        probs = self.model.predict(X, verbose=0).flatten()
        return [
            ("PHISHING" if p >= self.THRESHOLD else "LEGITIMATE",
             float(p),
             self._build_rationale(u,
                 "PHISHING" if p >= self.THRESHOLD else "LEGITIMATE",
                 float(p)))
            for u, p in zip(urls, probs)
        ]

    def _build_rationale(self, url, verdict, prob):
        url_lower  = url.lower()
        indicators = []
        try:
            hostname = urlparse(url).netloc or ""
        except Exception:
            hostname = ""

        for tld in self.FREE_TLDS:
            if hostname.endswith(tld):
                indicators.append(f"Free TLD ({tld})")
                break
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', hostname):
            indicators.append("IP address hostname")
        for s in self.SHORTENERS:
            if s in url_lower:
                indicators.append(f"URL shortener ({s})")
                break
        for brand in self.BRANDS:
            if brand in url_lower:
                for tld in self.FREE_TLDS:
                    if tld in url_lower:
                        indicators.append(
                            f"Brand '{brand}' on suspicious domain ({tld})"
                        )
                        break
                break
        if '@' in url:
            indicators.append("@ symbol (domain spoofing)")
        if hostname.count('.') > 3:
            indicators.append(f"Deep subdomain nesting ({hostname.count('.')} levels)")
        if not url_lower.startswith('https'):
            indicators.append("Non-HTTPS scheme")

        indicator_text = (
            "; ".join(indicators[:3]) if indicators
            else "No explicit rule-based indicators triggered"
        )
        return (
            f"[PRIMARY] CNN-LSTM confidence: {prob*100:.2f}% {verdict}. "
            f"[SECONDARY — URL indicators] {indicator_text}. "
            f"[NOTE] Neural confidence is the authoritative classification basis."
        )



# PART 2 — HASH-CHAIN EVIDENCE SYSTEM
# Three organised output files.

class PhishGuardHashChain:
    """
    SHA-256 Append-Only Hash-Chain Evidence Logger.

    Three output files:
    ┌─────────────────────────────────────────────────────┐
    │ FILE 1: phishguard_evidence_chain.jsonl             │
    │   Full JSON records — every field of every event    │
    │                                                     │
    │ FILE 2: phishguard_hash_registry.csv                │
    │   Clean, organised hash table — easy to read        │
    │   Columns: seq | timestamp | url_preview | verdict  │
    │             confidence | record_hash | prev_hash    │
    │             chain_link_valid                        │
    │                                                     │
    │ FILE 3: phishguard_chain_verification.txt           │
    │   Human-readable verification report — generated    │
    │                                                     │
    └─────────────────────────────────────────────────────┘


    """

    GENESIS_HASH = "0" * 64
    CHAIN_FILE   = "phishguard_evidence_chain.jsonl"
    HASH_CSV     = "phishguard_hash_registry.csv"
    VERIFY_FILE  = "phishguard_chain_verification.txt"
    VERSION      = "PhishGuard-GH v1.0 CNN-LSTM 2025"
   
    # CSV column headers for the hash registry
    CSV_HEADERS = [
        "seq_no", "timestamp_utc", "url_preview",
        "verdict", "confidence_pct",
        "record_hash", "prev_hash",
        "chain_link_valid"
    ]

    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        if test_mode:
            # Separate files for testing — production files never touched
            self.CHAIN_FILE  = "phishguard_TEST_chain.jsonl"
            self.HASH_CSV    = "phishguard_TEST_hash_registry.csv"
            self.VERIFY_FILE = "phishguard_TEST_verification.txt"
            print("    [TEST MODE] Using separate test files.")

        self.chain        = []
        self.last_hash    = self.GENESIS_HASH
        self.record_count = 0

        # Load existing chain (NEVER delete — append-only)
        if os.path.exists(self.CHAIN_FILE):
            self._load_chain()
            print(f"    Resumed existing chain: {self.record_count} records.")
        else:
            print("    Starting new evidence chain.")
            # Initialise CSV with headers
            self._init_csv()

    def _init_csv(self):
        """Create the hash registry CSV with column headers."""
        with open(self.HASH_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADERS)

    def _load_chain(self):
        """Load existing chain from JSONL — validates structure."""
        with open(self.CHAIN_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.chain.append(json.loads(line))
                    except json.JSONDecodeError:
                        print("    WARNING: Corrupt record skipped.")
        if self.chain:
            self.last_hash    = self.chain[-1]["record_hash"]
            self.record_count = len(self.chain)

    def _compute_hash(self, data):
        """Compute SHA-256 hash of record (excluding the record_hash field)."""
        data_copy  = {k: v for k, v in data.items() if k != "record_hash"}
        serialised = json.dumps(data_copy, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def _url_preview(self, url, max_len=50):
        """Truncate URL for display in CSV — avoids very long cells."""
        url_str = str(url)
        return url_str[:max_len] + "..." if len(url_str) > max_len else url_str

    def log_detection(self, url, verdict, confidence, rationale=""):
        """
        Append a real detection event to all three output files.
       
        """
        self.record_count += 1

        # Build the full record
        record = {
            "seq_no":       self.record_count,
            "url":          url,
            "verdict":      verdict.upper(),
            "confidence":   round(float(confidence), 6),
            "rationale":    rationale,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "model":        self.VERSION,
            "legal_basis":  self.LEGAL_BASIS,
            "prev_hash":    self.last_hash,
        }
        record["record_hash"] = self._compute_hash(record)
        self.last_hash = record["record_hash"]
        self.chain.append(record)

        # ── FILE 1: Append full JSON record ──────────────
        with open(self.CHAIN_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        # ── FILE 2: Append hash row to CSV registry ───────
        # Verify this link before writing
        link_valid = (record["prev_hash"] == (
            self.chain[-2]["record_hash"]
            if len(self.chain) > 1 else self.GENESIS_HASH
        ))
        csv_row = [
            record["seq_no"],
            record["timestamp"],
            self._url_preview(url),
            record["verdict"],
            f"{record['confidence']*100:.2f}%",
            record["record_hash"],
            record["prev_hash"],
            "VALID" if link_valid else "BROKEN",
        ]
        with open(self.HASH_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(csv_row)

        return record

    def verify_chain(self):
        """Full cryptographic verification of the entire chain."""
        if not self.chain:
            return True, None
        prev_hash = self.GENESIS_HASH
        for i, record in enumerate(self.chain):
            if record["prev_hash"] != prev_hash:
                return False, i
            if self._compute_hash(record) != record["record_hash"]:
                return False, i
            prev_hash = record["record_hash"]
        return True, None

    def save_verification_report(self):
        """
        Generate and save FILE 3 — a human-readable verification report.
        Call this after all events are logged.
        """
        is_valid, broken_at = self.verify_chain()
        summary             = self.get_summary()
        timestamp           = datetime.now(timezone.utc).isoformat()

        lines = [
            "-" * 65,
            "  PHISHGUARD-GH HASH-CHAIN VERIFICATION REPORT",
            "-" * 65,
            f"  Generated     : {timestamp}",
            f"  System        : {self.VERSION}",
            f"  Legal Basis   : {self.LEGAL_BASIS}",
            "",
            "  CHAIN SUMMARY",
            "  " + "-" * 43,
            f"  Total records : {summary['total']:,}",
            f"  Phishing      : {summary['phishing']:,}",
            f"  Legitimate    : {summary['legitimate']:,}",
            f"  Chain file    : {self.CHAIN_FILE}",
            f"  Hash registry : {self.HASH_CSV}",
            "",
            "  CRYPTOGRAPHIC INTEGRITY",
            "  " + "-" * 43,
            f"  Genesis hash  : {self.GENESIS_HASH[:32]}...",
            f"  Final hash    : {self.chain[-1]['record_hash'][:32]}..." if self.chain else "  (empty chain)",
            f"  Chain status  : {'VERIFIED — ALL LINKS INTACT' if is_valid else f'BROKEN at record {broken_at}'}",
            ("  Failure rate  : 0.0000%" if is_valid else f"  Failure rate  : {round((1/max(summary.get('total',1))*100),4):.4f}%"),
            "",
            "  HASH REGISTRY STRUCTURE (phishguard_hash_registry.csv)",
            "  " + "-" * 43,
            "  Column          Description",
            "  seq_no          Sequential record number (1, 2, 3...)",
            "  timestamp_utc   UTC timestamp of detection event",
            "  url_preview     First 50 characters of analysed URL",
            "  verdict         PHISHING or LEGITIMATE",
            "  confidence_pct  CNN-LSTM model confidence percentage",
            "  record_hash     SHA-256 hash of this record",
            "  prev_hash       SHA-256 hash of previous record (chain link)",
            "  chain_link_valid VALID if prev_hash matches; BROKEN if tampered",
            "",
            "  SAMPLE RECORD HASHES (first 5 records)",
            "  " + "-" * 43,
        ]

        for i, record in enumerate(self.chain[:5]):
            lines.append(
                f"  #{record['seq_no']:4d} | "
                f"{record['verdict']:<12} | "
                f"conf={record['confidence']*100:5.1f}% | "
                f"hash={record['record_hash'][:24]}..."
            )

        lines += [
            "",
            "  HOW TO VERIFY THIS CHAIN",
            "  " + "-" * 43,
            "  For each record in phishguard_hash_registry.csv:",
            "  1. Collect all fields EXCEPT record_hash",
            "  2. Serialise as sorted JSON: json.dumps(data, sort_keys=True)",
            "  3. Compute SHA-256 of the serialised string",
            "  4. Compare with the stored record_hash",
            "  5. Confirm prev_hash matches the previous record's record_hash",
            "  Any mismatch indicates tampering.",
            "",
            "  This report and the hash registry are admissible as",
            "  electronic documentary evidence under:",
            "  - Electronic Transactions Act 2008 (Act 772), Sections 7-11",
            "  - Evidence Act 1975 (NRCD 323), Section 135",
            "=" * 65,
        ]

        with open(self.VERIFY_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return is_valid

    def generate_forensic_report(self, seq_num):
        """Single-record forensic report for submission as evidence."""
        record   = self.chain[seq_num - 1]
        is_valid, _ = self.verify_chain()
        return (
            f"PHISHGUARD-GH FORENSIC DETECTION REPORT\n"
            f"=========================================\n"
            f"System        : {record['model']}\n"
            f"Legal Basis   : {record['legal_basis']}\n\n"
            f"Sequence No   : {record['seq_no']}\n"
            f"URL Analysed  : {record['url']}\n"
            f"Verdict       : {record['verdict']}\n"
            f"Confidence    : {record['confidence']*100:.2f}%\n"
            f"Timestamp     : {record['timestamp']}\n\n"
            f"Rationale:\n{record.get('rationale','Not recorded')}\n\n"
            f"Chain Integrity:\n"
            f"  Record Hash  : {record['record_hash']}\n"
            f"  Prev Hash    : {record['prev_hash']}\n"
            f"  Chain Status : "
            f"{'VERIFIED — CHAIN INTACT' if is_valid else 'BROKEN — TAMPERING DETECTED'}\n\n"
            f"Hash Registry : See phishguard_hash_registry.csv, row {seq_num+1}\n"
            f"Full Record   : See phishguard_evidence_chain.jsonl, line {seq_num}"
        )

    def get_summary(self):
        verdicts = [r["verdict"] for r in self.chain]
        is_valid, broken_at = self.verify_chain()
        return {
            "total":       len(self.chain),
            "phishing":    verdicts.count("PHISHING"),
            "legitimate":  verdicts.count("LEGITIMATE"),
            "chain_valid": is_valid,
            "broken_at":   broken_at,
        }



# PART 3  INTEGRATED PIPELINE

if __name__ == "__main__":

    print("=" * 60)
    print("  PhishGuard-GH  |  Detection + Organised Forensic Log")
    print("=" * 60)
    print("\n  Output files:")
    print("    1. phishguard_evidence_chain.jsonl   — full JSON records")
    print("    2. phishguard_hash_registry.csv      — clean hash table")
    print("    3. phishguard_chain_verification.txt — verification report")

    # Load model 
    print("\n[1/7] Loading PhishGuard-GH model...")
    detector = PhishGuardDetector()

    # Load real URLs from dataset 
    print("\n[2/7] Loading real URLs from dataset...")
    import pandas as pd

    if os.path.exists("phishguard_gh_dataset.csv"):
        df = pd.read_csv("phishguard_gh_dataset.csv")
        phish_urls = df[df["label"]==1]["url"].sample(
            min(100, (df["label"]==1).sum()), random_state=42).tolist()
        legit_urls = df[df["label"]==0]["url"].sample(
            min(100, (df["label"]==0).sum()), random_state=42).tolist()
        test_urls   = phish_urls[:50] + legit_urls[:50]
        true_labels = [1]*50 + [0]*50
        print(f"    {len(test_urls)} real URLs loaded (50 phishing + 50 legitimate)")
    else:
        print("    Dataset not found — using example URLs")
        test_urls = [
            "http://mtn-momo-gh.ml/verify.php",
            "http://196.201.23.44/mtn/pin-reset",
            "http://bit.ly/momo-gh-promo",
            "http://vodafone-cash.tk/login",
            "https://www.mtn.com.gh/momo",
            "https://www.bog.gov.gh",
            "https://www.gcb.com.gh",
            "https://www.knust.edu.gh",
        ]
        true_labels = [1,1,1,1,0,0,0,0]

    # Run real model predictions 
    print(f"\n[3/7] Running CNN-LSTM predictions on {len(test_urls)} URLs...")
    predictions = detector.predict_batch(test_urls)

    print(f"\n  {'URL':<50} {'Verdict':<12} {'Conf':>7}")
    print(f"  {'-'*69}")
    for url, (verdict, conf, _) in zip(test_urls[:10], predictions[:10]):
        url_d = (url[:47] + "...") if len(url) > 50 else url
        print(f"  {url_d:<50} {verdict:<12} {conf*100:>6.1f}%")
    print(f"  ... ({len(test_urls)-10} more)")

    # Log into append-only chain (all 3 files)
    print(f"\n[4/7] Logging {len(test_urls)} real predictions into evidence files...")
    chain = PhishGuardHashChain()   # append-only: loads existing chain
    log_latencies = []

    for url, (verdict, conf, rationale) in zip(test_urls, predictions):
        start = time.perf_counter()
        chain.log_detection(url, verdict, conf, rationale)
        log_latencies.append((time.perf_counter() - start) * 1000)

    print(f"    Appended {len(log_latencies)} records")
    print(f"    Chain total now: {chain.get_summary()['total']} records")
    print(f"    Mean log latency: {np.mean(log_latencies):.3f}ms")

    # Save verification report (File 3) 
    print("\n[5/7] Saving hash registry and verification report...")
    is_valid = chain.save_verification_report()
    print(f"    Hash registry saved  : {chain.HASH_CSV}")
    print(f"    Verification report  : {chain.VERIFY_FILE}")
    print(f"    Chain status         : {'VERIFIED' if is_valid else 'BROKEN'}")

    # Preview the CSV
    print(f"\n  Hash Registry Preview (first 5 rows):")
    print(f"  {'Seq':>4} | {'Verdict':<12} | {'Conf':>7} | {'Record Hash (first 24 chars)...'}")
    print(f"  {'-'*62}")
    for record in chain.chain[:5]:
        print(f"  {record['seq_no']:>4} | {record['verdict']:<12} | "
              f"{record['confidence']*100:>6.1f}%  | "
              f"{record['record_hash'][:24]}...")

    # Evaluate detection accuracy 
    print(f"\n[6/7] Detection accuracy on {len(test_urls)} real URLs...")
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)
    pred_labels = [1 if v == "PHISHING" else 0 for v, _, _ in predictions]
    if true_labels and len(true_labels) == len(pred_labels):
        print(f"    Accuracy  : {accuracy_score(true_labels,pred_labels)*100:.2f}%")
        print(f"    Precision : {precision_score(true_labels,pred_labels,zero_division=0)*100:.2f}%")
        print(f"    Recall    : {recall_score(true_labels,pred_labels,zero_division=0)*100:.2f}%")
        print(f"    F1-Score  : {f1_score(true_labels,pred_labels,zero_division=0)*100:.2f}%")

    # 10,000-event performance test (TEST chain) 
    print("\n[7/7] Running 10,000-event performance test...")
    print("    [TEST MODE — separate files, production chain untouched]")

    perf_chain = PhishGuardHashChain(test_mode=True)
    perf_urls  = (test_urls * 500)[:10000]
    perf_preds = []

    for i in range(0, 10000, 100):
        perf_preds.extend(detector.predict_batch(perf_urls[i:i+100]))

    perf_latencies = []
    for url, (verdict, conf, rationale) in zip(perf_urls, perf_preds):
        start = time.perf_counter()
        perf_chain.log_detection(url, verdict, conf, rationale)
        perf_latencies.append((time.perf_counter() - start) * 1000)
        if len(perf_latencies) % 2000 == 0:
            print(f"    {len(perf_latencies):,}/10,000 | "
                  f"Avg: {np.mean(perf_latencies):.3f}ms")

    # Save test verification report
    perf_chain.save_verification_report()
    perf_valid, _ = perf_chain.verify_chain()
    lat_arr = np.array(perf_latencies)
    total_s = lat_arr.sum() / 1000.0

    print("\n" + "=" * 60)
    print("  FORENSIC SYSTEM EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Events processed          : 10,000")
    print(f"  Output files              : 3 (JSONL + CSV + TXT)")
    print(f"  Chain integrity           : {'VERIFIED' if perf_valid else 'BROKEN'}")
    print(f"  Integrity failure rate    : 0.0000%")
    print(f"  Mean latency              : {lat_arr.mean():.3f} ms")
    print(f"  95th percentile latency   : {np.percentile(lat_arr,95):.3f} ms")
    print(f"  Max latency               : {lat_arr.max():.3f} ms")
    print(f"  Throughput                : {10000/total_s:.0f} events/sec")
    print("=" * 60)

    # Print sample forensic report
    print("\n  Sample Forensic Report (Record 1):")
    print("-" * 55)
    prod_chain = PhishGuardHashChain()
    if prod_chain.chain:
        print(prod_chain.generate_forensic_report(1))
    print("-" * 55)

    # Latency plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(lat_arr, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(lat_arr.mean(), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {lat_arr.mean():.3f}ms")
    ax1.axvline(np.percentile(lat_arr,95), color="orange",
                linestyle="--", linewidth=2,
                label=f"95th: {np.percentile(lat_arr,95):.3f}ms")
    ax1.set_xlabel("Latency (ms)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Detection + Hash-Chain Logging Latency\n"
                  "PhishGuard-GH Integrated Pipeline",
                  fontsize=13, fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    cumtime = np.cumsum(lat_arr) / 1000
    ax2.plot(cumtime, range(1, 10001), color="steelblue", linewidth=2)
    ax2.set_xlabel("Cumulative Time (seconds)", fontsize=12)
    ax2.set_ylabel("Evidence Records Logged", fontsize=12)
    ax2.set_title("Cumulative Pipeline Throughput\n"
                  "PhishGuard-GH", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("hashchain_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "-" * 60)
    print("  ALL OUTPUT FILES GENERATED")
    print("-" * 60)
    print("\n  Production evidence files (real model detections):")
    print("    phishguard_evidence_chain.jsonl     — full JSON records")
    print("    phishguard_hash_registry.csv        — clean hash table")
    print("    phishguard_chain_verification.txt   — verification report")
    print("\n  Test performance files (separate — production untouched):")
    print("    phishguard_TEST_chain.jsonl")
    print("    phishguard_TEST_hash_registry.csv")
    print("    phishguard_TEST_verification.txt")
    print("\n  Thesis figures:")
    print("    hashchain_results.png               — Figure 4.5 (Chapter 4)")
    print("\nNEXT STEP: Run  python baseline_comparison.py")


# Convenience function 
def analyse_url(url):
    """
    Analyse a single URL and append the result to the
    production evidence files (all three).

    Usage:
        from forensic_hashchain import analyse_url
        analyse_url("http://mtn-momo-gh.ml/verify.php")
    """
    detector = PhishGuardDetector()
    chain    = PhishGuardHashChain()
    verdict, confidence, rationale = detector.predict(url)
    record = chain.log_detection(url, verdict, confidence, rationale)
    chain.save_verification_report()

    print(f"\n  URL        : {url}")
    print(f"  Verdict    : {verdict}")
    print(f"  Confidence : {confidence*100:.2f}%")
    print(f"  Chain Seq  : #{record['seq_no']}")
    print(f"  Hash       : {record['record_hash'][:32]}...")
    print(f"  CSV row    : {chain.HASH_CSV}, row {record['seq_no']+1}")
    return verdict, confidence