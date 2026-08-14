
#PhishGuard-GH — Tamper-Injection Experiment



import json
import hashlib
import csv
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHAIN_FILE = "phishguard_evidence_chain.jsonl"
GENESIS_HASH = "0" * 64


TAMPER_SEQ_NO = 25000


def hash_record(record: dict) -> str:
 
    d = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def load_chain(path):
    chain = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chain.append(json.loads(line))
    return chain


def full_audit(chain):

    results = []
    prev_hash = GENESIS_HASH
    chain_intact_so_far = True
    for i, record in enumerate(chain):
        self_valid = hash_record(record) == record["record_hash"]
        link_valid = record["prev_hash"] == prev_hash
        record_ok = self_valid and link_valid and chain_intact_so_far
        if not (self_valid and link_valid):
            chain_intact_so_far = False  # everything from here on is unverifiable
        results.append({
            "seq_no": record["seq_no"],
            "self_valid": self_valid,
            "link_valid": link_valid,
            "status": "VALID" if record_ok else "BROKEN",
        })
        
        prev_hash = record["record_hash"]
    return results


def main():
    print("=" * 70)
    print("  PhishGuard-GH Tamper-Injection Experiment")
    print("=" * 70)

    print(f"\n[1/5] Loading evidence chain from {CHAIN_FILE} ...")
    chain = load_chain(CHAIN_FILE)
    print(f"    Loaded {len(chain):,} records.")

    print("\n[2/5] Baseline audit (untampered chain) ...")
    baseline = full_audit(chain)
    baseline_broken = [r for r in baseline if r["status"] == "BROKEN"]
    print(f"    Broken records: {len(baseline_broken)} / {len(baseline):,}")
    assert len(baseline_broken) == 0, "Chain is not intact BEFORE tampering — investigate before proceeding."
    print("    Confirmed: untampered chain is fully VALID.")

    print(f"\n[3/5] Tampering with record #{TAMPER_SEQ_NO} ...")
    target_idx = next(i for i, r in enumerate(chain) if r["seq_no"] == TAMPER_SEQ_NO)
    original_verdict = chain[target_idx]["verdict"]
    tampered_verdict = "PHISHING" if original_verdict == "LEGITIMATE" else "LEGITIMATE"
    print(f"    Record #{TAMPER_SEQ_NO}: verdict '{original_verdict}' -> '{tampered_verdict}'")
    print(f"    record_hash left UNCHANGED (this is the attack: edit data, don't recompute hash).")
    chain[target_idx]["verdict"] = tampered_verdict
    # deliberately NOT recomputing chain[target_idx]["record_hash"] here.

    print("\n[4/5] Auditing the tampered chain ...")
    audited = full_audit(chain)

    before = [r for r in audited if r["seq_no"] < TAMPER_SEQ_NO]
    at_tamper = [r for r in audited if r["seq_no"] == TAMPER_SEQ_NO][0]
    after = [r for r in audited if r["seq_no"] > TAMPER_SEQ_NO]

    before_valid = sum(1 for r in before if r["status"] == "VALID")
    after_broken = sum(1 for r in after if r["status"] == "BROKEN")

    print(f"    Records before #{TAMPER_SEQ_NO}: {before_valid:,} / {len(before):,} still VALID")
    print(f"    Record #{TAMPER_SEQ_NO} itself: {at_tamper['status']} "
          f"(self_valid={at_tamper['self_valid']}, link_valid={at_tamper['link_valid']})")
    print(f"    Records after #{TAMPER_SEQ_NO}: {after_broken:,} / {len(after):,} correctly reported BROKEN")

    assert before_valid == len(before), "Some record before the tamper point unexpectedly broke."
    assert at_tamper["status"] == "BROKEN", "Tampered record was not detected as broken."
    assert after_broken == len(after), "Not every downstream record was marked broken — investigate."

    print("\n    RESULT: tamper localised exactly at the point of injection; "
          "every record before it remains valid; every record after it is "
          "correctly reported as unverifiable.")

    print("\n[5/5] Saving results table and figure ...")
    with open("tamper_injection_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_no", "self_valid", "link_valid", "status"])
        for r in audited:
            w.writerow([r["seq_no"], r["self_valid"], r["link_valid"], r["status"]])
    print("    Saved: tamper_injection_results.csv (full per-record audit)")

    # Simple visual: status by seq_no around the tamper point
    window = 50
    lo = max(0, target_idx - window)
    hi = min(len(audited), target_idx + window)
    window_records = audited[lo:hi]
    seqs = [r["seq_no"] for r in window_records]
    statuses = [1 if r["status"] == "VALID" else 0 for r in window_records]

    plt.figure(figsize=(10, 3))
    colors = ["#2ecc71" if s == 1 else "#e74c3c" for s in statuses]
    plt.bar(range(len(seqs)), statuses, color=colors, width=1.0)
    plt.axvline(target_idx - lo, color="black", linestyle="--", linewidth=1,
                label=f"Tamper point (record #{TAMPER_SEQ_NO})")
    plt.yticks([0, 1], ["BROKEN", "VALID"])
    plt.xlabel(f"Records #{seqs[0]} to #{seqs[-1]}")
    plt.title("Tamper-Injection Experiment\nChain status before/after record "
              f"#{TAMPER_SEQ_NO} tampering", fontweight="bold")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig("tamper_injection_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("    Saved: tamper_injection_results.png (status around the tamper point)")

    print("\n" + "=" * 70)
    print("  SUMMARY FOR THESIS WRITE-UP")
    print("=" * 70)
    print(f"  Chain length                 : {len(chain):,} records")
    print(f"  Tamper point                 : record #{TAMPER_SEQ_NO}")
    print(f"  Field altered                : verdict ({original_verdict} -> {tampered_verdict})")
    print(f"  Records before tamper point   : {len(before):,} / {len(before):,} VALID (unaffected)")
    print(f"  Tampered record               : BROKEN (hash mismatch detected)")
    print(f"  Records after tamper point    : {len(after):,} / {len(after):,} BROKEN (correctly propagated)")
    print("=" * 70)


if __name__ == "__main__":
    main()