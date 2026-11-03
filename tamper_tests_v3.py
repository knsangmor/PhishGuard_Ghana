"""Read-only source-chain tamper experiments. Uses only Python standard library."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

GENESIS = '0' * 64

def record_hash(record):
    return hashlib.sha256(json.dumps(
        {k: v for k, v in record.items() if k != 'record_hash'},
        sort_keys=True, ensure_ascii=True,
    ).encode()).hexdigest()

def audit(chain):
    previous = GENESIS
    for index, record in enumerate(chain):
        if record.get('prev_hash') != previous:
            return False, index + 1, 'link mismatch'
        if record_hash(record) != record.get('record_hash'):
            return False, index + 1, 'record hash mismatch'
        previous = record['record_hash']
    return True, None, 'internally consistent'

def anchor_check(chain, count, final_hash):
    return len(chain) == count and bool(chain) and chain[-1]['record_hash'] == final_hash

def changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + '[EDIT]'
    return {'tampered': True, 'original': value}

def rehash(chain):
    previous = GENESIS
    for record in chain:
        record['prev_hash'] = previous
        record['record_hash'] = record_hash(record)
        previous = record['record_hash']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chain', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        demo = [{'seq_no': i, 'verdict': 'LEGITIMATE', 'confidence': .9} for i in range(1, 4)]
        rehash(demo)
        assert audit(demo)[0]
        truncated = demo[:-1]
        assert audit(truncated)[0]
        assert not anchor_check(truncated, 3, demo[-1]['record_hash'])
        edited = [dict(r) for r in demo]
        edited[1]['verdict'] = 'PHISHING'
        assert not audit(edited)[0]
        rehash(edited)
        assert audit(edited)[0]
        assert not anchor_check(edited, 3, demo[-1]['record_hash'])
        print('SELF-TEST PASSED. No model, dataset or evidence files accessed.')
        return
    project = Path(__file__).resolve().parent
    source = args.chain or project / 'corrected_results_v3' / f'seed_{args.seed}' / 'phishguard_evidence_chain.jsonl'
    source = source.resolve()
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    original = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    valid, _, reason = audit(original)
    if not original or not valid:
        raise ValueError(f'Source chain is empty or invalid: {reason}')
    if len(original) < 3:
        raise ValueError('At least three records required')
    count, final_hash = len(original), original[-1]['record_hash']
    # Captured from the original before simulated attacks. These are reference
    # anchors, not externally authenticated or independently trusted anchors.
    results = []
    def test(name, records, expected_internal):
        intact, position, reason = audit(records)
        anchored = intact and anchor_check(records, count, final_hash)
        assert intact == expected_internal, f'Unexpected outcome: {name}'
        assert not anchored or name == 'untampered', name
        results.append({'scenario': name, 'records': len(records),
                        'internal_check_passed': intact,
                        'reference_anchor_check_passed': anchored,
                        'first_failure_position': position, 'reason': reason})
    test('untampered', original, True)
    positions = {'first': 0, 'middle': count // 2, 'last': count - 1}
    for location, index in positions.items():
        for field in original[index]:
            records = list(original)
            records[index] = dict(original[index])
            records[index][field] = changed(records[index][field])
            test(f'edit_{location}_{field}', records, False)
        records = original[:index] + original[index + 1:]
        test(f'delete_{location}', records, location == 'last')
    for location, index in {'first': 0, 'middle': count // 2, 'last': count - 2}.items():
        records = list(original)
        records[index], records[index + 1] = records[index + 1], records[index]
        test(f'swap_adjacent_{location}', records, False)
    for removed in sorted({1, min(10, count - 1), max(1, count // 10)}):
        test(f'truncate_tail_{removed}', original[:-removed], True)
    test('remove_entire_chain', [], True)
    records = [dict(r) for r in original]
    records[count // 2]['verdict'] = changed(records[count // 2]['verdict'])
    rehash(records)
    test('edit_and_recompute_all_hashes', records, True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError('Source chain changed during experiment')
    output = source.parent / 'tamper_tests_v3'
    output.mkdir(exist_ok=False)
    with (output / 'tamper_results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    report = {'source_file': str(source), 'source_sha256': source_sha,
              'reference_record_count': count, 'reference_final_hash': final_hash,
              'anchor_origin': 'Captured locally from source before simulated attacks; not externally authenticated',
              'limitations': 'A plain hash chain cannot detect valid-prefix truncation, absence of the entire chain, or an attacker who rewrites data and recomputes all hashes. A separately protected and authenticated expected count/final hash is needed.',
              'downstream_status': 'A broken link invalidates chain-wide assurance; later records may still have individually valid hashes.',
              'results': results}
    (output / 'tamper_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'TAMPER TESTS COMPLETE: {len(results)} scenarios; {count:,} source records')
    print('Source chain unchanged. All expected outcomes passed.')
    print('Saved to:', output)
    print(report['limitations'])

if __name__ == '__main__':
    main()
