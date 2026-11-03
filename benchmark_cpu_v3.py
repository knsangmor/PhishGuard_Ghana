"""CPU inference and matched logging benchmark; no model training or source writes."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--records', type=int, default=5000)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.records < 1 or args.repeats < 2:
        raise ValueError('Use at least 1 record and 2 repeats')
    run = ROOT / 'corrected_results_v3' / f'seed_{args.seed}'
    module_path = ROOT / 'model_hash_v3.py'
    required = [module_path, run / 'phishguard_gh_model.keras', run / 'predictions.csv', run / 'predictions_hash.txt', run / 'dataset_training_snapshot.csv', run / 'dataset_hash.txt', ROOT / 'dataset_fix_outputs_v3' / 'test.csv']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    saved_hash = json.loads((run / 'predictions_hash.txt').read_text())
    if sha(run / 'predictions.csv') != saved_hash['sha256']:
        raise ValueError('Prediction fingerprint mismatch')
    dataset_record = json.loads((run / 'dataset_hash.txt').read_text())
    if sha(run / 'dataset_training_snapshot.csv') != dataset_record['sha256']:
        raise ValueError('Snapshot fingerprint mismatch')
    test_path = ROOT / 'dataset_fix_outputs_v3' / 'test.csv'
    if sha(test_path) != dataset_record['partitions']['test']['sha256']:
        raise ValueError('Test partition fingerprint mismatch')
    import numpy as np
    import pandas as pd
    test = pd.read_csv(test_path, keep_default_na=False)
    snapshot = pd.read_csv(run / 'dataset_training_snapshot.csv', keep_default_na=False)
    predictions = pd.read_csv(run / 'predictions.csv', keep_default_na=False)
    if not snapshot[['url','label']].rename(columns={'label':'true_label'}).equals(predictions[['url','true_label']]):
        raise ValueError('Snapshot/prediction alignment mismatch')
    if not predictions.probability.between(0,1).all():
        raise ValueError('Invalid probabilities')
    spec = importlib.util.spec_from_file_location('benchmark_pipeline', module_path)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)  # guarded main block does not train
    import tensorflow as tf
    import keras
    print(f'CHECK PASSED: seed {args.seed}, {len(test):,} test URLs, verified prediction snapshot.')
    if args.records > len(predictions):
        raise ValueError('Requested sample exceeds prediction count')
    if args.check_only:
        print('No training, inference benchmark or output directories created.')
        return
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'cpu_benchmarks_v3' / timestamp
    output.mkdir(parents=True, exist_ok=False)
    stages=[];logging=[];orders=[]
    def stage(name, repeat, start, count):
        elapsed=time.perf_counter()-start
        stages.append({'stage':name,'repeat':repeat,'records':count,'seconds':elapsed,'ms_per_record':1000*elapsed/count})
    for repeat in range(args.repeats):
        start=time.perf_counter()
        x=np.array([pipeline.url_to_seq(u) for u in test.url],dtype=np.int32)
        stage('test_url_encoding',repeat,start,len(x))
    start=time.perf_counter()
    with tf.device('/CPU:0'):
        model=keras.models.load_model(run / 'phishguard_gh_model.keras',compile=False)
    load_seconds=time.perf_counter()-start
    with tf.device('/CPU:0'):
        model.predict(x[:256],batch_size=256,verbose=0)
        for repeat in range(args.repeats):
            print(f'Inference repeat {repeat+1}/{args.repeats}',flush=True)
            start=time.perf_counter()
            prob=model.predict(x,batch_size=256,verbose=0).reshape(-1)
            stage('warm_cpu_model_predict_batch256',repeat,start,len(x))
    saved_test=pd.read_csv(run / 'test_predictions.csv',keep_default_na=False)
    if not test.url.equals(saved_test.url):
        raise ValueError('Test URL order changed')
    max_probability_difference=float(np.max(np.abs(prob-saved_test.probability.to_numpy())))
    prediction_agreement=float(np.mean((prob>=.5)==saved_test.pred_label.to_numpy()))
    for repeat in range(args.repeats):
        start=time.perf_counter()
        pd.DataFrame({'url':test.url,'true_label':test.label,'probability':prob}).to_csv(output / 'prediction_write_temporary.csv',index=False)
        stage('prediction_csv_write',repeat,start,len(test))
    (output / 'prediction_write_temporary.csv').unlink()
    # Fixed uniform sample across the saved full-dataset prediction order.
    indices=np.linspace(0,len(predictions)-1,args.records,dtype=int)
    sample=predictions.iloc[indices]
    items=[]
    for row in sample.itertuples(index=False):
        verdict='PHISHING' if int(row.pred_label)==1 else 'LEGITIMATE'
        confidence=float(row.probability) if verdict=='PHISHING' else 1-float(row.probability)
        items.append((row.url,verdict,confidence,pipeline.build_rationale(row.url,verdict,float(row.probability))))
    for repeat in range(args.repeats):
        start=time.perf_counter()
        for row in sample.itertuples(index=False):
            pipeline.build_rationale(row.url,'PHISHING' if int(row.pred_label)==1 else 'LEGITIMATE',float(row.probability))
        stage('rationale_generation',repeat,start,len(items))
    class PlainLogger(pipeline.PhishGuardHashChain):
        def _hash_record(self,data):
            return '0'*64
    rng=np.random.default_rng(42)
    for repeat in range(args.repeats):
        modes=['plain','hash_chain']
        rng.shuffle(modes)
        orders.append({'repeat':repeat,'order':modes})
        for mode in modes:
            print(f'Logging repeat {repeat+1}/{args.repeats}: {mode}, {len(items):,} records',flush=True)
            pipeline.CHAIN_JSONL=str(output / f'{mode}_temporary.jsonl')
            pipeline.HASH_REGISTRY_CSV=str(output / f'{mode}_temporary.csv')
            logger=(PlainLogger if mode=='plain' else pipeline.PhishGuardHashChain)()
            latencies=[]
            start=time.perf_counter()
            for item in items:
                tick=time.perf_counter()
                logger.log(*item)
                latencies.append((time.perf_counter()-tick)*1000)
            elapsed=time.perf_counter()-start
            assert len(logger.chain)==len(items)
            row={'repeat':repeat,'mode':mode,'records':len(items),'seconds':elapsed,
                 'mean_record_latency_ms':float(np.mean(latencies)),
                 'p95_record_latency_ms':float(np.percentile(latencies,95)),
                 'throughput_records_per_second':len(items)/elapsed}
            logging.append(row)
            table(output / 'logging_repeats.csv',logging)
            if mode=='hash_chain':
                tick=time.perf_counter()
                ok,_=logger.verify()
                stage('chain_verification',repeat,tick,len(items))
                assert ok
            Path(pipeline.CHAIN_JSONL).unlink()
            Path(pipeline.HASH_REGISTRY_CSV).unlink()
            del logger
    overhead=[]
    for repeat in range(args.repeats):
        a=next(r for r in logging if r['repeat']==repeat and r['mode']=='plain')
        b=next(r for r in logging if r['repeat']==repeat and r['mode']=='hash_chain')
        overhead.append({'repeat':repeat,'extra_ms_per_record':(b['seconds']-a['seconds'])*1000/len(items),'overhead_percent':100*(b['seconds']/a['seconds']-1)})
    table(output / 'stages.csv',stages)
    table(output / 'paired_logging_overhead.csv',overhead)
    summary={}
    for name in sorted({r['stage'] for r in stages}):
        rows=[r for r in stages if r['stage']==name]
        summary[name]={'mean_ms_per_record':statistics.mean(r['ms_per_record'] for r in rows),'sd_ms_per_record':statistics.stdev(r['ms_per_record'] for r in rows)}
    summary['hashing_overhead']={'mean_extra_ms_per_record':statistics.mean(r['extra_ms_per_record'] for r in overhead),'sd_extra_ms_per_record':statistics.stdev(r['extra_ms_per_record'] for r in overhead),'mean_percent':statistics.mean(r['overhead_percent'] for r in overhead)}
    report={'seed':args.seed,'test_records':len(test),'logging_sample_records':len(items),'repeats':args.repeats,
        'python':sys.version,'platform':platform.platform(),'logical_cpus':os.cpu_count(),'tensorflow':tf.__version__,'keras':keras.__version__,
        'visible_devices':[str(d) for d in tf.config.list_physical_devices()], 'oneDNN_env':os.environ['TF_ENABLE_ONEDNN_OPTS'],
        'tensorflow_intra_threads':tf.config.threading.get_intra_op_parallelism_threads(),'tensorflow_inter_threads':tf.config.threading.get_inter_op_parallelism_threads(),
        'model_load_seconds_single_measurement':load_seconds,'max_probability_difference_from_saved':max_probability_difference,'threshold_0_5_prediction_agreement':prediction_agreement,
        'input_hashes':{p.name:sha(p) for p in [module_path,run/'phishguard_gh_model.keras',run/'predictions.csv',test_path]},
        'logging_orders':orders,'summary':summary,
        'method':'Plain control subclasses the existing logger and replaces canonical serialization plus SHA256 with a fixed 64-character placeholder. All record construction, timestamps, memory retention, JSONL/CSV append patterns and link-flag calculations are otherwise identical. Plain output offers no integrity guarantee. Initialization excluded. Order randomized within paired repeats; same fixed evenly spaced sample. Rationale generation excluded from logging and timed separately.',
        'limitations':'Logging uses OS-cached writes with file close per record, no fsync or durable-disk guarantee. Warm batched predict includes Keras API overhead; ms/record is amortized throughput, not single-request latency. Load time is one cold API call without OS cache control. Negative overhead is retained as timing noise. Source hardware power, thermal state and competing processes can affect results. Sample benchmarks do not measure complete pipeline end-to-end latency.'}
    (output/'benchmark_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nBENCHMARK COMPLETE. No training performed. Source files unchanged.')
    print(json.dumps(summary,indent=2))
    print('Saved to:',output)
if __name__=='__main__':
    main()
