"""One seed-42 validation-only pilot: normalize empty HTTP(S) root paths to /.
Frozen v3 data and original models/results remain unchanged. No original test or
external-holdout predictions. A new holdout must remain reserved through development.
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
PARTS={'train':'28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f',
'validation':'8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e',
'test':'22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
HOLDOUT_SHA='b8a069c5d9671e74ad4c0f0b29ef1b3a40bd7c0c58c02d6884ae4e32bceeded8'
SOURCE_SHA='d83ce942075dd63ed4d11560cfdcd9d512caa3d680e292f22cab484e8f074d01'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def normalize(url):
    u=str(url).strip()
    if not u or any(c.isspace() for c in u) or any(c in u for c in "'\"\\"):return u
    m=re.match(r'^https?://',u,re.I)
    if m:prefix,rest=u[:m.end()],u[m.end():]
    elif u.startswith('//'):prefix,rest='//',u[2:]
    else:
        if '://' in u:return u
        prefix,rest='',u
    try:
        p=urlsplit('https://'+rest)
        if not p.hostname or p.path!='':return u
        _=p.port
        return prefix+p.netloc+'/'+rest[len(p.netloc):]
    except ValueError:return u


def encode(url,corrected=True):
    text=normalize(url) if corrected else str(url).strip()
    for prefix in ('https://','http://'):
        if text.lower().startswith(prefix):text=text[len(prefix):];break
    seq=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in text.lower().strip()[:200]]
    return seq+[0]*(200-len(seq))


def build(keras):
    from keras import layers,regularizers
    inp=keras.Input(shape=(200,),name='url_input')
    h=layers.Embedding(97,64,name='char_embedding')(inp)
    h=layers.SpatialDropout1D(.2,name='embed_dropout')(h)
    branches=[layers.Conv1D(128,k,activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4),name=f'conv_k{k}')(h) for k in (3,5)]
    h=layers.Concatenate(name='merge_cnn')(branches)
    h=layers.MaxPooling1D(2,name='maxpool')(h)
    h=layers.Bidirectional(layers.LSTM(64,kernel_regularizer=regularizers.l2(1e-4),name='lstm'),name='bilstm')(h)
    h=layers.Dropout(.5,name='dropout')(h)
    h=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-4),name='dense64')(h)
    h=layers.Dropout(.3,name='dropout2')(h)
    out=layers.Dense(1,activation='sigmoid',name='output')(h)
    model=keras.Model(inp,out,name='PhishGuard_GH')
    model.compile(optimizer=keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=['accuracy'])
    return model


def canonical_config(value):
    if isinstance(value,dict):return {k:canonical_config(v) for k,v in value.items() if k!='shared_object_id'}
    if isinstance(value,(list,tuple)):return [canonical_config(v) for v in value]
    return value


def signature(model):return [(l.__class__.__name__,canonical_config(l.get_config())) for l in model.layers]


def dump(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def self_test():
    for u,v in [('http://example.com','http://example.com/'),
                ('https://www.example.com?x=1#route','https://www.example.com/?x=1#route'),
                ('example.com#','example.com/#')]:
        assert normalize(u)==v and encode(u)==encode(v)
        assert normalize(normalize(u))==normalize(u)
    assert normalize('https://example.com/a')=='https://example.com/a'
    assert normalize('https://example.com/a/')=='https://example.com/a/'
    assert encode('www.example.com')!=encode('example.com')
    assert normalize("'example.com'")=="'example.com'"
    print('SELF-TEST PASSED: root equivalence, hostname preservation and idempotence.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--seed',type=int,choices=[42],help='Only the agreed seed-42 pilot is enabled')
    a=ap.parse_args()
    if a.self_test:self_test();return
    import numpy as np
    import pandas as pd
    root=Path(__file__).resolve().parent
    # Check holdout identity only: never parse URLs or feed it to a model here.
    reserve=root/'reserved_mendeley_holdout'
    freeze=json.loads((reserve/'freeze_report.json').read_text())
    if freeze['partition_sha256']!=PARTS or freeze['holdout_sha256']!=HOLDOUT_SHA:
        raise ValueError('Reserved holdout identity differs')
    if sha(reserve/'holdout.csv')!=HOLDOUT_SHA:raise ValueError('Reserved holdout bytes differ')
    if freeze['retained_rows']!=7659 or freeze['evaluation_policy']['model_scoring_performed'] is not False:
        raise ValueError('Unexpected holdout freeze status')
    audits=[]
    for p in sorted((root/'root_normalization_preflight').glob('*/report.json')):
        r=json.loads(p.read_text())
        if r.get('status')=='NO_CONFLICTS_FOUND' and r.get('partition_sha256')==PARTS and r.get('source_sha256')==SOURCE_SHA:
            audits.append((p,r))
    if not audits:raise ValueError('A passing root-normalization full-source audit is required')
    audit_path,audit=audits[-1]
    if sha(root/'malicious_phish.csv')!=SOURCE_SHA:raise ValueError('Original source identity differs')
    parts={};sets={};counts={}
    for name,h in PARTS.items():
        p=root/'dataset_fix_outputs_v3'/f'{name}.csv'
        if sha(p)!=h:raise ValueError(f'Frozen {name} bytes differ')
        d=pd.read_csv(p,keep_default_na=False)
        if not d.label.isin([0,1]).all():raise ValueError(f'Invalid {name} labels')
        keys=[bytes(encode(u)) for u in d.url]
        labels={}
        for k,y in zip(keys,d.label):labels.setdefault(k,set()).add(int(y))
        if any(len(v)>1 for v in labels.values()):raise ValueError(f'Conflicting new encodings in {name}')
        sets[name]=set(keys)
        counts[name]={'rows':len(d),'encoded_inputs_changed':sum(encode(u)!=encode(u,False) for u in d.url),
                      'repeated_input_rows':len(d)-len(sets[name])}
        if name in ['train','validation']:parts[name]=d
    if any(sets[x]&sets[y] for x,y in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Proposed encoding overlaps across partitions')
    expected_changed={'train':1736,'validation':474,'test':398}
    if any(counts[n]['encoded_inputs_changed']!=expected_changed[n] for n in counts):
        raise ValueError('Proposed normalization differs from reviewed audit')
    ref=root/'corrected_results_v3'/'seed_42'
    saved_val=pd.read_csv(ref/'validation_predictions.csv',keep_default_na=False)
    val=parts['validation']
    if val.url.tolist()!=saved_val.url.tolist() or val.label.tolist()!=saved_val.true_label.tolist():
        raise ValueError('Reference validation alignment differs')
    import keras
    keras.backend.clear_session();keras.utils.set_random_seed(42)
    architecture=build(keras)
    saved_model=keras.models.load_model(ref/'phishguard_gh_model.keras',compile=False)
    if architecture.count_params()!=244673 or signature(architecture)!=signature(saved_model):
        raise ValueError('Reference architecture differs; no training started')
    del architecture,saved_model
    keras.backend.clear_session()
    print('CHECK PASSED: frozen v3, audited normalization, reserved holdout hash and exact reference architecture.')
    print('Encoding changes:',counts)
    print('Pilot uses train/validation only; no test or external predictions.')
    if a.check_only:
        print('No training or output directories created.');return
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,roc_curve
    def evaluate(y,p):
        q=p>=.5;tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
        return {'accuracy':float(accuracy_score(y,q)),'precision':float(precision_score(y,q,zero_division=0)),
            'recall':float(recall_score(y,q)),'f1_score':float(f1_score(y,q,zero_division=0)),
            'auc_roc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
            'false_positive_rate':float(fp/(tn+fp)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    out=root/'root_normalized_pilot_v3'/'seed_42'
    if out.exists():raise FileExistsError(f'Existing pilot preserved: {out}')
    out.mkdir(parents=True)
    protocol={'seed':42,'change':'empty web root path -> /; retain www and nonempty paths',
        'partition_sha256':PARTS,'reserved_holdout_sha256':HOLDOUT_SHA,
        'preflight_report_sha256':sha(audit_path),'reference_model_sha256':sha(ref/'phishguard_gh_model.keras'),
        'script_sha256':sha(__file__),'epochs_max':50,'batch_size':256,'adam_learning_rate':.001,
        'early_stopping_patience':12,'reduce_lr_patience':4,'reduce_lr_factor':.5,'minimum_lr':1e-5,
        'architecture_parameters':244673,'within_partition_same_label_repeats':'retained to keep training membership fixed',
        'evaluation':'original validation only; no performance claim about reserved external holdout',
        'runtime':{'keras':keras.__version__,'numpy':np.__version__}}
    dump(out/'protocol.json',protocol)
    xs={n:np.asarray([encode(u) for u in d.url],dtype=np.float32) for n,d in parts.items()}
    ys={n:d.label.to_numpy(dtype=np.float32) for n,d in parts.items()}
    keras.backend.clear_session();keras.utils.set_random_seed(42)
    model=build(keras)
    tick=time.perf_counter()
    hist=model.fit(xs['train'],ys['train'],validation_data=(xs['validation'],ys['validation']),
        epochs=50,batch_size=256,verbose=1,callbacks=[
         keras.callbacks.EarlyStopping(monitor='val_loss',patience=12,restore_best_weights=True,verbose=1),
         keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=4,min_lr=1e-5,verbose=1),
         keras.callbacks.ModelCheckpoint(str(out/'best_checkpoint.keras'),monitor='val_loss',save_best_only=True,verbose=0),
         keras.callbacks.CSVLogger(str(out/'epoch_history.csv'))])
    seconds=time.perf_counter()-tick
    model.save(out/'model.keras');dump(out/'history.json',hist.history)
    p=model.predict(xs['validation'],batch_size=256,verbose=0).reshape(-1)
    y=ys['validation'];corrected=evaluate(y,p);reference=evaluate(y,saved_val.probability.to_numpy())
    pd.DataFrame({'url':val.url,'true_label':y.astype(int),'probability':p,'pred_label':(p>=.5).astype(int)}).to_csv(out/'validation_predictions.csv',index=False)
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    operating={}
    for cap in [.001,.005,.01]:
        ix=np.where(fpr<=cap)[0];i=ix[np.argmax(tpr[ix])];t=float(thresholds[i])
        operating[str(cap)]={'threshold':t if np.isfinite(t) else None,'positive_infinity':bool(np.isposinf(t)),
            'validation_fpr':float(fpr[i]),'validation_recall':float(tpr[i])}
    result={'seed':42,'experiment':'root_normalized_validation_pilot','partition_sha256':PARTS,
        'reserved_holdout_sha256':HOLDOUT_SHA,'model_sha256':sha(out/'model.keras'),
        'best_epoch':int(np.argmin(hist.history['val_loss']))+1,'epochs_run':len(hist.history['loss']),
        'training_seconds':seconds,'validation':corrected,'reference_validation':reference,
        'validation_metric_differences':{k:corrected[k]-reference[k] for k in ['accuracy','precision','recall','f1_score','auc_roc','average_precision','false_positive_rate']},
        'validation_selected_operating_points':operating,
        'test_evaluated':False,'external_evaluated':False,
        'note':'One-seed development pilot. No proof of generalization or deployment readiness. Root equivalence is imposed by the encoder; www sensitivity can remain.'}
    dump(out/'metrics.json',result)
    (out/'COMPLETE.txt').write_text('Seed-42 pilot completed. External holdout not scored.\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('PILOT COMPLETE:',out)

if __name__=='__main__':main()
