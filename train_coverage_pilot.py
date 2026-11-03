"""One seed-42 mixed-corpus pilot. No internal-test or reserved-holdout scoring.
Run --check-only before --seed 42. Existing outputs are never overwritten.
Requires the same installed packages as the v3 experiments.
"""
import argparse, hashlib, json, os, re, time
from pathlib import Path
from urllib.parse import urlsplit
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
BASE={'length':200,'embedding':64,'kernels':[3,5]}
PARTS={'train': '28cdda9a23f19c50d269132e44713d934c0b8b80ec8904b6dcbb2e3230b2703f', 'validation': '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e', 'test': '22772ff2322b29fa4acb87c537095f1a956c063e0109000e8ebf0716e711e14f'}
FILES={'train.csv': '5768ed99d8f785dcaa376284952720dcf74aef86642c7015d933d6fdb42b54bc', 'validation_v3.csv': '8fff2eec428c94bb099ad5299e2314b229979c19e13b9232b04bc77ef083908e', 'validation_phiusiil_development.csv': 'a328627ca4707b4f2edda9750a53b901a0409b742224cd2584d174c8599ce680', 'excluded_phiusiil.csv': '76bea42af818fe2a1ea40a998460f2101a743e50045d162f28d06fe4ba39bc85'}
HOLDOUT_SHA='b8a069c5d9671e74ad4c0f0b29ef1b3a40bd7c0c58c02d6884ae4e32bceeded8'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def normalize(url):
    u=str(url).strip()
    if not u or any(c.isspace() for c in u) or any(c in u for c in "'\"\\"):return u
    m=re.match(r'^https?://',u,re.I)
    if m:prefix,rest=u[:m.end()],u[m.end():]
    elif u.startswith('//'):prefix,rest='//',u[2:]
    elif '://' in u:return u
    else:prefix,rest='',u
    try:
        p=urlsplit('https://'+rest)
        if not p.hostname or p.path!='':return u
        _=p.port
        return prefix+p.netloc+'/'+rest[len(p.netloc):]
    except ValueError:return u
def build(config,keras):
 from keras import layers,regularizers
 inp=keras.Input(shape=(config['length'],),name='url_input')
 emb=layers.Embedding(97,config['embedding'],name='char_embedding')(inp)
 h=layers.SpatialDropout1D(.2,name='embed_dropout')(emb)
 branches=[layers.Conv1D(128,k,activation='relu',padding='same',kernel_regularizer=regularizers.l2(1e-4),name=f'conv_k{k}')(h) for k in config['kernels']]
 h=layers.Concatenate(name='merge_cnn')(branches) if len(branches)>1 else branches[0]
 h=layers.MaxPooling1D(2,name='maxpool')(h)
 h=layers.Bidirectional(layers.LSTM(64,kernel_regularizer=regularizers.l2(1e-4),name='lstm'),name='bilstm')(h)
 h=layers.Dropout(.5,name='dropout')(h)
 h=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-4),name='dense64')(h)
 h=layers.Dropout(.3,name='dropout2')(h)
 output=layers.Dense(1,activation='sigmoid',name='output')(h)
 model=keras.Model(inp,output,name='PhishGuard_GH')
 model.compile(optimizer=keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=['accuracy'])
 return model

def encode(url):
    text=normalize(url)
    for prefix in ('https://','http://'):
        if text.lower().startswith(prefix):text=text[len(prefix):];break
    seq=[ord(c)-31 if 32<=ord(c)<=126 else 96 for c in text.lower().strip()[:200]]
    return seq+[0]*(200-len(seq))

def config_normal(value):
    if isinstance(value,dict):return {k:config_normal(v) for k,v in value.items() if k!='shared_object_id'}
    if isinstance(value,(tuple,list)):return [config_normal(v) for v in value]
    return value

def signature(model):
    return [(l.__class__.__name__,config_normal(l.get_config())) for l in model.layers]

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def mean_bce(y,p):
    import numpy as np
    p=np.clip(np.asarray(p,dtype=np.float64).reshape(-1),1e-7,1-1e-7)
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    if len(y)!=len(p) or not np.isfinite(p).all():raise ValueError('Invalid validation probabilities')
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))

def metrics(y,p):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
    pred=p>=.5
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return dict(threshold=.5,accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),auc_roc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),false_positive_rate=fp/(fp+tn),tn=tn,fp=fp,fn=fn,tp=tp,binary_crossentropy=mean_bce(y,p))

def self_test():
    import numpy as np
    assert encode('http://example.com')==encode('https://example.com/')
    assert encode('www.example.com')!=encode('example.com')
    assert normalize('https://example.com/a')=='https://example.com/a'
    assert normalize("'example.com'")=="'example.com'"
    assert len(encode('x'*400))==200
    assert abs(mean_bce([0,1],[.5,.5])-np.log(2))<1e-12
    assert mean_bce([0,1],[0,1])<1e-6
    print('SELF-TEST PASSED: encoder and validation-loss arithmetic.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--seed',type=int,choices=[42])
    mode.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    import numpy as np,pandas as pd,keras
    folder=ROOT/'coverage_development_v1'
    report=json.loads((folder/'preparation_report.json').read_text())
    if report['experiment']!='coverage_development_v1' or report['reserved_holdout_scored']:
        raise ValueError('Unexpected preparation protocol')
    if report['partition_sha256']!=PARTS or report['output_sha256']!=FILES:
        raise ValueError('Preparation fingerprints differ from reviewed report')
    for name,h in FILES.items():
        if sha(folder/name)!=h:raise ValueError('Prepared file changed: '+name)
    for name,h in PARTS.items():
        if sha(ROOT/'dataset_fix_outputs_v3'/f'{name}.csv')!=h:raise ValueError('Frozen partition changed: '+name)
    # Identity check only: never parse or score reserved holdout rows.
    if sha(ROOT/'reserved_mendeley_holdout'/'holdout.csv')!=HOLDOUT_SHA:
        raise ValueError('Reserved holdout identity changed')
    train=pd.read_csv(folder/'train.csv',keep_default_na=False)
    val3=pd.read_csv(folder/'validation_v3.csv',keep_default_na=False)
    valphi=pd.read_csv(folder/'validation_phiusiil_development.csv',keep_default_na=False)
    original_train=pd.read_csv(ROOT/'dataset_fix_outputs_v3'/'train.csv',keep_default_na=False)
    if [len(train),len(val3),len(valphi)]!=[51664,7668,4000]:raise ValueError('Unexpected sizes')
    if not train.iloc[:35664][['url','label']].reset_index(drop=True).equals(original_train[['url','label']]):
        raise ValueError('Original training membership/order differs')
    if train.iloc[35664:].label.value_counts().to_dict()!={0:8000,1:8000} or valphi.label.value_counts().to_dict()!={0:2000,1:2000}:
        raise ValueError('Unexpected added class counts')
    frames={'train':train,'validation_v3':val3,'validation_phiusiil':valphi}
    xs={};ys={};keys={}
    for name,frame in frames.items():
        if not frame.label.isin([0,1]).all():raise ValueError('Invalid labels: '+name)
        xs[name]=np.array([encode(u) for u in frame.url],dtype=np.float32)
        ys[name]=frame.label.to_numpy(dtype=np.float32)
        labels={}
        for row,y in zip(xs[name],ys[name]):
            k=row.astype(np.uint8).tobytes()
            if k in labels and labels[k]!=y:raise ValueError('Contradictory encoded labels: '+name)
            labels[k]=y
        keys[name]=set(labels)
        print(f'{name}: {len(frame):,} rows; encoding/labels passed',flush=True)
    names=list(keys)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if keys[a]&keys[b]:raise ValueError(f'Encoded overlap: {a}/{b}')
    keras.backend.clear_session();keras.utils.set_random_seed(42)
    model=build(BASE,keras)
    reference=keras.models.load_model(ROOT/'corrected_results_v3'/'seed_42'/'phishguard_gh_model.keras',compile=False)
    if model.count_params()!=244673 or signature(model)!=signature(reference):
        raise ValueError('Reference architecture mismatch; no training started')
    del reference,model
    keras.backend.clear_session()
    print('CHECK PASSED: reviewed mixed-corpus files, encoder and reference architecture.',flush=True)
    if args.check_only:
        print('No training, predictions or output directories created.');return
    out=ROOT/'coverage_pilot_v1'/'seed_42'
    out.mkdir(parents=True,exist_ok=False)
    protocol={
        'seed':42,'training_rows':len(train),'partition_sha256':PARTS,'prepared_files_sha256':FILES,
        'script_sha256':sha(Path(__file__)),
        'model_selection':'0.5 * original-v3 validation mean BCE + 0.5 * PhiUSIIL-development validation mean BCE; excludes L2 penalties',
        'training':'all 51,664 rows equally weighted; Adam 0.001; batch 256; maximum 50 epochs; patience 12; ReduceLR factor 0.5 patience 4 minimum 1e-5',
        'phiusiil_role':'Previously examined corpus now development data; not independent test evidence',
        'encoding':'audited empty-root slash normalization, scheme stripped, lowercase, first 200 characters, ASCII 1..95/unknown96/right-padding0',
        'reserved_holdout_sha256':HOLDOUT_SHA,'reserved_holdout_scored':False,'internal_test_scored':False}
    write_json(out/'protocol.json',protocol)
    keras.utils.set_random_seed(42)
    model=build(BASE,keras)
    class SourceValidation(keras.callbacks.Callback):
        def __init__(self):super().__init__();self.rows=[]
        def on_epoch_end(self,epoch,logs=None):
            if logs is None:raise RuntimeError('Missing callback logs')
            losses={}
            for name in ('validation_v3','validation_phiusiil'):
                p=self.model.predict(xs[name],batch_size=256,verbose=0).reshape(-1)
                losses[name]=mean_bce(ys[name],p)
            balanced=.5*(losses['validation_v3']+losses['validation_phiusiil'])
            logs['val_source_balanced_bce']=balanced
            logs['val_v3_bce']=losses['validation_v3']
            logs['val_phiusiil_bce']=losses['validation_phiusiil']
            self.rows.append({'epoch':epoch+1,**{k:float(v) for k,v in logs.items()}})
            write_json(out/'epoch_history.json',self.rows)
            print(f"\nValidation BCE: v3={losses['validation_v3']:.5f}; Phi-development={losses['validation_phiusiil']:.5f}; balanced={balanced:.5f}",flush=True)
    validation=SourceValidation()
    monitor='val_source_balanced_bce'
    cb=[validation,
        keras.callbacks.EarlyStopping(monitor=monitor,mode='min',patience=12,restore_best_weights=True,verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor,mode='min',factor=.5,patience=4,min_lr=1e-5,verbose=1),
        keras.callbacks.ModelCheckpoint(str(out/'best_checkpoint.keras'),monitor=monitor,mode='min',save_best_only=True),
        keras.callbacks.CSVLogger(str(out/'epochs.csv'))]
    started=time.perf_counter()
    model.fit(xs['train'],ys['train'],epochs=50,batch_size=256,shuffle=True,callbacks=cb,verbose=1)
    elapsed=time.perf_counter()-started
    # Load the selected checkpoint explicitly; never evaluate last-epoch weights by accident.
    model=keras.models.load_model(out/'best_checkpoint.keras',compile=False)
    model.save(out/'model.keras')
    result={**protocol,'training_seconds':elapsed,'epochs_run':len(validation.rows),
            'best_epoch':min(validation.rows,key=lambda row:row[monitor])['epoch'],
            'model_sha256':sha(out/'model.keras'),'validation':{}}
    for name,frame in [('validation_v3',val3),('validation_phiusiil',valphi)]:
        p=model.predict(xs[name],batch_size=256,verbose=0).reshape(-1)
        result['validation'][name]=metrics(ys[name],p)
        pd.DataFrame({'url':frame.url,'true_label':ys[name].astype(int),'probability':p,'pred_label':(p>=.5).astype(int)}).to_csv(out/f'{name}_predictions.csv',index=False)
    reference_predictions=pd.read_csv(ROOT/'corrected_results_v3'/'seed_42'/'validation_predictions.csv',keep_default_na=False)
    if reference_predictions.url.tolist()!=val3.url.tolist() or not np.array_equal(reference_predictions.true_label,ys['validation_v3']):
        raise ValueError('Original-model validation alignment differs')
    result['original_model_validation_v3']=metrics(ys['validation_v3'],reference_predictions.probability.to_numpy())
    result['validation_v3_metric_differences']={k:result['validation']['validation_v3'][k]-result['original_model_validation_v3'][k] for k in ('accuracy','precision','recall','f1','auc_roc','average_precision','false_positive_rate')}
    result['limitations']=['One exploratory seed only.','PhiUSIIL-development validation is already examined development data.','Inherited corpus labels and source-format shortcuts remain limitations.','No internal-test or reserved-holdout predictions were made.']
    write_json(out/'metrics.json',result)
    (out/'COMPLETE.txt').write_text('Pilot completed. Review metrics.json before any additional training or holdout evaluation.\n')
    print(json.dumps(result['validation'],indent=2))
    print('COMPLETE:',out)
    print('No test or reserved-holdout scoring. Review this pilot before running more seeds.')

if __name__=='__main__':main()
