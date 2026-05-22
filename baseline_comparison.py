import os, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
import keras
from keras import layers, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,
                              f1_score,roc_auc_score,roc_curve)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, joblib

print("  PhishGuard-GH Baseline Model Comparison")

MAX_LEN=200; VOCAB_SIZE=97; CHAR_TO_INT={chr(i):i-31 for i in range(32,127)}
def url_to_seq(url):
    url=str(url).lower().strip()
    return [CHAR_TO_INT.get(c,96) for c in url[:MAX_LEN]]+[0]*(MAX_LEN-len(url[:MAX_LEN]))
def evaluate(yt,yp,ypr):
    return {"accuracy":accuracy_score(yt,yp),
            "precision":precision_score(yt,yp,zero_division=0),
            "recall":recall_score(yt,yp,zero_division=0),
            "f1":f1_score(yt,yp,zero_division=0),
            "auc":roc_auc_score(yt,ypr)}

print("\nLoading data...")
X_test=np.load("X_test.npy"); y_test=np.load("y_test.npy")
feat=pd.read_csv("phishguard_features.csv"); fcols=[c for c in feat.columns if c!="label"]
Xf=feat[fcols].values; yf=feat["label"].values
Xftr,Xfte,yftr,yfte=train_test_split(Xf,yf,test_size=0.15,random_state=42,stratify=yf)
url_df=pd.read_csv("phishguard_gh_dataset.csv")
Xs=np.array([url_to_seq(u) for u in url_df["url"]],dtype=np.float32); ys=url_df["label"].values
Xstr,Xste,ystr,yste=train_test_split(Xs,ys,test_size=0.15,random_state=42,stratify=ys)
_,Xsv,_,ysv=train_test_split(Xstr,ystr,test_size=0.15/0.85,random_state=42,stratify=ystr)

RESULTS={}; ROC={}
def store(name,yt,yp,ypr):
    RESULTS[name]=evaluate(yt,yp,ypr); fpr,tpr,_=roc_curve(yt,ypr); ROC[name]=(fpr,tpr)
    r=RESULTS[name]
    print(f"  {name:<25} Acc={r['accuracy']*100:.2f}%  F1={r['f1']*100:.2f}%  AUC={r['auc']:.4f}")

ES=callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True,verbose=0)

print("\nTraining Random Forest...")
rf=RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42)
rf.fit(Xftr,yftr); joblib.dump(rf,"model_rf.pkl")
store("Random Forest",yfte,rf.predict(Xfte),rf.predict_proba(Xfte)[:,1])

print("Training XGBoost...")
xgb=XGBClassifier(n_estimators=200,eval_metric="logloss",random_state=42)
xgb.fit(Xftr,yftr,verbose=False); joblib.dump(xgb,"model_xgb.pkl")
store("XGBoost",yfte,xgb.predict(Xfte),xgb.predict_proba(Xfte)[:,1])

print("Training Standalone CNN...")
i1=keras.Input(shape=(MAX_LEN,)); x1=layers.Embedding(VOCAB_SIZE,64)(i1)
x1=layers.Conv1D(128,3,activation="relu",padding="same")(x1)
x1=layers.GlobalMaxPooling1D()(x1); x1=layers.Dense(64,activation="relu")(x1)
o1=layers.Dense(1,activation="sigmoid")(x1); cnn=keras.Model(i1,o1)
cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
cnn.fit(Xstr,ystr,validation_data=(Xsv,ysv),epochs=30,batch_size=256,callbacks=[ES],verbose=0)
pc=cnn.predict(Xste,verbose=0).flatten()
store("Standalone CNN",yste,(pc>=0.5).astype(int),pc)

print("Training Standalone BiLSTM...")
i2=keras.Input(shape=(MAX_LEN,)); x2=layers.Embedding(VOCAB_SIZE,64)(i2)
x2=layers.Bidirectional(layers.LSTM(128))(x2); x2=layers.Dropout(0.4)(x2)
x2=layers.Dense(64,activation="relu")(x2); o2=layers.Dense(1,activation="sigmoid")(x2)
bl=keras.Model(i2,o2); bl.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
bl.fit(Xstr,ystr,validation_data=(Xsv,ysv),epochs=30,batch_size=256,callbacks=[ES],verbose=0)
pb2=bl.predict(Xste,verbose=0).flatten()
store("Standalone BiLSTM",yste,(pb2>=0.5).astype(int),pb2)

print("Loading PhishGuard-GH...")
mm=keras.models.load_model("phishguard_gh_model.keras",compile=False)
ppg=mm.predict(X_test,verbose=0).flatten()
store("PhishGuard-GH",y_test,(ppg>=0.5).astype(int),ppg)


print(f"  {'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")

for name,r in RESULTS.items():
    star=" ★" if name=="PhishGuard-GH" else ""
    print(f"  {name+star:<24} {r['accuracy']*100:>6.2f}% {r['precision']*100:>6.2f}% "
          f"{r['recall']*100:>6.2f}% {r['f1']*100:>6.2f}% {r['auc']:>6.4f}")


names=list(RESULTS.keys()); accs=[RESULTS[n]["accuracy"]*100 for n in names]
f1s=[RESULTS[n]["f1"]*100 for n in names]
colors=["steelblue","steelblue","steelblue","steelblue","#e74c3c"]
fig,axes=plt.subplots(1,2,figsize=(14,5)); x=np.arange(len(names))
for ax,vals,lbl in [(axes[0],accs,"Accuracy (%)"),(axes[1],f1s,"F1-Score (%)")]:
    ax.bar(x,vals,color=colors,edgecolor="white",width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names,rotation=20,ha="right",fontsize=9)
    ax.set_ylabel(lbl); ax.set_ylim(85,100)
    ax.set_title(f"{lbl} Comparison\nPhishGuard-GH",fontsize=12,fontweight="bold")
    ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); plt.savefig("model_comparison.png",dpi=300,bbox_inches="tight"); plt.close()

lc=["#95a5a6","#7f8c8d","#3498db","#2ecc71","#e74c3c"]; lw=[1.5,1.5,1.5,1.5,3.0]
plt.figure(figsize=(8,6))
for i,(name,(fpr,tpr)) in enumerate(ROC.items()):
    plt.plot(fpr,tpr,color=lc[i],lw=lw[i],label=f"{name} (AUC={RESULTS[name]['auc']:.4f})")
plt.plot([0,1],[0,1],"k--",alpha=0.4); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Comparison\nPhishGuard-GH",fontsize=12,fontweight="bold")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("roc_comparison.png",dpi=300,bbox_inches="tight"); plt.close()
print("\n  Saved: model_comparison.png  roc_comparison.png")

