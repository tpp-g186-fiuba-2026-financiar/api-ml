"""Mismas variables, distintas familias de modelo (arboles, SVM, MLP, LSTM)."""
import os, warnings, sys
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
ALL=[c for c in sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"}) if c not in ("e_month","e_dow")]
class Clip:
    def fit(self,X,y=None): return self
    def transform(self,X): return np.clip(X,-5,5)
from sklearn.preprocessing import FunctionTransformer
def pipe(est): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),FunctionTransformer(lambda X:np.clip(X,-5,5)),est)
MODELS={
 "HGB (arboles potenciados)":lambda: HistGradientBoostingClassifier(max_depth=2,learning_rate=0.05,max_iter=150,min_samples_leaf=500,l2_regularization=5.0,random_state=0),
 "Random Forest":lambda: make_pipeline(SimpleImputer(strategy="median"),RandomForestClassifier(n_estimators=200,min_samples_leaf=200,max_features=0.3,n_jobs=1,random_state=0)),
 "Extra Trees":lambda: make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=200,min_samples_leaf=200,max_features=0.3,n_jobs=1,random_state=0)),
 "SVM rbf":lambda: pipe(SVC(C=1.0,gamma="scale",probability=False)),
 "MLP (red neuronal densa)":lambda: pipe(MLPClassifier(hidden_layer_sizes=(32,),alpha=0.05,early_stopping=True,max_iter=200,random_state=0)),
}
def score(m,X): 
    return m.decision_function(X) if hasattr(m,"decision_function") and not hasattr(m,"predict_proba") else (m.predict_proba(X)[:,1] if hasattr(m,"predict_proba") else m.decision_function(X))
def wf(h,name,refit=126):
    f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok]; y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[ALL].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; sc=np.full(len(y),np.nan)
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr)<3000 or len(te)==0: continue
        if name=="SVM rbf": tr=tr[-7000:]
        m=MODELS[name]().fit(X[tr],y[tr])
        sc[te]=m.decision_function(X[te]) if name=="SVM rbf" else m.predict_proba(X[te])[:,1]
    mk=~np.isnan(sc); yy,ss=y[mk],sc[mk]; dts=pd.DatetimeIndex(dates[mk]); mo=dts.to_period("M").astype(str).to_numpy()
    rng=np.random.default_rng(0); um=np.unique(mo); idx={u:np.where(mo==u)[0] for u in um}; v=[roc_auc_score(yy[ix],ss[ix]) for ix in (np.concatenate([idx[u] for u in rng.choice(um,len(um),replace=True)]) for _ in range(120))]
    per=[roc_auc_score(yy[dts.year==yr],ss[dts.year==yr]) for yr in range(2020,2027) if len(np.unique(yy[dts.year==yr]))>1]
    return dict(h=h,modelo=name,auc=round(roc_auc_score(yy,ss),3),lo=round(np.percentile(v,2.5),3),hi=round(np.percentile(v,97.5),3),anios_sobre_050=f"{sum(a>0.5 for a in per)}/{len(per)}")
if __name__=="__main__":
    res=Parallel(n_jobs=9)(delayed(wf)(h,n) for h in (20,30) for n in MODELS)
    pd.set_option("display.width",200); print(pd.DataFrame(res).to_string(index=False))
