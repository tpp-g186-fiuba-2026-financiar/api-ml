import os, warnings
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
ALL=sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"})
def hgb(): return HistGradientBoostingClassifier(max_depth=2,learning_rate=0.05,max_iter=150,min_samples_leaf=500,l2_regularization=5.0,random_state=0)
def wf_auc(h,feats,refit=126):
    f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok]; y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[feats].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; sc=np.full(len(y),np.nan)
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr)<3000 or len(te)==0: continue
        sc[te]=hgb().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(sc); return roc_auc_score(y[m],sc[m])
fams={"t_":"tecnicas extra","m_":"mercado interno/amplitud","s_":"accion vs mercado","g_":"mercado global","a_":"macro argentina","e_":"eventos/calendario"}
def ablate(h):
    out={"h":h,"ALL":round(wf_auc(h,ALL),3)}
    for p,n in fams.items(): out["sin "+n]=round(wf_auc(h,[c for c in ALL if not c.startswith(p)]),3)
    return out
def only(h):
    out={"h":h}
    for p,n in fams.items(): out["solo "+n]=round(wf_auc(h,[c for c in ALL if c.startswith(p)]),3)
    return out
if __name__=="__main__":
    pd.set_option("display.width",300); pd.set_option("display.max_columns",30)
    r1=Parallel(n_jobs=4)(delayed(ablate)(h) for h in (20,30,40)); print("### AUC quitando una familia (si baja mucho, esa familia aporta)\n",pd.DataFrame(r1).to_string(index=False))
    r2=Parallel(n_jobs=4)(delayed(only)(h) for h in (20,30,40)); print("\n### AUC usando SOLO una familia\n",pd.DataFrame(r2).to_string(index=False))
    # importancia por permutacion: entrena con <=2022 y mide en 2023+
    for h in (30,):
        f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok]; y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[ALL].to_numpy(float); d=D.index.values
        tr=np.where(d<=np.datetime64("2022-12-31")-np.timedelta64(2*h,"D"))[0]; te=np.where(d>=np.datetime64("2023-01-01"))[0]
        m=hgb().fit(X[tr],y[tr]); pi=permutation_importance(m,X[te],y[te],scoring="roc_auc",n_repeats=5,random_state=0,n_jobs=1)
        imp=pd.Series(pi.importances_mean,index=ALL).sort_values(ascending=False)
        print(f"\n### Variables mas importantes (h={h}, entrenado hasta 2022, medido 2023-26; caida de AUC al desordenarla)\n{imp.head(22).round(4).to_string()}")
