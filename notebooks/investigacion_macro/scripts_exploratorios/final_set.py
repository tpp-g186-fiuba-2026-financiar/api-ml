import os, warnings, sys
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
col=lambda *pre:[c for c in sorted(P.columns) if c.startswith(pre) and c not in ("e_month","e_dow")]
SETS={"ALL (con mercado interno)":[c for c in sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"}) if c not in ("e_month","e_dow")],
      "FINAL: tecnicas+global+argentina+eventos":col("t_","g_","a_","e_")}
def models():
    return {"hgb":HistGradientBoostingClassifier(max_depth=2,learning_rate=0.05,max_iter=150,min_samples_leaf=500,l2_regularization=5.0,random_state=0),
            "et":make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=200,min_samples_leaf=200,max_features=0.3,n_jobs=1,random_state=0)),
            "mlp":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),FunctionTransformer(lambda X:np.clip(X,-5,5)),MLPClassifier(hidden_layer_sizes=(32,),alpha=0.05,early_stopping=True,max_iter=200,random_state=0))}
def run(h,sname,refit=126):
    f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok]; y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[SETS[sname]].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; sc={k:np.full(len(y),np.nan) for k in ("hgb","et","mlp")}
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr)<3000 or len(te)==0: continue
        for name,m in models().items(): sc[name][te]=m.fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    mk=~np.isnan(sc["hgb"]); yy=y[mk]; dts=pd.DatetimeIndex(dates[mk]); mo=dts.to_period("M").astype(str).to_numpy(); out=[]
    ens=np.mean([pd.Series(sc[k][mk]).rank(pct=True).to_numpy() for k in sc],axis=0)
    for name,s in list(((k,sc[k][mk]) for k in sc))+[("ENSAMBLE (promedio de rangos)",ens)]:
        rng=np.random.default_rng(0); um=np.unique(mo); idx={u:np.where(mo==u)[0] for u in um}
        v=[roc_auc_score(yy[ix],s[ix]) for ix in (np.concatenate([idx[u] for u in rng.choice(um,len(um),replace=True)]) for _ in range(120))]
        per=[roc_auc_score(yy[dts.year==yr],s[dts.year==yr]) for yr in range(2020,2027) if len(np.unique(yy[dts.year==yr]))>1]
        out.append(dict(h=h,set=sname,modelo=name,auc=round(roc_auc_score(yy,s),3),lo=round(np.percentile(v,2.5),3),hi=round(np.percentile(v,97.5),3),anios=f"{sum(a>0.5 for a in per)}/{len(per)}"))
    return out
if __name__=="__main__":
    res=Parallel(n_jobs=6)(delayed(run)(h,s) for h in (20,30,40) for s in SETS)
    pd.set_option("display.width",220); print(pd.DataFrame([r for rr in res for r in rr]).to_string(index=False))
