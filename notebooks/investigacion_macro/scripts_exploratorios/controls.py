"""Controles: panel con retrasos conservadores, placebo y prueba pseudo-prospectiva."""
import os, sys, warnings
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
BASE=["t_ret1","t_ret10","t_rsi14","t_sma20","t_macd","t_boll","t_gk20","t_dd60","t_volz","m_ret5","m_ret20","m_ret60","m_vol20","s_rel20"]
fam=lambda p:[c for c in P.columns if c.startswith(p)]
ALL=sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"})
SETS={"BASE":BASE,"+G global":BASE+fam("g_"),"+A argentina":BASE+fam("a_"),"ALL":ALL}
MACRO=fam("g_")+fam("a_")+fam("e_")
def hgb(seed=0): return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=120,min_samples_leaf=250,l2_regularization=3.0,random_state=seed)
def data(h,placebo=False,seed=0):
    f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok].copy()
    if placebo:   # rompe la alineacion temporal de las variables macro (desfasa por fecha, mantiene su distribucion)
        rng=np.random.default_rng(seed); days=np.unique(D.index.values); perm=pd.Series(rng.permutation(len(days)),index=days)
        by=D[MACRO].groupby(level=0).first(); shuffled=by.iloc[perm.values].set_axis(by.index)
        D[MACRO]=shuffled.reindex(D.index).to_numpy()
    return D
def wf(h,sname,placebo=False,seed=0,refit=126):
    D=data(h,placebo,seed); y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[SETS[sname]].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; sc=np.full(len(y),np.nan)
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr)<3000 or len(te)==0: continue
        sc[te]=hgb(seed).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(sc); dts=pd.DatetimeIndex(dates[m]); yy,ss=y[m],sc[m]; mo=dts.to_period("M").astype(str).to_numpy()
    rng=np.random.default_rng(0); um=np.unique(mo); idx={u:np.where(mo==u)[0] for u in um}; v=[roc_auc_score(yy[ix],ss[ix]) for ix in (np.concatenate([idx[u] for u in rng.choice(um,len(um),replace=True)]) for _ in range(150))]
    q90,q10=np.quantile(ss,[.9,.1])
    return dict(h=h,set=sname,placebo=placebo,seed=seed,n=int(m.sum()),base=yy.mean()*100,auc=roc_auc_score(yy,ss),lo=np.percentile(v,2.5),hi=np.percentile(v,97.5),top10=yy[ss>=q90].mean()*100,bottom10=yy[ss<=q10].mean()*100)
def prospective(h,sname):
    D=data(h); y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[SETS[sname]].to_numpy(float); dates=D.index.values
    tr=np.where(dates<=np.datetime64("2024-12-31")-np.timedelta64(h*2,"D"))[0]; te=np.where(dates>=np.datetime64("2025-01-01"))[0]
    m=hgb().fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]; ptr=m.predict_proba(X[tr])[:,1]
    hi_thr=np.quantile(ptr,0.9); lo_thr=np.quantile(ptr,0.1)   # umbrales fijados con el pasado
    yt=y[te]; out=dict(h=h,set=sname,n=len(te),base_down=yt.mean()*100,auc=roc_auc_score(yt,p))
    for lab,mask in (("baja(P>=umbral90)",p>=hi_thr),("alza(P<=umbral10)",p<=lo_thr)):
        out[lab+"_n"]=int(mask.sum())
        out[lab+"_acierto"]=((yt[mask]==1).mean()*100 if lab.startswith("baja") else (yt[mask]==0).mean()*100) if mask.sum()>20 else np.nan
    return out
if __name__=="__main__":
    jobs=[(h,s) for h in (14,20,30,40) for s in ("BASE","+G global","+A argentina","ALL")]
    res=Parallel(n_jobs=9)(delayed(wf)(*j) for j in jobs); a=pd.DataFrame(res)
    pl=Parallel(n_jobs=9)(delayed(wf)(h,"ALL",True,sd) for h in (20,40) for sd in (1,2,3,4)); b=pd.DataFrame(pl)
    pr=[prospective(h,s) for h in (14,20,30,40) for s in ("BASE","+G global","ALL")]; c=pd.DataFrame(pr)
    pd.set_option("display.width",250)
    print("### 1) Con retrasos conservadores"); print(a.round(3).to_string(index=False))
    print("\n### 2) PLACEBO (macro desordenada en el tiempo): AUC deberia ser ~0.50"); print(b.round(3).to_string(index=False))
    print("\n### 3) PSEUDO-PROSPECTIVO: entrena hasta 2024, prueba 2025-26 sin reentrenar"); print(c.round(3).to_string(index=False))
