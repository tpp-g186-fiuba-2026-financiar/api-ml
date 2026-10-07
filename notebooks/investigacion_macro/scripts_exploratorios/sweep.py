"""Barrido: que familias de variables ayudan a detectar caidas (y subas), por horizonte."""
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SP='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(SP+"panel.pkl")
BASE=["t_ret1","t_ret10","t_rsi14","t_sma20","t_macd","t_boll","t_gk20","t_dd60","t_volz","m_ret5","m_ret20","m_ret60","m_vol20","s_rel20"]
fam=lambda p:[c for c in P.columns if c.startswith(p)]
SETS={"BASE":BASE,"+T tecnicas":BASE+fam("t_"),"+M mercado/amplitud":BASE+fam("m_")+fam("s_"),"+G global":BASE+fam("g_"),
      "+A argentina":BASE+fam("a_"),"+E eventos":BASE+fam("e_"),"ALL":[c for c in P.columns if c.split("_")[0] in "tmsgae" and c[1]=="_" ]}
SETS["ALL"]=sorted(set(SETS["ALL"]))
def targets(h):
    f,z=P[f"fwd{h}"],P[f"zfwd{h}"]; ok=f.notna()&(f!=0)
    return ok,(f<0).astype(int),(z< -1.5).astype(int),(f>0).astype(int)
def run(h,tname,sname,refit=126):
    ok,down,severe,up=targets(h); y={"down":down,"severe":severe,"up":up}[tname]
    D=P[ok]; yy=y[ok].to_numpy(); X=D[SETS[sname]].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; sc=np.full(len(yy),np.nan)
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr)<3000 or len(te)==0: continue
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=120,min_samples_leaf=250,l2_regularization=3.0,random_state=0).fit(X[tr],yy[tr])
        sc[te]=m.predict_proba(X[te])[:,1]
    out=[]; months=pd.DatetimeIndex(dates).to_period("M").astype(str).to_numpy()
    for wname,lo,hi in (("dev 2020-22","2020-01-01","2022-12-31"),("confirm 2023-26","2023-01-01","2026-12-31")):
        m=(~np.isnan(sc))&(dates>=np.datetime64(lo))&(dates<=np.datetime64(hi))
        if m.sum()<1000 or len(np.unique(yy[m]))<2: continue
        s_,y_,mo=sc[m],yy[m],months[m]; q=np.quantile(s_,0.9); top=s_>=q; base=y_.mean()
        auc=roc_auc_score(y_,s_)
        rng=np.random.default_rng(0); um=np.unique(mo); idx={u:np.where(mo==u)[0] for u in um}; aucs=[];lifts=[]
        for _ in range(150):
            ix=np.concatenate([idx[u] for u in rng.choice(um,len(um),replace=True)])
            if len(np.unique(y_[ix]))>1:
                aucs.append(roc_auc_score(y_[ix],s_[ix])); t_=top[ix]; lifts.append(y_[ix][t_].mean()-y_[ix].mean() if t_.any() else np.nan)
        out.append(dict(h=h,target=tname,set=sname,window=wname,n=int(m.sum()),base=base*100,auc=auc,auc_lo=np.percentile(aucs,2.5),auc_hi=np.percentile(aucs,97.5),
                        top10=y_[top].mean()*100,lift_pp=(y_[top].mean()-base)*100,lift_lo=np.nanpercentile(lifts,2.5)*100,lift_hi=np.nanpercentile(lifts,97.5)*100))
    return out
if __name__=="__main__":
    jobs=[(h,t,s) for h in (1,5,10,20) for t in ("down","severe","up") for s in SETS]
    print(len(jobs),"corridas",flush=True); t0=time.time()
    res=Parallel(n_jobs=9)(delayed(run)(*j) for j in jobs)
    df=pd.DataFrame([r for rr in res for r in rr]); df.to_csv(SP+"sweep1.csv",index=False); print("listo en",round(time.time()-t0),"s")
