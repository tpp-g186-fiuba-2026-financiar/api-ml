"""Señales alza/baja/neutral con umbrales calibrados como en produccion (85% entrena, 15% final calibra)."""
import os, sys, warnings
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
BASE=["t_ret1","t_ret10","t_rsi14","t_sma20","t_macd","t_boll","t_gk20","t_dd60","t_volz","m_ret5","m_ret20","m_ret60","m_vol20","s_rel20"]
fam=lambda p:[c for c in P.columns if c.startswith(p)]
ALL=sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"})
SETS={"BASE":BASE,"+G global":BASE+fam("g_"),"+G+A":BASE+fam("g_")+fam("a_"),"ALL":ALL}
def run(h,sname,qs=(0.15,0.10),refit=126):
    f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok]; y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[SETS[sname]].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; rows=[]
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr_all=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr_all)<3000 or len(te)==0: continue
        sp=days[int(np.searchsorted(days,cut)*0.85)]; tr=tr_all[dates[tr_all]<=sp]; va=tr_all[dates[tr_all]>sp]
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=120,min_samples_leaf=250,l2_regularization=3.0,random_state=0).fit(X[tr],y[tr])
        pv=m.predict_proba(X[va])[:,1]; pt=m.predict_proba(X[te])[:,1]
        for q in qs:
            hi,lo=np.quantile(pv,1-q),np.quantile(pv,q)
            rows.append(pd.DataFrame({"q":q,"date":dates[te],"sig":np.where(pt>=hi,-1,np.where(pt<=lo,1,0)),"down":y[te]}))
    r=pd.concat(rows); out=[]
    for q,g in r.groupby("q"):
        mo=pd.DatetimeIndex(g.date).to_period("M").astype(str).to_numpy(); rng=np.random.default_rng(0)
        for lab,code,hit in (("baja",-1,g.down==1),("alza",1,g.down==0)):
            mask=(g.sig==code).to_numpy(); hit=hit.to_numpy()
            if mask.sum()<60: continue
            um=np.unique(mo[mask]); idx={u:np.where((mo==u)&mask)[0] for u in um}
            v=[hit[np.concatenate([idx[u] for u in rng.choice(um,len(um),replace=True)])].mean()*100 for _ in range(200)]
            base=(g.down==1).mean() if code==-1 else (g.down==0).mean()
            out.append(dict(h=h,set=sname,q=q,señal=lab,n=int(mask.sum()),cobertura=mask.mean()*100,acierto=hit[mask].mean()*100,lo=np.percentile(v,2.5),hi=np.percentile(v,97.5),base=base*100,meses=len(um)))
    return out
if __name__=="__main__":
    jobs=[(h,s) for h in (14,20,30,40) for s in SETS]
    res=Parallel(n_jobs=9)(delayed(run)(*j) for j in jobs)
    df=pd.DataFrame([r for rr in res for r in rr]); df.to_csv(R+"signals.csv",index=False)
    pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
    for h in (14,20,30,40):
        print(f"\n### horizonte {h} ruedas (umbral: 10% mas seguro de cada lado)")
        print(df[(df.h==h)&(df.q==0.10)].drop(columns=["h","q"]).round(1).to_string(index=False))
