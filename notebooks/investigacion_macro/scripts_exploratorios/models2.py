"""Compara clases de modelo (HGB prof.2/3, logistica, ensamble) y mide calibracion por probabilidad."""
import os, warnings
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
BASE=["t_ret1","t_ret10","t_rsi14","t_sma20","t_macd","t_boll","t_gk20","t_dd60","t_volz","m_ret5","m_ret20","m_ret60","m_vol20","s_rel20"]
fam=lambda p:[c for c in P.columns if c.startswith(p)]
ALL=sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"})
SETS={"+G+A":BASE+fam("g_")+fam("a_"),"ALL":ALL}
def mk(kind):
    if kind=="hgb3": return HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=120,min_samples_leaf=250,l2_regularization=3.0,random_state=0)
    if kind=="hgb2": return HistGradientBoostingClassifier(max_depth=2,learning_rate=0.05,max_iter=150,min_samples_leaf=500,l2_regularization=5.0,random_state=0)
    if kind=="logit": return make_pipeline(SimpleImputer(strategy="median"),QuantileTransformer(n_quantiles=200,output_distribution="normal"),LogisticRegression(C=0.02,max_iter=500))
def run(h,sname,kind,refit=126):
    f=P[f"fwd{h}"]; ok=f.notna()&(f!=0); D=P[ok]; y=(D[f"fwd{h}"]<0).astype(int).to_numpy(); X=D[SETS[sname]].to_numpy(float); dates=D.index.values
    days=np.unique(dates); starts=days[days>=np.datetime64("2020-01-01")][::refit]; sc=np.full(len(y),np.nan)
    kinds=["hgb3","logit"] if kind=="ens" else [kind]
    for k,s in enumerate(starts):
        e=starts[k+1] if k+1<len(starts) else None
        cut=days[max(np.searchsorted(days,s)-h,0)]; tr=np.where(dates<=cut)[0]; te=np.where((dates>=s)&((dates<e) if e is not None else True))[0]
        if len(tr)<3000 or len(te)==0: continue
        sc[te]=np.mean([mk(kd).fit(X[tr],y[tr]).predict_proba(X[te])[:,1] for kd in kinds],axis=0)
    m=~np.isnan(sc); yy,ss=y[m],sc[m]; dts=pd.DatetimeIndex(dates[m]); mo=dts.to_period("M").astype(str).to_numpy()
    rng=np.random.default_rng(0); um=np.unique(mo); idx={u:np.where(mo==u)[0] for u in um}; v=[roc_auc_score(yy[ix],ss[ix]) for ix in (np.concatenate([idx[u] for u in rng.choice(um,len(um),replace=True)]) for _ in range(150))]
    per=[roc_auc_score(yy[dts.year==yr],ss[dts.year==yr]) for yr in range(2020,2027) if len(np.unique(yy[dts.year==yr]))>1]
    # calibracion por probabilidad absoluta
    cal={}
    for lo,hi in ((0,0.2),(0.2,0.35),(0.35,0.5),(0.5,0.65),(0.65,1.01)):
        mm=(ss>=lo)&(ss<hi); cal[f"P[{lo:.2f},{min(hi,1):.2f})"]=f"{mm.mean()*100:4.1f}% de dias -> real {yy[mm].mean()*100:5.1f}%" if mm.sum()>50 else "n<50"
    return dict(h=h,set=sname,modelo=kind,auc=round(roc_auc_score(yy,ss),3),lo=round(np.percentile(v,2.5),3),hi=round(np.percentile(v,97.5),3),peor_anio=round(min(per),3),anios_sobre_050=f"{sum(a>0.5 for a in per)}/{len(per)}",**cal)
if __name__=="__main__":
    jobs=[(h,s,k) for h in (20,30,40) for s in SETS for k in ("hgb3","hgb2","logit","ens")]
    res=Parallel(n_jobs=9)(delayed(run)(*j) for j in jobs); df=pd.DataFrame(res); df.to_csv(R+"models2.csv",index=False)
    pd.set_option("display.width",300); pd.set_option("display.max_columns",30); pd.set_option("display.max_colwidth",40)
    print(df.drop(columns=[c for c in df.columns if c.startswith("P[")]).to_string(index=False))
    print("\nCalibracion (ensamble, ALL):"); print(df[(df.modelo=="ens")&(df.set=="ALL")][["h"]+[c for c in df.columns if c.startswith("P[")]].to_string(index=False))
