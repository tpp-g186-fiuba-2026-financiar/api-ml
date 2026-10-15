import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
P=pd.read_pickle("panel.pkl")
R1=P.pivot(index="date",columns="ticker",values="r1"); R5=P.pivot(index="date",columns="ticker",values="r5"); R20=P.pivot(index="date",columns="ticker",values="r20")
# vecinos por correlacion usando solo datos hasta el 31/12 del anio previo (sin lookahead)
peer={k:[] for k in ("peer_r1","peer_r5","peer_r20")}
parts=[]
for yr in range(2013,2027):
    est=R1[(R1.index<f"{yr}-01-01")&(R1.index>=f"{yr-2}-01-01")]
    C=est.corr(min_periods=200).fillna(0); np.fill_diagonal(C.values,0)
    cur=R1.index.year==yr
    for t in R1.columns:
        top=C[t].nlargest(5); top=top[top>0.2]
        idx=R1.index[cur]
        if len(top)==0:
            df=pd.DataFrame({"date":idx,"ticker":t,"peer_r1":np.nan,"peer_r5":np.nan,"peer_r20":np.nan})
        else:
            w=top/top.sum()
            df=pd.DataFrame({"date":idx,"ticker":t,"peer_r1":(R1.loc[cur,top.index]*w).sum(axis=1,min_count=1).values,
                             "peer_r5":(R5.loc[cur,top.index]*w).sum(axis=1,min_count=1).values,"peer_r20":(R20.loc[cur,top.index]*w).sum(axis=1,min_count=1).values})
        parts.append(df)
PE=pd.concat(parts)
P=P.merge(PE,on=["date","ticker"],how="left")
P["peer_gap1"]=P.peer_r1-P.r1; P["peer_gap5"]=P.peer_r5-P.r5; P["peer_gap20"]=P.peer_r20-P.r20
# dispersión y rankings cross-seccionales
d=P.groupby("date")
P["disp20"]=d.r20.transform("std"); P["disp5"]=d.r5.transform("std")
P["mkt_up1"]=d.r1.transform(lambda s:(s>0).mean())
for c in ("r5","r20","vol20","volu_z","dd252"): P[f"x_{c}"]=d[c].rank(pct=True)
# residual vs mercado
P["res20"]=P.r20-P.beta60*P.mkt_r20; P["res5"]=P.r5-P.beta60*P.mkt_r5
P["idio_vol"]=P.vol20  # placeholder (vol total)
P.to_pickle("panel2.pkl"); print(P.shape, P.peer_r5.notna().mean())
