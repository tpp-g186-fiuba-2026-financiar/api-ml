from ev import *
g=G
TM=g["tech"]+g["mkt"]
for c in g["adr"]: P[c]=P[c].fillna(0)
LIQ=["GGAL","BMA","YPFD","PAMP","TXAR","ALUA","CRES","EDN","TECO2","TGSU2","CEPU","COME","LOMA","BYMA","BBAR","SUPV","TGNO4","VALO","MIRG","CVH"]
def lift(o,q,rng=np.random.default_rng(1),nb=200):
    o=o.copy(); yr=o.date.dt.year
    o["thr_hi"]=o.groupby(yr).p.transform(lambda s:s.quantile(1-q)); o["thr_lo"]=o.groupby(yr).p.transform(lambda s:s.quantile(q))
    o["ybase"]=o.groupby(yr).lab.transform("mean")
    o["m"]=o.date.dt.to_period("M")
    res={}
    for side in ("hi","lo"):
        s=o[o.p>=o.thr_hi] if side=="hi" else o[o.p<=o.thr_lo]
        hit=(s.lab==1) if side=="hi" else (s.lab==0)
        base=o.loc[s.index,"ybase"] if side=="hi" else 1-o.loc[s.index,"ybase"]
        d=(hit.astype(float)-base).groupby(s.m).mean()
        boots=[d.sample(len(d),replace=True,random_state=int(rng.integers(1e9))).mean() for _ in range(nb)]
        res[side]=(hit.mean(),base.mean(),d.mean(),np.percentile(boots,2.5),np.percentile(boots,97.5),len(s))
    return res
def rpt(name,o):
    o=o[o.ticker.isin(LIQ)]
    a,l,u=boot_auc(o,n=150)
    out=f"{name:28s} AUC {a:.3f}[{l:.3f},{u:.3f}]"
    for q in (0.1,0.25):
        r=lift(o,q)
        out+=f" | q{int(q*100)} ALZA {r['hi'][0]:.3f} vs {r['hi'][1]:.3f} ({r['hi'][2]*100:+.1f}pp [{r['hi'][3]*100:+.1f},{r['hi'][4]*100:+.1f}]) BAJA {r['lo'][0]:.3f} vs {r['lo'][1]:.3f} ({r['lo'][2]*100:+.1f}pp [{r['lo'][3]*100:+.1f},{r['lo'][4]*100:+.1f}])"
    print(out,flush=True)
for h in (5,20):
    print("== absoluto, evaluado en liquidas, h",h)
    rpt("tech+mkt",run(TM,f"y{h}"))
    rpt("tech+mkt+adr(0)",run(TM+g["adr"],f"y{h}"))
    rpt("tech+mkt, entrena solo liquidas",run(TM,f"y{h}"))
