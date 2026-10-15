exec(open("e7.py").read().split('for h in (5,20):')[0])
import ev
P=pd.read_pickle("panel2.pkl"); ev.P=P
for c in g["adr"]: P[c]=P[c].fillna(0)
from sklearn.inspection import permutation_importance
def fit_gen(Pm,X,lab,yz,yr,h,mode,w_hl=None,cols=None):
    cut=pd.Timestamp(f"{yr}-01-01")-pd.Timedelta(days=int(h*1.6)+2)
    tr=(Pm.date<cut)&lab.notna()&yz.notna(); te=Pm.date.dt.year==yr
    if tr.sum()<3000: return None
    Xtr=X[tr] if cols is None else X.loc[tr,cols]; Xte=X[te] if cols is None else X.loc[te,cols]
    w=None
    if w_hl: w=0.5**((cut-Pm.loc[tr,"date"]).dt.days/(365*w_hl)).values
    if mode=="base":
        m=HistGradientBoostingClassifier(**PARAMS).fit(Xtr,lab[tr],sample_weight=w); p=m.predict_proba(Xte)[:,1]
    elif mode=="bag":
        ps=[]
        for s in range(6):
            m=HistGradientBoostingClassifier(**{**PARAMS,"random_state":s,"max_features":0.5,"max_depth":3+(s%2)}).fit(Xtr,lab[tr],sample_weight=w); ps.append(m.predict_proba(Xte)[:,1])
        p=np.mean(ps,0)
    elif mode=="zreg":
        m=HistGradientBoostingRegressor(**PARAMS).fit(Xtr,yz[tr],sample_weight=w); p=m.predict(Xte)
    elif mode=="zbag":
        ps=[]
        for s in range(6):
            m=HistGradientBoostingRegressor(**{**PARAMS,"random_state":s,"max_features":0.5}).fit(Xtr,yz[tr],sample_weight=w); ps.append(m.predict(Xte))
        p=np.mean(ps,0)
    out=Pm.loc[te,["date","ticker"]].copy(); out["p"]=p; return out
def run2(cols,h,mode,w_hl=None,sel=None):
    t=f"y{h}"; lab=(P[t]>0).astype(float).where(P[t].notna())
    yz=(P[t]/(P.vol20*np.sqrt(h))).clip(-3,3)
    X=P[cols]
    outs=Parallel(n_jobs=10)(delayed(fit_gen)(P,X,lab,yz,yr,h,mode,w_hl,sel) for yr in YEARS)
    o=pd.concat([x for x in outs if x is not None]); o["yy"]=P[t].reindex(o.index); o["lab"]=lab.reindex(o.index); return o.dropna(subset=["lab"])
def sc(name,o):
    rpt(name,o)
PROD=["r1","rng","rsi14","dma20","r10","volu_z"]
BEST=TM+g["adr"]+["peer_r1","peer_r5","peer_r20","peer_gap1","peer_gap5","peer_gap20"]
def rpt_recent(name,o,yr0):
    o=o[o.date.dt.year>=yr0]; rpt(name+f" (>= {yr0})",o)
def xs_daily(name,o,q=0.2):
    o=o[o.ticker.isin(LIQ)].copy(); o["r"]=o.groupby("date").p.rank(pct=True); o["n"]=o.groupby("date").p.transform("count")
    o=o[o.n>=8]; yb=o.groupby(o.date.dt.year).lab.transform("mean")
    for side in ("ALZA","BAJA"):
        s=o[o.r>1-q] if side=="ALZA" else o[o.r<=q]
        hit=(s.lab==1) if side=="ALZA" else (s.lab==0); base=yb[s.index] if side=="ALZA" else 1-yb[s.index]
        d=(hit.astype(float)-base).groupby(s.date.dt.to_period("M")).mean()
        t=d.mean()/(d.std()/np.sqrt(len(d)))
        print(f"   {name} xs-diario q{int(q*100)} {side}: {hit.mean():.3f} vs base {base.mean():.3f} ({d.mean()*100:+.1f}pp, t_mes {t:+.1f}, n={len(s)})")
for h in (5,20):
    print("== produccion-like vs mejor, h",h)
    op=run2(PROD,h,"base"); ob=run2(BEST,h,"bag",w_hl=3)
    rpt("prod-like (6 feats, base)",op); rpt("MEJOR (mkt+adr+peer, bag, decay)",ob)
    rpt_recent("MEJOR",ob,2024); rpt_recent("prod-like",op,2024)
    xs_daily("MEJOR",ob); xs_daily("prod-like",op)
    ob.to_pickle(f"best_h{h}.pkl")
