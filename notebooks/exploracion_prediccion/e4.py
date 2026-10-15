from ev import *
from scipy.stats import spearmanr, norm
g=G
def fit_reg(Pm,X,yv,yr,h,params,window=None):
    cut=pd.Timestamp(f"{yr}-01-01")-pd.Timedelta(days=int(h*1.6)+2)
    tr=(Pm.date<cut)&yv.notna()
    if window: tr&=Pm.date>=cut-pd.Timedelta(days=int(window*365))
    te=Pm.date.dt.year==yr
    if tr.sum()<3000 or te.sum()==0: return None
    m=HistGradientBoostingRegressor(loss="squared_error",**{k:v for k,v in params.items()})
    m.fit(X[tr],yv[tr]); o=Pm.loc[te,["date","ticker"]].copy(); o["s"]=m.predict(X[te]); return o
def run_rel(cols,h,rank=False,window=None,params=None):
    t=f"y{h}"
    # target: rank-gauss del retorno relativo por fecha
    r=P.groupby("date")[t].rank(pct=True)
    n=P.groupby("date")[t].transform("count")
    yv=pd.Series(norm.ppf(((r*n-0.5)/n).clip(0.001,0.999)),index=P.index).where(P[t].notna()&(n>=10))
    X=xs_rank(P,cols) if rank else P[cols]
    pr={**PARAMS,**(params or {})}
    outs=Parallel(n_jobs=10)(delayed(fit_reg)(P,X,yv,yr,h,pr,window) for yr in YEARS)
    o=pd.concat([x for x in outs if x is not None]); o["yy"]=P[t].reindex(o.index); return o.dropna(subset=["yy"])
def rep_ic(name,o):
    ics=[]
    for d,gp in o.groupby("date"):
        if len(gp)>=10: ics.append((d,spearmanr(gp.s,gp.yy)[0]))
    ic=pd.Series(dict(ics)); bym=ic.groupby(ic.index.to_period("M")).mean()
    t=bym.mean()/(bym.std()/np.sqrt(len(bym)))
    # long-short top/bottom quintile
    o=o.assign(q=o.groupby("date").s.rank(pct=True)); top=o[o.q>0.8].groupby("date").yy.mean(); bot=o[o.q<0.2].groupby("date").yy.mean()
    ls=(top-bot).dropna(); lm=ls.groupby(ls.index.to_period("M")).mean()
    print(f"{name:30s} IC {ic.mean():+.4f} (t_mes {t:+.2f}) | %meses IC>0 {(bym>0).mean():.2f} | L-S quintil {ls.mean()*100:+.2f}% por {name[-3:] if False else 'periodo'} (t_mes {lm.mean()/(lm.std()/np.sqrt(len(lm))):+.2f}) top {top.mean()*100:+.2f}% bot {bot.mean()*100:+.2f}",flush=True)
TM=g["tech"]+g["mkt"]
for h in (5,20):
    print("== relativo h",h)
    rep_ic("tech",run_rel(g["tech"],h)); 
    rep_ic("tech+mkt",run_rel(TM,h))
    rep_ic("tech+mkt xs-rank",run_rel(TM,h,rank=True))
    rep_ic("tech+mkt+fx",run_rel(TM+g["fx"],h))
    rep_ic("tech+mkt+fx+mac",run_rel(TM+g["fx"]+g["mac"],h))
    rep_ic("all",run_rel(sum(g.values(),[]),h))
