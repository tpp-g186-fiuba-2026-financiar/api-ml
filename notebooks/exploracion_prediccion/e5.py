exec(open("e4.py").read().split("TM=g[")[0])
TM=g["tech"]+g["mkt"]
for L in (1,2,3):
    P[f"y5L{L}"]=P.groupby("ticker").y5.shift(-L)
def run_rel2(cols,t,h,**kw):
    r=P.groupby("date")[t].rank(pct=True); n=P.groupby("date")[t].transform("count")
    yv=pd.Series(norm.ppf(((r*n-0.5)/n).clip(0.001,0.999)),index=P.index).where(P[t].notna()&(n>=10))
    X=P[cols]; outs=Parallel(n_jobs=10)(delayed(fit_reg)(P,X,yv,yr,h,PARAMS) for yr in YEARS)
    o=pd.concat([x for x in outs if x is not None]); o["yy"]=P[t].reindex(o.index); return o.dropna(subset=["yy"])
print("== efecto de rezago de entrada (microestructura/precio estancado)")
for L in (0,1,2,3):
    t="y5" if L==0 else f"y5L{L}"
    rep_ic(f"tech+mkt entrada t+{L}",run_rel2(TM,t,5))
print("== IC por anio (tech+mkt, h5, lag0)")
o=run_rel2(TM,"y5",5)
for yr,gp in o.groupby(o.date.dt.year):
    ics=[spearmanr(x.s,x.yy)[0] for _,x in gp.groupby("date") if len(x)>=10]; print(yr,round(np.mean(ics),4),end=" | ")
print()
print("== all sin adr / sin glob / sin mac")
al=sum(g.values(),[])
rep_ic("all",run_rel2(al,"y5",5))
rep_ic("all - adr",run_rel2([c for c in al if c not in g["adr"]],"y5",5))
rep_ic("all - adr - glob",run_rel2([c for c in al if c not in g["adr"] and c not in g["glob"]],"y5",5))
rep_ic("solo adr+tech",run_rel2(g["tech"]+g["adr"],"y5",5))
