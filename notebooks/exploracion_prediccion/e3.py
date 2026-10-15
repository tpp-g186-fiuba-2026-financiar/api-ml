from ev import *
g=G
# panel de mercado: una fila por fecha
mk=["mkt_r5","mkt_r20","mkt_r60","mkt_vol20","brd50","mkt_dd"]
fxc=["ccl_r5","ccl_r20","ccl_r60","brecha","brecha_d20","of_r20"]
mac=g["mac"]; gl=[c for c in g["glob"]]
cols=mk+fxc+mac+gl
agg=P.groupby("date")[cols].first()
for h in (5,20,40):
    fw=(P.assign(mf=P[f"y{h}"]-P[f"ym{h}"]) if f"y{h}" in P else None)
for h in (5,20):
    pass
M=agg.copy()
for h in (5,20):
    M[f"mf{h}"]=(P[f"y{h}"]-P[f"ym{h}"]).groupby(P.date).first()
# h=40 desde el indice de la canasta: reconstruir
idx=(P.groupby("date").apply(lambda d:(d.y5-d.ym5).iloc[0])) # fw5
M=M.reset_index()
def runM(cols,h,**kw):
    y=M[f"mf{h}"]; lab=(y>0).astype(float).where(y.notna())
    class _:  # reuse fit_predict_year con M como panel
        pass
    Pm=M.assign(ticker="MKT")
    X=M[cols]
    outs=Parallel(n_jobs=10)(delayed(fit_predict_year)(Pm,X,lab,yr,h,{**PARAMS,"min_samples_leaf":60,**kw.get("params",{})},min_train=700) for yr in YEARS)
    o=pd.concat([x for x in outs if x is not None]); o["yy"]=y.reindex(o.index); o["lab"]=lab.reindex(o.index)
    return o.dropna(subset=["lab"])
def rep(name,o):
    a,l,u=boot_auc(o,n=200)
    pa=[roc_auc_score(gp.lab,gp.p) for _,gp in o.groupby(o.date.dt.year) if gp.lab.nunique()>1]
    acc_thr=((o.p>0.5)==(o.lab==1)).mean()
    print(f"{name:26s} AUC {a:.3f} [{l:.3f},{u:.3f}] anios>.5 {sum(x>.5 for x in pa)}/{len(pa)} | acc@.5 {acc_thr:.3f} base-up {o.lab.mean():.3f}",flush=True)
for h in (5,20):
    print("== MERCADO h=",h)
    for name,c in {"tecnico mercado":mk,"+fx":mk+fxc,"+macro":mk+fxc+mac,"+global":mk+fxc+mac+gl,"solo macro+fx":fxc+mac,"solo global":gl,"mk+global":mk+gl}.items():
        rep(name,runM(c,h))
