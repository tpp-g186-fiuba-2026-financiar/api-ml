"""LSTM / GRU sobre ventanas de las mismas variables, vs arboles y MLP, con un corte unico (<=2022 / 2023+)."""
import os, warnings, sys
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); torch.set_num_threads(8)
R='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research/'
P=pd.read_pickle(R+"panel_safe.pkl")
ALL=[c for c in sorted({c for c in P.columns if len(c)>2 and c[1]=="_" and c[0] in "tmsgae"}) if c not in ("e_month","e_dow")]
W=20
def make(h):
    cut=np.datetime64("2022-12-31"); X,Y,D=[],[],[]
    for t,g in P.groupby("ticker"):
        g=g.sort_index(); F=g[ALL].to_numpy(float); f=g[f"fwd{h}"].to_numpy(); d=g.index.values
        for i in range(W-1,len(g)):
            if np.isnan(f[i]) or f[i]==0: continue
            X.append(F[i-W+1:i+1]); Y.append(int(f[i]<0)); D.append(d[i])
    return np.asarray(X,np.float32),np.asarray(Y),np.asarray(D)
class Net(nn.Module):
    def __init__(s,nf,kind):
        super().__init__(); s.kind=kind
        s.rnn=(nn.LSTM if kind=="LSTM" else nn.GRU)(nf,32,batch_first=True); s.drop=nn.Dropout(0.3); s.out=nn.Linear(32,1)
    def forward(s,x): o,_=s.rnn(x); return s.out(s.drop(o[:,-1])).squeeze(-1)
def fit_net(Xtr,ytr,Xva,yva,Xte,kind,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu=np.nanmean(Xtr.reshape(-1,Xtr.shape[-1]),axis=0); sd=np.nanstd(Xtr.reshape(-1,Xtr.shape[-1]),axis=0)+1e-8
    prep=lambda X: torch.tensor(np.clip(np.nan_to_num((X-mu)/sd,nan=0.0),-5,5),dtype=torch.float32)
    A,B,C=prep(Xtr),prep(Xva),prep(Xte); ya=torch.tensor(ytr,dtype=torch.float32)
    net=Net(Xtr.shape[-1],kind); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3); lossf=nn.BCEWithLogitsLoss()
    best,state,bad=9,None,0
    for ep in range(25):
        net.train(); perm=torch.randperm(len(A))
        for i in range(0,len(A),256):
            ix=perm[i:i+256]; opt.zero_grad(); lossf(net(A[ix]),ya[ix]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl=lossf(net(B),torch.tensor(yva,dtype=torch.float32)).item()
        if vl<best-1e-4: best,state,bad=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=4: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return net(C).numpy()
for h in (20,30):
    X,Y,D=make(h); cutd=np.datetime64("2022-12-31")-np.timedelta64(2*h,"D")
    tr_all=np.where(D<=cutd)[0]; te=np.where(D>=np.datetime64("2023-01-01"))[0]
    days=np.unique(D[tr_all]); vday=days[int(len(days)*0.85)]; tr=tr_all[D[tr_all]<=vday]; va=tr_all[D[tr_all]>vday]
    res={}
    for kind in ("LSTM","GRU"):
        sc=np.mean([fit_net(X[tr],Y[tr],X[va],Y[va],X[te],kind,s) for s in (0,1,2)],axis=0); res[f"{kind} (ventana {W} dias, 3 semillas)"]=roc_auc_score(Y[te],sc)
    flat=lambda A:A[:,-1,:]   # solo el ultimo dia
    hg=HistGradientBoostingClassifier(max_depth=2,learning_rate=0.05,max_iter=150,min_samples_leaf=500,l2_regularization=5.0,random_state=0).fit(flat(X[tr_all]),Y[tr_all]); res["HGB (solo ultimo dia)"]=roc_auc_score(Y[te],hg.predict_proba(flat(X[te]))[:,1])
    mlp=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),FunctionTransformer(lambda Z:np.clip(Z,-5,5)),MLPClassifier(hidden_layer_sizes=(32,),alpha=0.05,early_stopping=True,max_iter=200,random_state=0)).fit(flat(X[tr_all]),Y[tr_all]); res["MLP (solo ultimo dia)"]=roc_auc_score(Y[te],mlp.predict_proba(flat(X[te]))[:,1])
    print(f"\n### horizonte {h}: entrena <=2022, prueba 2023-26 (n_test={len(te)}; base baja {Y[te].mean()*100:.1f}%)")
    for k,v in res.items(): print(f"  {k:36s} AUC={v:.3f}")
