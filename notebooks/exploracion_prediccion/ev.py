import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import feat

P = pd.read_pickle("panel.pkl")
G = feat.define_groups(P)
YEARS = list(range(2016, 2027))
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def xs_rank(df, cols):
    """Normalizacion cross-seccional (rank 0-1 por fecha)."""
    out = df[["date"] + cols].copy()
    out[cols] = out.groupby("date")[cols].rank(pct=True)
    return out[cols]


def make_label(P, target, kind):
    y = P[target]
    if kind == "up":
        return (y > 0).astype(float).where(y.notna())
    if kind == "ex":  # supera a la mediana del dia (relativo)
        med = P.groupby("date")[target].transform("median")
        return (y > med).astype(float).where(y.notna())
    if kind.startswith("big"):  # P(mov > k*sigma) / extremos
        raise NotImplementedError


def fit_predict_year(Pm, X, lab, yr, h, params, window=None, per_ticker=False, weight=None, model="hgb", min_train=3000):
    test = (Pm.date.dt.year == yr)
    cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=int(h * 1.6) + 2)
    tr = (Pm.date < cut) & lab.notna()
    if window:
        tr &= Pm.date >= cut - pd.Timedelta(days=int(window * 365))
    te = test & True
    if tr.sum() < min_train or te.sum() == 0:
        return None
    Xtr, Xte = X[tr], X[te]
    ytr = lab[tr]
    w = None if weight is None else weight[tr]
    if model == "hgb":
        m = HistGradientBoostingClassifier(**params)
        m.fit(Xtr, ytr, sample_weight=w)
        p = m.predict_proba(Xte)[:, 1]
    elif model == "logit":
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        m = LogisticRegression(C=0.05, max_iter=500)
        m.fit(((Xtr - mu) / sd).fillna(0), ytr, sample_weight=w)
        p = m.predict_proba(((Xte - mu) / sd).fillna(0))[:, 1]
    out = Pm.loc[te, ["date", "ticker"]].copy()
    out["p"] = p
    return out


def run(cols, target="y20", kind="up", h=None, params=None, rank=False, **kw):
    import re
    h = h or int(re.sub(r"\D", "", target))
    params = {**PARAMS, **(params or {})}
    X = P[cols].copy()
    if rank:
        X = xs_rank(P, cols)
    lab = make_label(P, target, kind)
    outs = Parallel(n_jobs=10)(delayed(fit_predict_year)(P, X, lab, yr, h, params, **kw) for yr in YEARS)
    o = pd.concat([x for x in outs if x is not None])
    o["yy"] = P[target].reindex(o.index)
    o["lab"] = lab.reindex(o.index)
    return o.dropna(subset=["lab"])


def boot_auc(o, n=300, seed=0):
    """AUC pooled con IC por bootstrap de bloques mensuales."""
    rng = np.random.default_rng(seed)
    o = o.assign(m=o.date.dt.to_period("M"))
    groups = {k: g for k, g in o.groupby("m")}
    keys = list(groups)
    a = roc_auc_score(o.lab, o.p)
    vals = []
    for _ in range(n):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys))])
        vals.append(roc_auc_score(s.lab, s.p))
    return a, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def deciles(o, q=0.1):
    hi = o[o.p >= o.groupby(o.date.dt.year).p.transform(lambda s: s.quantile(1 - q))]
    lo = o[o.p <= o.groupby(o.date.dt.year).p.transform(lambda s: s.quantile(q))]
    return hi.lab.mean(), o.lab.mean(), 1 - lo.lab.mean(), 1 - o.lab.mean(), hi.yy.mean(), lo.yy.mean(), len(hi)


def report(name, o, quick=False):
    a, l, u = boot_auc(o, n=100 if quick else 300)
    pa = [roc_auc_score(g.lab, g.p) for _, g in o.groupby(o.date.dt.year) if g.lab.nunique() > 1]
    th, base, tl, basel, rh, rl, n = deciles(o)
    print(f"{name:34s} AUC {a:.3f} [{l:.3f},{u:.3f}] anios>.5: {sum(x > .5 for x in pa)}/{len(pa)} | top10% acierta {th:.3f} (base {base:.3f}) | bot10% baja {tl:.3f} (base {basel:.3f}) | ret top {rh:+.3f} bot {rl:+.3f}", flush=True)
    return a
