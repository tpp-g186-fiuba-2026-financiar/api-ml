"""Construye el panel (fecha x accion) con grupos de features y targets."""
import warnings, glob, os
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
D = os.path.join(os.path.dirname(__file__), "d")

AR = "GGAL BMA YPFD PAMP TXAR ALUA CRES EDN TECO2 TGSU2 TGNO4 CEPU COME LOMA MIRG SUPV VALO BYMA CVH HARG TRAN METR BBAR BPAT MOLI AGRO BHIP IRSA CELU SEMI DGCU2 LEDE OEST GCLA CAPX".split()
ADR = {"GGAL": "GGAL", "BMA": "BMA", "YPFD": "YPF", "PAMP": "PAM", "TECO2": "TEO", "EDN": "EDN", "CEPU": "CEPU",
       "CRES": "CRESY", "LOMA": "LOMA", "SUPV": "SUPV", "TGSU2": "TGS", "BBAR": "BBAR", "IRSA": "IRS"}


def rd(name):
    p = f"{D}/{name}.csv"
    df = pd.read_csv(p, parse_dates=["date"]).drop_duplicates("date").set_index("date").sort_index()
    return df


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def series_csv(name, col="valor", datecol="fecha"):
    df = pd.read_csv(f"{D}/{name}.csv")
    df[datecol] = pd.to_datetime(df[datecol])
    return df.drop_duplicates(datecol).set_index(datecol).sort_index()[col]


def load_macro(idx):
    """Series macro alineadas a `idx` con rezago conservador (dato del dia anterior)."""
    m = pd.DataFrame(index=idx)
    ccl = series_csv("ar_ccl", "venta"); of = series_csv("ar_oficial", "venta"); mep = series_csv("ar_mep", "venta")
    m["ccl"] = ccl.reindex(idx, method="ffill").shift(1)
    m["of"] = of.reindex(idx, method="ffill").shift(1)
    m["mep"] = mep.reindex(idx, method="ffill").shift(1)
    m["riesgo"] = series_csv("ar_riesgo").reindex(idx, method="ffill").shift(1)
    for n in ("badlar", "reservas", "base"):
        m[n] = series_csv(f"bcra_{n}").reindex(idx, method="ffill").shift(5)
    return m


def load_global(idx):
    g = pd.DataFrame(index=idx)
    for name, f in {"spy": "SPY", "eem": "EEM", "ewz": "EWZ", "vix": "_VIX", "dxy": "DX-Y.NYB", "tnx": "_TNX",
                    "oil": "CL_F", "gold": "GC_F", "brl": "BRL_X", "merv": "_MERV", "argt": "ARGT"}.items():
        s = rd(f)["close"]
        g[name] = s.reindex(idx, method="ffill")
    return g


def build(start="2011-01-01", horizons=(5, 20)):
    # calendario: dias de GGAL.BA con datos
    cal = rd("GGAL.BA").index
    cal = cal[cal >= start]
    px, vol, hi, lo, op = {}, {}, {}, {}, {}
    for t in AR:
        d = rd(f"{t}.BA")
        d = d[~d.index.duplicated()].reindex(cal)
        px[t], vol[t], hi[t], lo[t], op[t] = d.close, d.volume, d.high, d.low, d.open
    px, vol, hi, lo, op = [pd.DataFrame(x) for x in (px, vol, hi, lo, op)]
    # calidad: descartar dias sin cambio de precio (dato estancado) por accion
    stale = (px.pct_change(fill_method=None).abs() < 1e-12)
    # ventana movil de 60 dias con >50% estancado => serie no usable en ese tramo
    bad = stale.rolling(60, min_periods=30).mean() > 0.5
    px = px.where(~bad)
    m = load_macro(cal); g = load_global(cal)
    lr = np.log(px).diff()
    # --- canasta (proxy Merval interno): promedio de retornos de los 8 mas liquidos por volumen mediano en ARS
    dv = (px * vol).rolling(120, min_periods=60).median()
    rank = dv.rank(axis=1, ascending=False)
    lead = (rank <= 8)
    mkt_lr = lr.where(lead).mean(axis=1)
    mkt_idx = mkt_lr.fillna(0).cumsum()
    brd50 = (px > px.rolling(50).mean()).where(px.notna()).mean(axis=1)
    feats = {}
    ccl = m["ccl"]; ofi = m["of"]
    for t in AR:
        c = px[t]; l = lr[t]; v = vol[t]
        f = pd.DataFrame(index=cal)
        # --- tecnicas
        for h in (1, 2, 3, 5, 10, 20, 60, 120, 250):
            f[f"r{h}"] = np.log(c).diff(h)
        f["vol20"] = l.rolling(20).std(); f["vol60"] = l.rolling(60).std(); f["vol_ratio"] = f.vol20 / f.vol60
        f["rsi14"] = rsi(c); f["rsi5"] = rsi(c, 5)
        for n in (20, 50, 200):
            f[f"dma{n}"] = c / c.rolling(n).mean() - 1
        f["dd252"] = c / c.rolling(250, min_periods=100).max() - 1
        f["up252"] = c / c.rolling(250, min_periods=100).min() - 1
        f["sharpe20"] = f.r20 / (f.vol20 * np.sqrt(20))
        f["sharpe60"] = f.r60 / (f.vol60 * np.sqrt(60))
        f["rng"] = (hi[t] - lo[t]) / c
        f["rng_z"] = f.rng / f.rng.rolling(60).mean()
        f["gap"] = np.log(op[t] / c.shift(1))
        lv = np.log1p(v.replace(0, np.nan))
        f["volu_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
        f["skew20"] = l.rolling(20).skew(); f["kurt60"] = l.rolling(60).kurt()
        f["maxr20"] = l.rolling(20).max(); f["minr20"] = l.rolling(20).min()
        f["ac1"] = l.rolling(60).corr(l.shift(1))
        f["up_days20"] = (l > 0).rolling(20).mean()
        # --- mercado interno
        f["mkt_r5"] = mkt_idx.diff(5); f["mkt_r20"] = mkt_idx.diff(20); f["mkt_r60"] = mkt_idx.diff(60)
        f["mkt_vol20"] = mkt_lr.rolling(20).std(); f["brd50"] = brd50
        f["rel5"] = f.r5 - f.mkt_r5; f["rel20"] = f.r20 - f.mkt_r20; f["rel60"] = f.r60 - f.mkt_r60
        f["beta60"] = l.rolling(60).cov(mkt_lr) / mkt_lr.rolling(60).var()
        f["mkt_dd"] = mkt_idx - mkt_idx.rolling(250, min_periods=100).max()
        # --- fx / usd
        usd = c / ccl
        f["usd_r20"] = np.log(usd).diff(20); f["usd_r60"] = np.log(usd).diff(60); f["usd_r120"] = np.log(usd).diff(120)
        f["usd_dd"] = usd / usd.rolling(250, min_periods=100).max() - 1
        for h in (5, 20, 60):
            f[f"ccl_r{h}"] = np.log(ccl).diff(h)
        f["brecha"] = ccl / ofi - 1; f["brecha_d20"] = f.brecha.diff(20)
        f["mep_ccl"] = m["mep"] / ccl - 1
        f["of_r20"] = np.log(ofi).diff(20)
        f["ars_vs_ccl20"] = f.r20 - f.ccl_r20   # = retorno en USD
        feats_fx = f
        # --- macro AR
        f["riesgo"] = m["riesgo"]; f["riesgo_d5"] = m["riesgo"].diff(5); f["riesgo_d20"] = m["riesgo"].diff(20); f["riesgo_d60"] = m["riesgo"].diff(60)
        f["riesgo_pct250"] = m["riesgo"].rolling(250, min_periods=100).rank(pct=True)
        f["badlar"] = m["badlar"]; f["badlar_d20"] = m["badlar"].diff(20)
        f["real_rate"] = m["badlar"] - np.log(ccl).diff(20) * 100 * 12   # tasa menos devaluacion anualizada (aprox)
        f["res_r60"] = np.log(m["reservas"]).diff(60); f["base_r60"] = np.log(m["base"]).diff(60)
        # --- global
        for k in ("spy", "eem", "ewz", "dxy", "oil", "gold", "brl", "merv", "argt"):
            s = g[k]
            f[f"g_{k}_r5"] = np.log(s).diff(5); f[f"g_{k}_r20"] = np.log(s).diff(20)
        f["g_vix"] = g["vix"]; f["g_vix_d5"] = g["vix"].diff(5); f["g_tnx"] = g["tnx"]; f["g_tnx_d20"] = g["tnx"].diff(20)
        f["g_merv_usd_r20"] = np.log(g["merv"] / ccl).diff(20)
        f["g_spy_dd"] = g["spy"] / g["spy"].rolling(250, min_periods=100).max() - 1
        # --- ADR
        if t in ADR:
            a = rd(ADR[t])["close"].reindex(cal, method="ffill")
            ratio = c / (a * ccl)
            lg = np.log(ratio)
            f["adr_gap"] = lg - lg.rolling(60).median()
            f["adr_gap_d5"] = f.adr_gap.diff(5)
            f["adr_r1"] = np.log(a).diff(1); f["adr_r5"] = np.log(a).diff(5)
            f["adr_ex_r5"] = f.adr_r5 - (f.r5 - f.ccl_r5)  # retorno ADR (USD) menos retorno local en USD
        f["ticker"] = t
        # --- targets
        for h in horizons:
            fw = np.log(c).shift(-h) - np.log(c)
            f[f"y{h}"] = fw
            f[f"yu{h}"] = np.log(usd).shift(-h) - np.log(usd)
            f[f"ym{h}"] = fw - (mkt_idx.shift(-h) - mkt_idx)
            f[f"vf{h}"] = l.rolling(h).std().shift(-h)
        feats[t] = f
    P = pd.concat(feats.values()).reset_index().rename(columns={"index": "date"})
    P["date"] = pd.to_datetime(P["date"])
    P = P.dropna(subset=["r20", "vol60"]).reset_index(drop=True)
    return P


GROUPS = {}


def define_groups(P):
    cols = [c for c in P.columns if c not in ("date", "ticker") and not c.startswith(("y", "vf"))]
    tech = [c for c in cols if c.split("_")[0] in () or c in
            [f"r{h}" for h in (1, 2, 3, 5, 10, 20, 60, 120, 250)] + ["vol20", "vol60", "vol_ratio", "rsi14", "rsi5", "dma20", "dma50", "dma200",
             "dd252", "up252", "sharpe20", "sharpe60", "rng", "rng_z", "gap", "volu_z", "skew20", "kurt60", "maxr20", "minr20", "ac1", "up_days20"]]
    mkt = ["mkt_r5", "mkt_r20", "mkt_r60", "mkt_vol20", "brd50", "rel5", "rel20", "rel60", "beta60", "mkt_dd"]
    fx = ["usd_r20", "usd_r60", "usd_r120", "usd_dd", "ccl_r5", "ccl_r20", "ccl_r60", "brecha", "brecha_d20", "mep_ccl", "of_r20", "ars_vs_ccl20"]
    mac = ["riesgo", "riesgo_d5", "riesgo_d20", "riesgo_d60", "riesgo_pct250", "badlar", "badlar_d20", "real_rate", "res_r60", "base_r60"]
    glob_ = [c for c in cols if c.startswith("g_")]
    adr = [c for c in cols if c.startswith("adr_")]
    return dict(tech=tech, mkt=mkt, fx=fx, mac=mac, glob=glob_, adr=adr)
