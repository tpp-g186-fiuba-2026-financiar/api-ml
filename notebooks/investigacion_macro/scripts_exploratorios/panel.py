"""Panel de investigacion: (accion, dia) con features por familia y targets de caida.

Reglas anti-lookahead: toda feature usa solo datos <= t. Las series de EEUU se
retrasan 1 dia (cierran casi a la hora que BYMA: se evita cualquier solapamiento).
Targets (solo para etiquetar): retorno futuro a h ruedas.
"""
import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0,'/Users/matimorales01/Desktop/tp_profesional/api-ml')
from src.indicators import rsi_series, macd_histogram, sma, ema
SP='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/'
UNI=["CRES","ALUA","TECO2","BBAR","METR","SUPV","BYMA","PAMP","COME","CEPU","EDN","TRAN","GGAL","TXAR","VALO","TGSU2","TGNO4","YPFD","BMA","LOMA"]
HOR=[1,5,10,14,20,30,40,60]
ELECTIONS=pd.to_datetime(["2017-10-22","2019-08-11","2019-10-27","2021-09-12","2021-11-14","2023-08-13","2023-10-22","2023-11-19","2025-10-26"])

def lg(s): return np.log(s)

def tech(df):
    c=df["close"]; o=df["open"]; h=df["high"]; l=df["low"]; v=df["volume"].replace(0,np.nan)
    r=lg(c).diff(); f=pd.DataFrame(index=df.index)
    for k in (1,3,5,10,20,60,120,250): f[f"t_ret{k}"]=lg(c).diff(k)
    dn=(r<0).astype(int); up=(r>0).astype(int)
    grp=(dn.diff()!=0).cumsum(); f["t_down_streak"]=dn.groupby(grp).cumsum()*dn
    grp=(up.diff()!=0).cumsum(); f["t_up_streak"]=up.groupby(grp).cumsum()*up
    f["t_dd252"]=c/c.rolling(252,min_periods=60).max()-1; f["t_dd60"]=c/c.rolling(60).max()-1
    f["t_dist_min60"]=c/c.rolling(60).min()-1; f["t_new_low252"]=(c<=c.rolling(252,min_periods=60).min()*1.0001).astype(float)
    s50=c.rolling(50).mean(); s200=c.rolling(200,min_periods=120).mean(); s20=c.rolling(20).mean()
    f["t_sma20"]=c/s20-1; f["t_sma50"]=c/s50-1; f["t_sma200"]=c/s200-1; f["t_death_cross"]=(s50<s200).astype(float)
    f["t_gap"]=o/c.shift()-1; f["t_gap_down"]=(f["t_gap"]<-0.02).astype(float)
    for k in (5,20,60): f[f"t_vol{k}"]=r.rolling(k).std()
    f["t_vol_ratio5_20"]=f["t_vol5"]/f["t_vol20"]; f["t_vol_ratio20_60"]=f["t_vol20"]/f["t_vol60"]
    neg=r.where(r<0); f["t_downdev"]=neg.rolling(20,min_periods=3).std()/f["t_vol20"]
    f["t_skew20"]=r.rolling(20).skew(); f["t_kurt20"]=r.rolling(20).kurt()
    hl=np.log(h/l.replace(0,np.nan))**2; co=np.log(c/o.replace(0,np.nan))**2; gk=(0.5*hl-(2*np.log(2)-1)*co).clip(lower=0)
    f["t_gk20"]=np.sqrt(gk.rolling(20).mean()); f["t_gk_ratio"]=np.sqrt(gk.rolling(5).mean())/f["t_gk20"].replace(0,np.nan)
    lv=np.log1p(df["volume"]); f["t_volz"]=(lv-lv.rolling(20).mean())/lv.rolling(20).std()
    f["t_volz_x_ret"]=f["t_volz"].fillna(0)*np.sign(r); f["t_vol_chg5"]=lv.diff(5)
    obv=(np.sign(r).fillna(0)*df["volume"]).cumsum(); f["t_obv_slope"]=(obv-obv.shift(20))/(df["volume"].rolling(20).sum()+1)
    f["t_amihud"]=(r.abs()/(c*v)).rolling(20).mean(); f["t_amihud"]=np.log1p(f["t_amihud"]/ (f["t_amihud"].rolling(250,min_periods=60).median()+1e-12))
    cl=np.asarray(c,dtype=float); f["t_rsi14"]=rsi_series(cl); f["t_rsi5"]=rsi_series(cl,5); f["t_macd"]=macd_histogram(cl)/cl
    lo14=l.rolling(14).min(); hi14=h.rolling(14).max(); f["t_stoch"]=(c-lo14)/(hi14-lo14).replace(0,np.nan)
    f["t_boll"]=(c-s20)/c.rolling(20).std()
    return f.replace([np.inf,-np.inf],np.nan)

def market_block(closes):
    """Features de mercado/amplitud calculadas cruzando las acciones (misma fecha)."""
    C=closes; R=np.log(C).diff(); mk=R.mean(axis=1); lvl=mk.fillna(0).cumsum(); m=pd.DataFrame(index=C.index)
    for k in (1,5,20,60): m[f"m_ret{k}"]=lvl.diff(k)
    m["m_vol20"]=mk.rolling(20).std(); m["m_vol60"]=mk.rolling(60).std(); m["m_dd252"]=lvl-lvl.rolling(252,min_periods=60).max()
    s20=C.rolling(20).mean(); s50=C.rolling(50).mean()
    m["m_breadth20"]=(C>s20).mean(axis=1); m["m_breadth50"]=(C>s50).mean(axis=1)
    m["m_neg20"]=(np.log(C).diff(20)<0).mean(axis=1); m["m_low60"]=(C<=C.rolling(60).min()*1.0001).mean(axis=1)
    var_i=R.rolling(20).var().mean(axis=1); m["m_corr_proxy"]=mk.rolling(20).var()/var_i.replace(0,np.nan)
    m["m_dispersion"]=R.std(axis=1).rolling(5).mean()
    return m,mk,R

def beta_block(R,mk):
    out={}
    cov=R.rolling(60).cov(mk); var=mk.rolling(60).var()
    beta=cov.div(var,axis=0); out["beta60"]=beta
    return out

def glob(E,idx):
    """Mercado global, retrasado 1 dia."""
    g=pd.DataFrame(index=E.index); L=lambda s:s.shift(1)
    sp=L(E["sp500"]); g["g_sp1"]=lg(sp).diff(1); g["g_sp5"]=lg(sp).diff(5); g["g_sp20"]=lg(sp).diff(20); g["g_sp_dd"]=sp/sp.rolling(252,min_periods=60).max()-1
    vx=L(E["vix"]); g["g_vix"]=vx; g["g_vix_chg5"]=vx.diff(5); g["g_vix_z"]=(vx-vx.rolling(252,min_periods=60).mean())/vx.rolling(252,min_periods=60).std()
    ve=L(E["vxeem"]); g["g_vxeem"]=ve; g["g_vxeem_chg5"]=ve.diff(5)
    g["g_dxy20"]=lg(L(E["dxy"])).diff(20); g["g_wti20"]=lg(L(E["wti"])).diff(20)
    g["g_us10y_chg20"]=L(E["us10y"]).diff(20); g["g_curve"]=L(E["us10y"])-L(E["us2y"]); g["g_us2y_chg20"]=L(E["us2y"]).diff(20)
    g["g_brl5"]=lg(L(E["brl"])).diff(5); g["g_brl20"]=lg(L(E["brl"])).diff(20); g["g_mxn20"]=lg(L(E["mxn"])).diff(20)
    g["g_nasdaq5"]=lg(L(E["nasdaq"])).diff(5); g["g_nasdaq20"]=lg(L(E["nasdaq"])).diff(20)
    return g.reindex(idx,method="ffill")

def argmac(E,idx):
    a=pd.DataFrame(index=E.index); ccl=E["ar_contadoconliqui"]; of=E["ar_oficial"]; may=E["ar_mayorista"]
    for k in (1,5,20,60): a[f"a_ccl{k}"]=lg(ccl).diff(k)
    a["a_brecha"]=ccl/of-1; a["a_brecha_chg20"]=a["a_brecha"].diff(20); a["a_brecha_z"]=(a["a_brecha"]-a["a_brecha"].rolling(250,min_periods=60).mean())/a["a_brecha"].rolling(250,min_periods=60).std()
    a["a_mep_ccl"]=E["ar_bolsa"]/ccl-1; a["a_blue_gap"]=E["ar_blue"]/of-1
    a["a_of20"]=lg(may).diff(20)
    rp=E["riesgo_pais"]; a["a_rp"]=rp; a["a_rp_chg5"]=lg(rp).diff(5); a["a_rp_chg20"]=lg(rp).diff(20); a["a_rp_z"]=(rp-rp.rolling(250,min_periods=60).mean())/rp.rolling(250,min_periods=60).std()
    res=E["reservas"]; a["a_res20"]=lg(res).diff(20); a["a_res60"]=lg(res).diff(60)
    a["a_badlar"]=E["badlar"]; a["a_badlar_chg20"]=E["badlar"].diff(20)
    bm=E["base_monetaria"]; a["a_bm20"]=lg(bm).diff(20)
    a["a_real_rate"]=E["badlar"]-(lg(ccl).diff(20)*(365/20)*100)   # tasa vs depreciacion anualizada
    return a.reindex(idx,method="ffill")

def events(idx):
    e=pd.DataFrame(index=idx); d=idx.values.astype("datetime64[D]")
    el=ELECTIONS.values.astype("datetime64[D]")
    nxt=[(el[el>=x].min()-x).astype(int) if (el>=x).any() else 999 for x in d]; prv=[(x-el[el<=x].max()).astype(int) if (el<=x).any() else 999 for x in d]
    e["e_days_to_el"]=np.minimum(nxt,90); e["e_days_since_el"]=np.minimum(prv,90)
    e["e_el_window"]=((e["e_days_to_el"]<=10)|(e["e_days_since_el"]<=10)).astype(float)
    e["e_dow"]=idx.dayofweek; e["e_month"]=idx.month; e["e_month_end"]=(idx.is_month_end|(idx+pd.Timedelta(days=3)).is_month_start).astype(float)
    e["e_aguinaldo"]=idx.month.isin([6,12]).astype(float)
    return e

def build(h_list=HOR,safe=False,out=None):
    hist=pickle.load(open(SP+"hist.pkl","rb")); E=pd.read_pickle(SP+"ext/E.pkl")
    if safe:  # retrasos conservadores: mercado argentino 2 dias, BCRA (publica con demora) 5 dias calendario
        for c in ("ar_oficial","ar_mayorista","ar_contadoconliqui","ar_bolsa","ar_blue","riesgo_pais"): E[c]=E[c].shift(2)
        for c in ("reservas","badlar","base_monetaria","tc_mayorista_bcra"): E[c]=E[c].shift(5)
    closes=pd.DataFrame({t:hist[t]["close"] for t in UNI}).sort_index(); closes=closes[closes.index>="2016-06-01"]
    m,mk,R=market_block(closes); B=beta_block(R,mk)["beta60"]
    G=glob(E,closes.index); A=argmac(E,closes.index); EV=events(closes.index)
    frames=[]
    for t in UNI:
        df=hist[t].loc[closes.index[0]:]; f=tech(df)
        f=f.join(m,how="left").join(G).join(A).join(EV)
        f["s_beta60"]=B[t]; f["s_rel20"]=f["t_ret20"]-f["m_ret20"]; f["s_rel60"]=f["t_ret60"]-f["m_ret60"]; f["s_rel5"]=f["t_ret5"]-f["m_ret5"]
        f["s_idiovol"]=f["t_vol20"]-f["m_vol20"]
        c=df["close"]
        for h in h_list:
            fwd=c.shift(-h)/c-1; f[f"fwd{h}"]=fwd
            z=fwd/(f["t_vol20"]*np.sqrt(h)); f[f"zfwd{h}"]=z
        f["ticker"]=t; frames.append(f)
    P=pd.concat(frames); P=P[P.index>="2017-01-02"]
    return P
if __name__=="__main__":
    P=build(safe=True); P.to_pickle(SP+"research/panel_safe.pkl"); print(P.shape)
    fam={}
    for c in P.columns:
        fam.setdefault(c.split("_")[0] if "_" in c else "otro",[]).append(c)
    print({k:len(v) for k,v in fam.items()})
