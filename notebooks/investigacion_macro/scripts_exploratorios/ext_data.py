"""Descarga y arma la tabla diaria de variables externas (mercado global + Argentina)."""
import json, subprocess, os, pandas as pd, numpy as np
D=os.path.dirname(os.path.abspath(__file__))+"/../ext/"
def curl(url,out,k=False):
    if os.path.exists(out) and os.path.getsize(out)>200: return
    subprocess.run(["curl","-s","-m","90"]+(["-k"] if k else [])+["-o",out,url],check=False)
FRED={"sp500":"SP500","vix":"VIXCLS","vxeem":"VXEEMCLS","dxy":"DTWEXBGS","wti":"DCOILWTICO","us10y":"DGS10","us2y":"DGS2","brl":"DEXBZUS","mxn":"DEXMXUS","nasdaq":"NASDAQCOM"}
for k,v in FRED.items(): curl(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={v}",D+f"f_{v}.csv")
for c in ("oficial","mayorista","contadoconliqui","bolsa","blue"): curl(f"https://api.argentinadatos.com/v1/cotizaciones/dolares/{c}",D+f"ad_{c}.json")
curl("https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais",D+"ad_riesgo.json")
BCRA={"reservas":1,"badlar":7,"base_monetaria":15,"tc_mayorista_bcra":5}
for k,v in BCRA.items(): curl(f"https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias/{v}?limit=3000",D+f"bcra_{v}.json",k=True)
def fred(v):
    d=pd.read_csv(D+f"f_{v}.csv"); d.columns=["date","v"]; d["date"]=pd.to_datetime(d["date"]); d["v"]=pd.to_numeric(d["v"],errors="coerce"); return d.dropna().set_index("date")["v"]
def ad(c,key="venta"):
    d=pd.DataFrame(json.load(open(D+f"ad_{c}.json"))); d["fecha"]=pd.to_datetime(d["fecha"]); return d.set_index("fecha")[key].astype(float).sort_index()
def bcra(v):
    det=json.load(open(D+f"bcra_{v}.json"))["results"][0]["detalle"]; return pd.Series({pd.Timestamp(x["fecha"]):x["valor"] for x in det}).sort_index()
def build():
    idx=pd.date_range("2016-01-01","2026-10-06",freq="D")
    S={}
    for k,v in FRED.items(): S[k]=fred(v)
    for c in ("oficial","mayorista","contadoconliqui","bolsa","blue"): S["ar_"+c]=ad(c)
    S["riesgo_pais"]=ad("riesgo","valor")
    for k,v in BCRA.items(): S[k]=bcra(v)
    E=pd.DataFrame({k:s.reindex(idx.union(s.index)).ffill().reindex(idx) for k,s in S.items()})
    return E
if __name__=="__main__":
    E=build(); E.to_pickle(D+"E.pkl"); print(E.shape); print(E.dropna().index[[0,-1]]); print(E.isna().mean().round(3).to_string())
