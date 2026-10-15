import requests, pandas as pd, time, sys, os
H={"User-Agent":"Mozilla/5.0"}
AR="GGAL BMA YPFD PAMP TXAR ALUA CRES EDN TECO2 TGSU2 TGNO4 CEPU COME LOMA MIRG SUPV VALO BYMA CVH HARG TRAN METR BBAR BPAT MOLI AGRO BHIP IRSA CELU SEMI DGCU2 LEDE OEST RICH GCLA CAPX".split()
US="GGAL BMA YPF PAM TEO EDN CEPU CRESY LOMA SUPV TGS BBAR IRS VIST SPY EEM EWZ ^VIX DX-Y.NYB ^TNX CL=F GC=F ARGT ^MERV ^GSPC BRL=X".split()
def get(sym):
    f=f"d/{sym.replace('^','_').replace('=','_')}.csv"
    if os.path.exists(f): return
    for i in range(3):
        r=requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",params={"period1":1262304000,"period2":1792000000,"interval":"1d"},headers=H,timeout=30)
        if r.status_code==200: break
        time.sleep(3)
    else:
        print("FAIL",sym,r.status_code); return
    j=r.json()["chart"]["result"][0]; q=j["indicators"]["quote"][0]
    df=pd.DataFrame({"date":pd.to_datetime(j["timestamp"],unit="s").normalize(),"open":q["open"],"high":q["high"],"low":q["low"],"close":q["close"],"volume":q["volume"]})
    adj=j["indicators"].get("adjclose")
    if adj: df["adjclose"]=adj[0]["adjclose"]
    df.dropna(subset=["close"]).to_csv(f,index=False); print("ok",sym,len(df),df.date.min().date())
    time.sleep(0.4)
for s in [a+".BA" for a in AR]+US: get(s)
