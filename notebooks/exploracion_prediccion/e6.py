exec(open("e5.py").read().split('print("== efecto')[0])
P["has_adr"]=P.adr_gap.notna().astype(float)
# liquidez: ranking por fecha de valor operado mediano (aprox con volumen*precio no esta en P): usar ticker fijo de lideres
LIQ=["GGAL","BMA","YPFD","PAMP","TXAR","ALUA","CRES","EDN","TECO2","TGSU2","CEPU","COME","LOMA","BYMA","BBAR","SUPV","TGNO4","VALO","MIRG","CVH"]
print("== has_adr dummy (identidad) y subuniverso liquido")
rep_ic("tech+mkt",run_rel2(TM,"y5",5))
rep_ic("tech+mkt+has_adr",run_rel2(TM+["has_adr"],"y5",5))
rep_ic("tech+adr",run_rel2(g["tech"]+g["adr"],"y5",5))
Pfull=P
def sub(o): return o[o.ticker.isin(LIQ)]
o=run_rel2(TM,"y5",5); rep_ic("tech+mkt | evaluado solo liquidas",sub(o))
o=run_rel2(g["tech"]+g["adr"],"y5",5); rep_ic("tech+adr | evaluado solo liquidas",sub(o))
# ADR-pair solo
o=run_rel2(g["tech"]+g["adr"],"y5",5); rep_ic("tech+adr | solo con ADR",o[o.ticker.isin(feat.ADR)])
o=run_rel2(TM,"y5",5); rep_ic("tech+mkt | solo con ADR",o[o.ticker.isin(feat.ADR)])
o=run_rel2(TM,"y5",5); rep_ic("tech+mkt | solo SIN ADR",o[~o.ticker.isin(feat.ADR)])
