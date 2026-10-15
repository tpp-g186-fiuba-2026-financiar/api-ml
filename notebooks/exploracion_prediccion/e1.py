from ev import *
import sys
T=sys.argv[1]
g=G
sets={"tech":g["tech"],"tech+mkt":g["tech"]+g["mkt"],"tech+mkt+fx":g["tech"]+g["mkt"]+g["fx"],
"tech+mkt+fx+mac":g["tech"]+g["mkt"]+g["fx"]+g["mac"],"+glob":g["tech"]+g["mkt"]+g["fx"]+g["mac"]+g["glob"],
"all(+adr)":sum(g.values(),[]),"mac+fx only":g["mac"]+g["fx"],"mac+fx+mkt":g["mac"]+g["fx"]+g["mkt"]}
print("TARGET",T)
for k,c in sets.items():
    report(k,run(c,target=T,kind="up"),quick=True)
