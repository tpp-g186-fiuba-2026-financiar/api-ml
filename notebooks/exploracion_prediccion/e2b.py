from ev import *
g=G; TM=g["tech"]+g["mkt"]
report("base hgb h20",run(TM,"y20"),quick=True)
print("== labels @h=20")
report("rel label (>median)",run(TM,"y20",kind="ex"),quick=True)
report("usd label",run(TM+g["fx"],"yu20"),quick=True)
print("== h=5")
report("h5 base",run(TM,"y5"),quick=True)
report("h5 +fx+mac",run(TM+g["fx"]+g["mac"],"y5"),quick=True)
report("h5 xs-rank",run(TM,"y5",rank=True),quick=True)
print("== h=10")
P["y10"]=P.groupby("ticker").y5.shift(-5)+P.y5 # aprox 10d
report("h10 base",run(TM,"y10",h=10),quick=True)
