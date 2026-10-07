import os, warnings, sys
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0,R:='/private/tmp/claude-501/-Users-matimorales01-Desktop-tp-profesional/86c17208-1581-462a-bd90-b25530b67cdc/scratchpad/research')
from ablate import wf_auc, ALL
warnings.filterwarnings("ignore")
LEVELS=["a_rp","a_badlar","a_brecha","a_mep_ccl","a_blue_gap","a_real_rate","g_vix","g_vxeem","g_curve"]
CAL=["e_month","e_dow"]
V={"ALL":ALL,
   "sin mes/dia-semana":[c for c in ALL if c not in CAL],
   "sin niveles (solo cambios y z-scores)":[c for c in ALL if c not in LEVELS],
   "sin niveles y sin mes/dia":[c for c in ALL if c not in LEVELS+CAL],
   "NUCLEO: cambios macro AR+global + tecnicas base":[c for c in ALL if (c.startswith(("a_","g_")) and c not in LEVELS) or c in ["t_ret1","t_ret10","t_rsi14","t_sma20","t_macd","t_boll","t_gk20","t_dd60","t_volz","m_ret5","m_ret20","m_ret60","m_vol20","s_rel20"]]}
def job(h,n): return dict(h=h,variante=n,n_features=len(V[n]),auc=round(wf_auc(h,V[n]),3))
if __name__=="__main__":
    res=Parallel(n_jobs=9)(delayed(job)(h,n) for h in (20,30,40) for n in V)
    d=pd.DataFrame(res); pd.set_option("display.width",250); print(d.pivot(index="variante",columns="h",values="auc").to_string())
