"""Post-review numerical sensitivity, frozen estimator on fresh noisy frames."""
from channel import *
from diagnostics import predictions
import joblib,pandas as pd,numpy as np
models=joblib.load(ROOT/'data/equalizers.joblib');rows=[]
for power in [4,8]:
 for seed in range(3500,3512):
  rng=np.random.default_rng(seed);x,b=modulation(rng);a,h=transmit(x,power)
  for steps in [16,64]:
   out=propagate(a,np.random.default_rng(seed+950000),steps=steps);out=add_lo(out,np.random.default_rng(seed+960000),linewidth=0,theta=.83)
   for name,y in predictions(out,power,h,models).items():
    m,_=pilot_recover(y,x,b);rows.append(dict(steps=steps,power_dbm=power,seed=seed,method=name,**m))
pd.DataFrame(rows).to_csv(ROOT/'data/grid_transfer_results.csv',index=False)
print(pd.DataFrame(rows).groupby(['power_dbm','steps','method']).snr_db.mean().to_string())
