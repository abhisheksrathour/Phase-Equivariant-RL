from channel import *
import json
rng=np.random.default_rng(10);x,bits=modulation(rng);rows=[]
for power in [-8,-4,0,4,8]:
 a,h=transmit(x,power);out=propagate(a,rng)
 for k in [0,1,4]:
  yy=receive(out,power,h) if k==0 else receive(dbp(out,power,k),power,h,linear=False);m,_=pilot_recover(yy,x,bits);rows.append(dict(power=power,dbp_steps=k,**m));print(rows[-1],flush=True)
# Round-trip deterministic check with equal SSFM step counts.
a,h=transmit(x,0);out=propagate(a,rng,noise=False);back=dbp(out,0,16);norm=float(np.linalg.norm(back-a)/np.linalg.norm(a));print('noise-free inverse relative error',norm,flush=True)
(ROOT/'data/development_check.json').write_text(json.dumps({'rows':rows,'noisefree_inverse_error':norm},indent=2))
