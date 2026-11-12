"""Frozen exploratory diagnostics; independent seeds, no primary-test tuning."""
from channel import *
from train import infer_net,infer_poly,fit_net
import numpy as np,pandas as pd,joblib,json,time
METHODS=['CDC','Polynomial','RawMLP','AugmentedMLP','CanonicalNN','DBP1','DBP4']
def predictions(a,power,h,models,db=True):
 z=receive(a,power,h);out={'CDC':z,'Polynomial':infer_poly(z,power,models['Polynomial'])}
 for k,n in [('Raw','RawMLP'),('Augmented','AugmentedMLP'),('Canonical','CanonicalNN')]:out[n]=infer_net(z,power,models[k])[0]
 if db:
  for s in [1,4]:out['DBP'+str(s)]=receive(dbp(a,power,s),power,h,linear=False)
 return out

def main():
 t=time.time();models=joblib.load(ROOT/'data/equalizers.joblib');rows=[]
 for seed in range(3000,3012):
  rng=np.random.default_rng(seed);x,b=modulation(rng);a,h=transmit(x,4);out=propagate(a,rng)
  for lw in [0,1e4,1e5,5e5,1e6,2e6]:
   a=add_lo(out,np.random.default_rng(seed+900000),linewidth=lw,theta=.83)
   for name,y in predictions(a,4,h,models).items():
    m,_=pilot_recover(y,x,b);rows.append(dict(experiment='linewidth',value=lw,seed=seed,method=name,**m))
 for seed in range(3100,3112):
  rng=np.random.default_rng(seed);x,b=modulation(rng);a,h=transmit(x,4);out=propagate(a,rng);a=add_lo(out,np.random.default_rng(seed+900000),linewidth=1e5,theta=.83);pred=predictions(a,4,h,models)
  for spacing in [8,16,32,64]:
   CFG['pilot_spacing']=spacing
   for name,y in pred.items():
    m,_=pilot_recover(y,x,b);rows.append(dict(experiment='pilot_spacing',value=spacing,seed=seed,method=name,**m))
 CFG['pilot_spacing']=16;pd.DataFrame(rows).to_csv(ROOT/'data/diagnostic_results.csv',index=False)
 print('linewidth and pilots done',flush=True)
 # Data-budget study: same nested commissioning frames, one newly fixed initialization.
 tr=dict(np.load(ROOT/'data/training_windows.npz'));va=dict(np.load(ROOT/'data/validation_windows.npz'));bud=[];rep=[]
 tests=[]
 for seed in range(3200,3212):
  rng=np.random.default_rng(seed);x,b=modulation(rng);a,h=transmit(x,4);out=propagate(a,rng);tests.append((seed,x,b,receive(add_lo(out,np.random.default_rng(seed+900000),linewidth=0,theta=1.17),4,h)))
 for n in [20,50,100]:
  keep=tr['frame']<2000+n;sub={k:v[keep] for k,v in tr.items()}
  for kind,name in [('Raw','RawMLP'),('Augmented','AugmentedMLP'),('Canonical','CanonicalNN')]:
   ob,re=fit_net(sub,va,kind,91);rep.append(dict(frames=n,**re))
   for seed,x,b,z in tests:
    m,_=pilot_recover(infer_net(z,4,[ob])[0],x,b);bud.append(dict(training_frames=n,training_examples=len(sub['w']),seed=seed,method=name,**m))
 pd.DataFrame(bud).to_csv(ROOT/'data/data_budget_results.csv',index=False);(ROOT/'data/data_budget_training.json').write_text(json.dumps(rep,indent=2));print('data budgets done',flush=True)
 # Numerical resolution. The nonlinear reference is 64 steps per span.
 conv=[]
 for power in [4,8]:
  for seed in range(3300,3306):
   rng=np.random.default_rng(seed);x,b=modulation(rng);a,h=transmit(x,power);ref=propagate(a,rng,steps=64,noise=False)
   for steps in [8,16,32,64]:
    one=propagate(a,rng,steps=steps,noise=False);err=float(np.linalg.norm(one-ref)/np.linalg.norm(ref));conv.append(dict(power_dbm=power,seed=seed,steps=steps,relative_field_error=err))
 pd.DataFrame(conv).to_csv(ROOT/'data/numerical_convergence.csv',index=False)
 # Mathematical consistency audits. Also retain the full phase orbit diagnostic.
 rng=np.random.default_rng(3400);x,b=modulation(rng);a,h=transmit(x,4);out=propagate(a,rng);z=receive(out,4,h);rr,mm=pilot_recover(receive(a,4,h,linear=False),x,b)
 checks={'zero_channel_evm_percent':rr['evm_percent'],'zero_channel_bit_errors':rr['errors']}
 back=dbp(propagate(a,rng,steps=16,noise=False),4,steps=16);checks['same_grid_inverse_relative_error']=float(np.linalg.norm(back-a)/np.linalg.norm(a))
 orbit=[];base=predictions(out,4,h,models,db=False)
 for phase in np.linspace(0,2*np.pi,73):
  rot=np.exp(1j*phase);new=predictions(out*rot,4,h,models,db=False)
  for name in base:
   err=float(np.linalg.norm(new[name]-rot*base[name])/np.linalg.norm(base[name]));orbit.append(dict(phase_rad=float(phase),method=name,relative_equivariance_error=err))
 pd.DataFrame(orbit).to_csv(ROOT/'data/phase_orbit.csv',index=False)
 checks['max_equivariance_error']={m:float(max(r['relative_equivariance_error'] for r in orbit if r['method']==m)) for m in base}
 checks['elapsed_seconds']=time.time()-t;(ROOT/'data/diagnostic_audits.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
