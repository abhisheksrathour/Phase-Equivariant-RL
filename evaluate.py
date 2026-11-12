from channel import *
from train import infer_net,infer_poly
import numpy as np,pandas as pd,json,joblib,time,sys
from scipy.stats import bootstrap
POWERS=[-8,-6,-4,-2,0,2,4,6,8]
CASES=['Aligned','PhaseOffset','Wiener100k','Wiener1M','GammaShift']
METHODS=['CDC','Polynomial','RawMLP','AugmentedMLP','CanonicalNN','DBP1','DBP4']

def main():
 start=time.time();models=joblib.load(ROOT/'data/equalizers.joblib');rows=[];member=[];waves=[];targets=[];codes=[];meta=[];snaps={}
 for power in POWERS:
  for seed in range(1200,1224):
   rng=np.random.default_rng(seed);x,bits=modulation(rng);a,h=transmit(x,power);out=propagate(a,rng);rg=np.random.default_rng(seed);xg,bg=modulation(rg);ag,hg=transmit(xg,power);shift=propagate(ag,rg,gamma_scale=1.2)
   waves.append(np.stack([out,shift]));targets.append(x);codes.append(bits);meta.append([power,seed]);theta=np.random.default_rng(500000+seed+power*100).uniform(-np.pi,np.pi)
   for case in CASES:
    line=0. if case in ['Aligned','PhaseOffset'] else 1e6 if case=='Wiener1M' else 1e5
    received=add_lo(shift if case=='GammaShift' else out,np.random.default_rng(600000+seed+power*100),linewidth=line,theta=0. if case=='Aligned' else theta)
    z=receive(received,power,h);pred={'CDC':z,'Polynomial':infer_poly(z,power,models['Polynomial'])}
    for kind,name in [('Raw','RawMLP'),('Augmented','AugmentedMLP'),('Canonical','CanonicalNN')]:
     yp,ym=infer_net(z,power,models[kind]);pred[name]=yp
     for k,one in enumerate(ym):
      met,_=pilot_recover(one,x,bits);member.append(dict(case=case,power_dbm=power,seed=seed,method=name,member_seed=[41,42,43][k],**met))
    for steps in [1,4]:pred['DBP'+str(steps)]=receive(dbp(received,power,steps),power,h,linear=False)
    for method,y in pred.items():
     met,v=pilot_recover(y,x,bits);rows.append(dict(case=case,power_dbm=power,seed=seed,method=method,theta=0. if case=='Aligned' else theta,linewidth_Hz=line,**met))
     if seed==1200 and power in [4,8] and case in ['PhaseOffset','Wiener1M']:
      snaps[f'{case}_{power}_{method}']=v;snaps[f'{case}_{power}_target']=x
  print('power',power,'completed',flush=True)
 df=pd.DataFrame(rows);df.to_csv(ROOT/'data/primary_results.csv',index=False);pd.DataFrame(member).to_csv(ROOT/'data/retraining_results.csv',index=False)
 np.savez_compressed(ROOT/'data/primary_waveforms.npz',received=np.stack(waves),symbols=np.stack(targets),bits=np.stack(codes),power_seed=np.array(meta));np.savez_compressed(ROOT/'data/constellation_snapshots.npz',**snaps)
 summary=df.groupby(['case','power_dbm','method'],sort=False).agg(mean_snr_db=('snr_db','mean'),mean_evm_percent=('evm_percent','mean'),errors=('errors','sum'),bits=('bits','sum'),frames=('seed','size')).reset_index();summary['ber']=summary.errors/summary.bits;summary.to_csv(ROOT/'data/primary_summary.csv',index=False)
 # Paired bootstrap over the 24 independent physical-frame seeds, not over bits.
 rng=np.random.default_rng(777);ints=[];diffs=[]
 for (case,power),d in df.groupby(['case','power_dbm'],sort=False):
  draw=rng.integers(0,24,(4000,24));s={}
  for method in METHODS:
   z=d[d.method==method].sort_values('seed');s[method]=z.snr_db.to_numpy()[draw].mean(1)
   for metric,col in [('snr_db','snr_db'),('ber','ber')]:
    v=z[col].to_numpy()[draw].mean(1);lo,hi=np.quantile(v,[.025,.975]);ints.append(dict(case=case,power_dbm=power,method=method,metric=metric,low=lo,high=hi))
  for base in ['CDC','Polynomial','RawMLP','AugmentedMLP','DBP1','DBP4']:
   v=s['CanonicalNN']-s[base];lo,hi=np.quantile(v,[.025,.975]);mean=float(d[d.method=='CanonicalNN'].snr_db.mean()-d[d.method==base].snr_db.mean());diffs.append(dict(case=case,power_dbm=power,comparison='CanonicalNN_vs_'+base,mean=mean,low=lo,high=hi))
 pd.DataFrame(ints).to_csv(ROOT/'data/bootstrap_intervals.csv',index=False);pd.DataFrame(diffs).to_csv(ROOT/'data/paired_snr_differences.csv',index=False)
 (ROOT/'data/execution_report.json').write_text(json.dumps(dict(config=CFG,primary_rows=len(df),retraining_rows=len(member),physical_waveform_frames=len(waves)*2,test_seeds=list(range(1200,1224)),powers_dbm=POWERS,cases=CASES,methods=METHODS,payload_bits_per_frame=int(df.bits.iloc[0]),elapsed_seconds=time.time()-start,python=sys.version,numpy=np.__version__),indent=2));print('finished',time.time()-start,flush=True)
if __name__=='__main__':main()
