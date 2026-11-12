from channel import *
import numpy as np,pandas as pd,joblib,json,copy,time,warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore',category=ConvergenceWarning)
KINDS=['Raw','Augmented','Canonical']

def make_data(start,count,perframe):
 ww=[];xx=[];pp=[];ids=[]
 for i,seed in enumerate(range(start,start+count)):
  rng=np.random.default_rng(seed);power=[-6,-3,0,3,6][i%5];x,bits=modulation(rng);a,h=transmit(x,power);out=propagate(a,rng);z=receive(out,power,h);pi=np.arange(CFG['guard'],CFG['n_symbols']-CFG['guard'],CFG['pilot_spacing']);phase=float(np.angle(np.sum(z[:,pi]*x[:,pi].conj())));aligned=x*np.exp(1j*phase);w=windows(z);idx=rng.choice(np.arange(CFG['guard'],CFG['n_symbols']-CFG['guard']),perframe,replace=False)
  ww.append(w[idx]);xx.append(aligned[:,idx].T);pp.extend([power]*len(idx));ids.extend([seed]*len(idx))
 return dict(w=np.concatenate(ww),target=np.concatenate(xx),power=np.array(pp),frame=np.array(ids))

def make_xy(d,kind,seed,validation=False):
 w=d['w'].copy();target=d['target'].copy();rng=np.random.default_rng(seed)
 if kind=='Augmented' or validation:
  phase=rng.uniform(-np.pi,np.pi,len(w))
  if validation:phase[::2]=0
  rot=np.exp(1j*phase);w*=rot[:,None,None];target*=rot[:,None]
 if kind=='Canonical':v,eta=canonical(w)
 else:v=w;eta=np.ones(len(w),complex)
 X=np.c_[v.real.reshape(len(v),-1),v.imag.reshape(len(v),-1),d['power']/6]
 res=(target-w[:,CFG['window_half'],:])*eta.conj()[:,None];Y=np.c_[res.real,res.imag];return X,Y

def fit_net(tr,va,kind,seed):
 X,Y=make_xy(tr,kind,900+seed);XV,YV=make_xy(va,kind,990,True);sx=StandardScaler().fit(X);sy=StandardScaler().fit(Y);X=sx.transform(X);Y=sy.transform(Y);XV=sx.transform(XV);YV=sy.transform(YV)
 m=MLPRegressor(hidden_layer_sizes=(48,24),activation='tanh',solver='adam',alpha=.001,batch_size=256,learning_rate_init=.001,random_state=seed,shuffle=True,max_iter=1,early_stopping=False)
 best=np.inf;stall=0;hist=[];chosen=None
 for epoch in range(80):
  m.partial_fit(X,Y);loss=float(np.mean((m.predict(XV)-YV)**2));hist.append(loss)
  if loss<best-1e-5:best=loss;chosen=copy.deepcopy(m);stall=0;best_epoch=epoch+1
  else:stall+=1
  if stall>=10:break
 return dict(kind=kind,seed=seed,model=chosen,sx=sx,sy=sy),dict(kind=kind,seed=seed,epochs=len(hist),best_epoch=best_epoch,validation_standardized_mse=best,history=hist)

def infer_net(z,power,group):
 w=windows(z);kind=group[0]['kind'];X,eta=features(w,power,kind);outs=[]
 for obj in group:
  v=obj['sy'].inverse_transform(obj['model'].predict(obj['sx'].transform(X)));r=v[:,:2]+1j*v[:,2:];y=z.T+eta[:,None]*r
  if kind=='Canonical':y[np.max(np.abs(w),axis=(1,2))<=1e-15]=0
  outs.append(y.T)
 return np.mean(outs,axis=0),outs

def fit_poly(tr,va):
 def ff(d):
  w=d['w'];en=np.sum(np.abs(w)**2,axis=-1);return np.c_[w.reshape(len(w),-1),(w*en[:,:,None]*10**(d['power']/10)[:,None,None]).reshape(len(w),-1)]
 X=ff(tr);Y=tr['target']-tr['w'][:,CFG['window_half'],:];XV=ff(va);YV=va['target']-va['w'][:,CFG['window_half'],:];scale=np.sqrt(np.mean(np.abs(X)**2,axis=0));scale=np.maximum(scale,1e-8);X=X/scale;XV=XV/scale;best=np.inf
 for lam in [1e-5,1e-4,1e-3,.01,.1,1.]:
  c=np.linalg.solve(X.conj().T@X/len(X)+lam*np.eye(X.shape[1]),X.conj().T@Y/len(X));loss=float(np.mean(np.abs(XV@c-YV)**2))
  if loss<best:best=loss;ans=dict(coef=c,scale=scale,lambda_=lam,validation_mse=best)
 return ans

def infer_poly(z,power,poly):return z+(polynomial_features(z,power)/poly['scale']@poly['coef']).T

def main():
 t=time.time();tr=make_data(2000,100,384);va=make_data(2200,20,512);np.savez_compressed(ROOT/'data/training_windows.npz',**tr);np.savez_compressed(ROOT/'data/validation_windows.npz',**va);models={};reports=[]
 for kind in KINDS:
  models[kind]=[]
  for seed in [41,42,43]:
   ob,rep=fit_net(tr,va,kind,seed);models[kind].append(ob);reports.append(rep);print(kind,seed,'epochs',rep['epochs'],'best',rep['best_epoch'],flush=True)
 models['Polynomial']=fit_poly(tr,va);joblib.dump(models,ROOT/'data/equalizers.joblib');(ROOT/'data/training_report.json').write_text(json.dumps(dict(records=reports,training_examples=len(tr['w']),training_frames=100,validation_frames=20,polynomial_lambda=models['Polynomial']['lambda_'],elapsed_seconds=time.time()-t),indent=2));print('Training finished',time.time()-t,flush=True)
if __name__=='__main__':main()
