"""Dual-polarization periodic-block Manakov SSFM, all waveforms synthetic."""
from pathlib import Path
import numpy as np
from scipy.fft import fft,ifft,fftfreq
ROOT=Path(__file__).resolve().parents[1]
CFG=dict(symbol_rate=32e9,sps=2,n_symbols=2048,spans=8,span_km=80.,alpha_db_km=.2,beta2_ps2_km=-21.7,gamma_W_km=1.3,noise_figure_db=5.,wavelength_m=1550e-9,rolloff=.1,forward_steps=16,window_half=4,pilot_spacing=16,guard=128)
LEVEL=np.array([-3,-1,1,3])/np.sqrt(10);GRAY=np.array([[0,0],[0,1],[1,1],[1,0]],int)

def modulation(rng,N=None):
 N=CFG['n_symbols'] if N is None else N;codes=rng.integers(0,4,(2,N,2));x=LEVEL[codes[:,:,0]]+1j*LEVEL[codes[:,:,1]];bits=GRAY[codes].reshape(2,N,4);return x,bits

def rrc_filter(N,sps,rolloff):
 f=fftfreq(N,d=1/sps);a=np.abs(f);p=np.zeros(N);p[a<=(1-rolloff)/2]=1
 m=(a>(1-rolloff)/2)&(a<=(1+rolloff)/2);p[m]=.5*(1+np.cos(np.pi/rolloff*(a[m]-(1-rolloff)/2)))
 return np.sqrt(sps*p)

def transmit(x,power_dbm):
 sps=CFG['sps'];N=x.shape[1]*sps;up=np.zeros((2,N),complex);up[:,::sps]=x;h=rrc_filter(N,sps,CFG['rolloff']);pulse=ifft(fft(up,axis=-1)*h,axis=-1)
 # E|x_p|^2=1 and root-Nyquist pulse energy gives mean waveform power=1/pol.
 power=1e-3*10**(power_dbm/10);return pulse*np.sqrt(CFG['sps'])*np.sqrt(power/2),h

def propagate(a,rng,steps=None,spans=None,gamma_scale=1.,beta_scale=1.,noise=True):
 steps=CFG['forward_steps'] if steps is None else steps;spans=CFG['spans'] if spans is None else spans;a=a.copy();alpha=CFG['alpha_db_km']*np.log(10)/10;dz=CFG['span_km']/steps;leff=(1-np.exp(-alpha*dz))/alpha
 w=2*np.pi*fftfreq(a.shape[-1],d=1/(CFG['symbol_rate']*CFG['sps']));beta=CFG['beta2_ps2_km']*1e-24*beta_scale
 half=np.exp(1j*beta*w*w*dz/4);gamma=8/9*CFG['gamma_W_km']*gamma_scale
 G=np.exp(alpha*CFG['span_km']);nf=10**(CFG['noise_figure_db']/10);h=6.62607015e-34;nu=299792458/CFG['wavelength_m'];ase=(nf/2)*h*nu*(G-1)*(CFG['symbol_rate']*CFG['sps'])
 for span in range(spans):
  for j in range(steps):
   a=ifft(fft(a,axis=-1)*half,axis=-1);a*=np.exp(1j*gamma*np.sum(np.abs(a)**2,axis=0)*leff)[None];a*=np.exp(-alpha*dz/2);a=ifft(fft(a,axis=-1)*half,axis=-1)
  a*=np.sqrt(G)
  if noise:a+=np.sqrt(ase/2)*(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))
 return a

def dbp(a,power_dbm,steps=1,spans=None):
 # Reverse each loss-normalized span. In normalized coordinates the reverse
 # nonlinear integral for each step is exp(-alpha*z_start)*Leff.
 spans=CFG['spans'] if spans is None else spans;a=a.copy();alpha=CFG['alpha_db_km']*np.log(10)/10;dz=CFG['span_km']/steps;leff=(1-np.exp(-alpha*dz))/alpha;gamma=8/9*CFG['gamma_W_km']
 w=2*np.pi*fftfreq(a.shape[-1],d=1/(CFG['symbol_rate']*CFG['sps']));beta=CFG['beta2_ps2_km']*1e-24;half=np.exp(-1j*beta*w*w*dz/4)
 for span in range(spans):
  for j in reversed(range(steps)):
   a=ifft(fft(a,axis=-1)*half,axis=-1);a*=np.exp(-1j*gamma*np.sum(np.abs(a)**2,axis=0)*np.exp(-alpha*j*dz)*leff)[None];a=ifft(fft(a,axis=-1)*half,axis=-1)
 return a

def receive(a,power_dbm,h,linear=True):
 a=a.copy()
 if linear:
  w=2*np.pi*fftfreq(a.shape[-1],d=1/(CFG['symbol_rate']*CFG['sps']));beta=CFG['beta2_ps2_km']*1e-24;a=ifft(fft(a,axis=-1)*np.exp(-1j*beta*w*w*CFG['span_km']*CFG['spans']/2),axis=-1)
 mf=ifft(fft(a,axis=-1)*h/np.sqrt(CFG['sps']),axis=-1)[:,::CFG['sps']]
 return mf/np.sqrt(1e-3*10**(power_dbm/10)/2)

def add_lo(a,rng,linewidth=0.,theta=0.,adc_bits=None):
 dt=1/(CFG['symbol_rate']*CFG['sps']);phi=theta+np.cumsum(rng.normal(size=a.shape[-1])*np.sqrt(2*np.pi*linewidth*dt));a=a*np.exp(1j*phi)[None]
 if adc_bits:
  # Fixed full-scale per waveform from receiver RMS; no true symbols used.
  scale=4*np.sqrt(np.mean(np.abs(a)**2)/2);step=2*scale/(2**adc_bits-1)
  a=(np.round(np.clip(a.real,-scale,scale)/step)+1j*np.round(np.clip(a.imag,-scale,scale)/step))*step
 return a

def windows(z):
 M=CFG['window_half'];return np.stack([np.roll(z,-l,axis=1).T for l in range(-M,M+1)],axis=1) # N,9,2

def canonical(w):
 # Largest-magnitude sample in the local two-polarization window anchors phase; fixed index order breaks ties.
 c=w.reshape(len(w),-1);j=np.argmax(np.abs(c),axis=1);a=c[np.arange(len(c)),j];mag=np.abs(a);eta=np.where(mag>1e-15,a/np.maximum(mag,1e-15),1.+0j)
 return w*eta.conj()[:,None,None],eta

def features(w,power,kind):
 if kind=='Canonical':v,eta=canonical(w)
 else:v=w;eta=np.ones(len(w),complex)
 X=np.c_[v.real.reshape(len(v),-1),v.imag.reshape(len(v),-1),np.full(len(v),power/6)]
 return X,eta

def polynomial_features(z,power):
 w=windows(z);energy=np.sum(np.abs(w)**2,axis=-1);p=10**(power/10)
 return np.c_[w.reshape(len(w),-1),(w*energy[:,:,None]*p).reshape(len(w),-1)]

def pilot_recover(y,x,bits):
 N=y.shape[1];g=CFG['guard'];pi=np.arange(g,N-g,CFG['pilot_spacing']);ph=np.unwrap(np.angle(np.sum(y[:,pi]*x[:,pi].conj(),axis=0)));interp=np.interp(np.arange(N),pi,ph);v=y*np.exp(-1j*interp)[None]
 gain=np.sum(v[:,pi]*x[:,pi].conj(),axis=1)/np.sum(np.abs(x[:,pi])**2,axis=1);v/=gain[:,None]
 keep=np.zeros(N,bool);keep[g:N-g]=True;keep[pi]=False;err=v[:,keep]-x[:,keep];mse=float(np.mean(np.abs(err)**2));es=float(np.mean(np.abs(x[:,keep])**2));snr=10*np.log10(es/max(mse,1e-30));d=v[:,keep];ir=np.argmin(np.abs(d.real[:,:,None]-LEVEL),axis=-1);ii=np.argmin(np.abs(d.imag[:,:,None]-LEVEL),axis=-1);bh=np.concatenate([GRAY[ir],GRAY[ii]],axis=-1);truebits=bits[:,keep];errors=int(np.sum(bh!=truebits));nbits=int(truebits.size)
 return dict(snr_db=float(snr),evm_percent=float(100*np.sqrt(mse/es)),ber=errors/nbits,errors=errors,bits=nbits),v
