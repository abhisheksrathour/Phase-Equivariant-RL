"""Scientific figures from executed CSV/NPZ records; 600 dpi plus SVG."""
from channel import ROOT
import json,numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib import font_manager as fm
from matplotlib.ticker import NullLocator,NullFormatter
fonts={x.name for x in fm.fontManager.ttflist};FONT='Times New Roman' if 'Times New Roman' in fonts else 'Liberation Serif'
plt.rcParams.update({'font.family':FONT,'font.size':10,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':8.5,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'figure.constrained_layout.use':True})
D=ROOT/'data';F=ROOT/'figures';F.mkdir(exist_ok=True)
S=pd.read_csv(D/'primary_summary.csv');CI=pd.read_csv(D/'bootstrap_intervals.csv');R=pd.read_csv(D/'primary_results.csv');DI=pd.read_csv(D/'diagnostic_results.csv');CO=pd.read_csv(D/'numerical_convergence.csv');BU=pd.read_csv(D/'data_budget_results.csv');OR=pd.read_csv(D/'phase_orbit.csv')
M=['CDC','Polynomial','RawMLP','AugmentedMLP','CanonicalNN','DBP1','DBP4'];N={'CDC':'Linear CDC','Polynomial':'Cubic ridge','RawMLP':'Raw MLP','AugmentedMLP':'Phase-augmented MLP','CanonicalNN':'Canonical residual NN','DBP1':'DBP 1 step/span','DBP4':'DBP 4 steps/span'}
C={m:plt.cm.viridis(v) for m,v in zip(M,[.12,.32,.45,.58,.75,.88,.02])};ST=['o','v','s','D','^','P','x'];LS=['--',':','-.','--','-',':','-']
metadata=[]
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=600,bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.svg'),bbox_inches='tight',facecolor='white');metadata.append(dict(name=name,font=FONT,png_dpi=600,size_inches=list(fig.get_size_inches())));plt.close(fig)
def box(ax,x,y,w,h,text,fc='#eaf3ef',fs=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.03',fc=fc,ec='#244b52',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arr(ax,a,b,label=None):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.05,color='#344b51'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.17,label,ha='center',fontsize=9)
def panel(ax,t):ax.set_title(t,loc='left',fontweight='bold');ax.grid(alpha=.18)
# 1 Pictorial link and offline twin.
fig,ax=plt.subplots(figsize=(7.0,3.5));ax.set(xlim=(0,10),ylim=(0,4.2));ax.axis('off')
box(ax,.1,2.65,1.55,1.0,'DP 16-QAM\n32 GBaud\nRRC pulses');arr(ax,(1.7,3.15),(2.05,3.15))
for j in range(3):
 x=2.05+j*1.27;t=np.linspace(0,2*np.pi,120);ax.plot(x+.43+.27*np.cos(t),3.15+.26*np.sin(2*t),color=C['CanonicalNN'],lw=1.6);ax.add_patch(Polygon([[x+.93,2.94],[x+1.15,3.15],[x+.93,3.36]],closed=True,fc='#d2dfb0',ec='#344b51'))
ax.text(3.85,2.5,'8 × 80 km fiber spans and EDFAs',ha='center');arr(ax,(5.82,3.15),(6.05,3.15));box(ax,6.08,2.65,1.4,1,'Coherent\nreceiver\nLO phase');arr(ax,(7.52,3.15),(7.83,3.15));box(ax,7.86,2.65,1.85,1,'CDC → learned\nresidual → pilots\n→ decisions',fs=9.4)
box(ax,.25,.6,2.2,1.05,'Configured digital twin\nManakov split-step\nSynthetic commissioning');box(ax,3.2,.6,2.2,1.05,'Phase-preserving labels\nLocal canonicalization\nOffline training');box(ax,6.35,.6,2.9,1.05,'Frozen equalizer\nWaveform-level evaluation\nNo live hardware connection')
arr(ax,(2.5,1.12),(3.15,1.12));arr(ax,(5.45,1.12),(6.3,1.12));arr(ax,(8.77,1.7),(8.77,2.6));ax.text(.2,4.02,'A  Simulated optical link',fontweight='bold');ax.text(.2,1.98,'B  Offline twin-to-equalizer workflow',fontweight='bold');save(fig,'fig01_link')
# 2 Architecture with a phase orbit as a pictorial inset.
fig,ax=plt.subplots(figsize=(7,3.6));ax.set(xlim=(0,10),ylim=(0,4.3));ax.axis('off')
box(ax,.12,2.65,1.75,1,'9-symbol\n2-polarization\ncomplex window');box(ax,2.27,2.65,1.65,1,'Largest magnitude\nanchor a\nη = a / |a|',fs=9.5);box(ax,4.32,2.65,1.65,1,'Rotate by η*\n36 real features\n+ launch power',fs=9.5);box(ax,6.37,2.65,1.45,1,'37 inputs\n48 → 24 hidden\n4 residual outputs',fs=9.0);box(ax,8.22,2.65,1.65,1,'Restore η\nAdd center sample\n2 complex outputs',fs=9.3)
for a,b in [(1.9,2.23),(3.95,4.28),(6.0,6.33),(7.85,8.18)]:arr(ax,(a,3.15),(b,3.15))
ax.text(.15,4.03,'A  Observable local phase canonicalization',fontweight='bold')
for x,label,phase in [(1.3,'Original window',.2),(4.2,'Shared phase rotation',1.1),(7.25,'Same canonical window',0.)]:
 ax.add_patch(Circle((x,1.23),.62,fill=False,ec='#90a6a6',lw=.9));ax.plot([x-.7,x+.7],[1.23,1.23],color='#b0b0b0',lw=.6);ax.plot([x,x],[.52,1.94],color='#b0b0b0',lw=.6)
 for mag,ang in [(.5,0),(.35,1.4),(.28,3.3),(.25,4.6)]:ax.scatter(x+mag*np.cos(ang+phase),1.23+mag*np.sin(ang+phase),c=[C['CanonicalNN']],s=16)
 ax.text(x,.2,label,ha='center',fontsize=9)
arr(ax,(2.05,1.23),(3.38,1.23),'Rotate by ψ');arr(ax,(4.95,1.23),(6.45,1.23),'Canonicalize');ax.text(.15,2.2,'B  Phase symmetry without a true-phase input',fontweight='bold');save(fig,'fig02_architecture')
# 3 Algorithm flowchart with branches and no text overlaps.
fig,ax=plt.subplots(figsize=(6.7,5.0));ax.set(xlim=(0,10),ylim=(0,8.6));ax.axis('off')
box(ax,3.3,7.6,3.4,.65,'Configured channel and disjoint seeds');box(ax,.25,6.05,4.1,.95,'Commissioning frames\nKnown training symbols and fixed pilots');box(ax,5.65,6.05,4.1,.95,'Held-out received waveforms\nKnown pilots only at inference');arr(ax,(4.3,7.55),(2.3,7.05));arr(ax,(5.7,7.55),(7.7,7.05))
box(ax,.25,4.45,4.1,.95,'Estimate shared pilot phase\nPreserve that phase in all training targets');box(ax,5.65,4.45,4.1,.95,'Linear CDC and matched filter\nExtract local dual-polarization windows');arr(ax,(2.3,6.0),(2.3,5.45));arr(ax,(7.7,6.0),(7.7,5.45))
box(ax,.25,2.85,4.1,.95,'Fit raw, augmented and canonical models\nChoose epochs on separate validation frames');box(ax,5.65,2.85,4.1,.95,'Canonicalize → frozen residual → restore\nAll-small window returns zero');arr(ax,(2.3,4.4),(2.3,3.85));arr(ax,(7.7,4.4),(7.7,3.85));arr(ax,(4.4,3.32),(5.6,3.32),'frozen weights')
box(ax,5.65,1.2,4.1,.95,'Pilot phase interpolation and gain fit\nExclude pilots and guards from metrics');arr(ax,(7.7,2.8),(7.7,2.2));box(ax,.25,1.2,4.1,.95,'Paired frame bootstrap and audits\nRetain weak gains and failed comparisons');arr(ax,(5.6,1.68),(4.4,1.68));ax.text(.25,.4,'Offline training',fontweight='bold');ax.text(5.65,.4,'Frozen inference and evaluation',fontweight='bold');save(fig,'fig03_flowchart')
# 4 Primary power curves; zero errors are shown separately, not assigned BER=0 on log.
fig,axs=plt.subplots(1,2,figsize=(7,4.05))
for m,mark,ls in zip(M,ST,LS):
 d=S[(S.case=='PhaseOffset')&(S.method==m)].sort_values('power_dbm');axs[0].plot(d.power_dbm,d.mean_snr_db,label=N[m],color=C[m],marker=mark,ms=3,lw=1.2,ls=ls)
 non=d.ber>0;axs[1].semilogy(d.power_dbm[non],d.ber[non],label=N[m],color=C[m],marker=mark,ms=3,lw=1.2,ls=ls)
 if (~non).any():axs[1].scatter(d.power_dbm[~non],np.full((~non).sum(),1/d.bits.iloc[0]),color=C[m],marker='v',s=24)
for ax in axs:ax.set_xlabel('Total launch power (dBm)')
axs[0].set_ylabel('EVM-derived SNR (dB)');axs[1].set_ylabel('Empirical payload BER');panel(axs[0],'A  Shared random phase');panel(axs[1],'B  Direct bit-error counting');fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,ncol=3,loc='outside lower center',fontsize=8);save(fig,'fig04_primary')
# 5 Symmetry audit and paired differences.
fig,axs=plt.subplots(1,2,figsize=(7,3.3))
for m in ['Polynomial','RawMLP','AugmentedMLP','CanonicalNN']:
 d=OR[OR.method==m];axs[0].semilogy(d.phase_rad/np.pi,np.maximum(d.relative_equivariance_error,1e-17),label=N[m],color=C[m],lw=1.2)
axs[0].set(xlabel='Applied shared phase / π',ylabel='Relative equivariance error');panel(axs[0],'A  Full common-phase orbit');axs[0].legend(frameon=False,fontsize=7.7)
P=pd.read_csv(D/'paired_snr_differences.csv');d=P[(P.case=='PhaseOffset')&(P.power_dbm==4)];bases=['CDC','Polynomial','RawMLP','AugmentedMLP','DBP1']
q=d.set_index('comparison').loc[['CanonicalNN_vs_'+m for m in bases]];v=q['mean'].values;axs[1].errorbar(v,np.arange(len(q)),xerr=np.array([v-q.low.values,q.high.values-v]),fmt='o',color=C['CanonicalNN'],capsize=3);axs[1].set_yticks(np.arange(len(q)),[N[m] for m in bases]);axs[1].axvline(0,color='#666666',lw=.8);axs[1].set_xlabel('Canonical NN minus comparator (dB)');panel(axs[1],'B  Paired 95% bootstrap intervals');save(fig,'fig05_symmetry')
# 6 Extra linewidth and pilot diagnostics.
fig,axs=plt.subplots(1,2,figsize=(7,4.05))
for m,mark,ls in zip(M,ST,LS):
 d=DI[(DI.experiment=='linewidth')&(DI.method==m)].groupby('value').snr_db.mean();axs[0].plot(d.index/1e6,d.values,label=N[m],marker=mark,ms=3,color=C[m],ls=ls,lw=1.1)
 d=DI[(DI.experiment=='pilot_spacing')&(DI.method==m)].groupby('value').snr_db.mean();axs[1].plot(100/d.index,d.values,marker=mark,ms=3,color=C[m],ls=ls,lw=1.1)
axs[0].set(xlabel='Receiver LO linewidth (MHz)',ylabel='EVM-derived SNR (dB)');axs[1].set(xlabel='Pilot share within retained symbols (%)',ylabel='EVM-derived SNR (dB)');panel(axs[0],'A  Time-varying phase at 4 dBm');panel(axs[1],'B  Pilot overhead at 100 kHz');fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,ncol=3,loc='outside lower center',fontsize=8);save(fig,'fig06_diagnostics')
# 7 Physical mismatch and high linewidth.
fig,axs=plt.subplots(1,2,figsize=(7,4.05))
for ax,case,title in zip(axs,['Wiener1M','GammaShift'],['A  Receiver LO linewidth 1 MHz','B  Nonlinearity coefficient +20%']):
 for m,mark,ls in zip(M,ST,LS):
  d=S[(S.case==case)&(S.method==m)];ax.plot(d.power_dbm,d.mean_snr_db,label=N[m],color=C[m],marker=mark,ms=3,ls=ls,lw=1.1)
 ax.set(xlabel='Total launch power (dBm)',ylabel='EVM-derived SNR (dB)');panel(ax,title)
fig.legend(*axs[1].get_legend_handles_labels(),frameon=False,ncol=3,loc='outside lower center',fontsize=8);save(fig,'fig07_shift')
# 8 Data budget and numerical grid convergence.
fig,axs=plt.subplots(1,2,figsize=(7,3.3))
for m,mark in [('RawMLP','s'),('AugmentedMLP','D'),('CanonicalNN','^')]:
 d=BU[BU.method==m].groupby('training_frames').snr_db.mean();axs[0].plot(d.index,d.values,label=N[m],color=C[m],marker=mark,lw=1.2)
axs[0].set(xlabel='Commissioning frames',ylabel='Mean EVM-derived SNR (dB)');panel(axs[0],'A  One fixed initialization per fit');axs[0].legend(frameon=False,fontsize=7.6)
for power,c in [(4,C['CanonicalNN']),(8,C['DBP4'])]:
 d=CO[(CO.power_dbm==power)&(CO.steps<64)].groupby('steps').relative_field_error.mean();axs[1].loglog(d.index,d.values,'o-',label=f'{power} dBm',color=c)
axs[1].set(xlabel='Forward steps per span',ylabel='Relative field error vs 64-step reference');axs[1].set_xticks([8,16,32],['8','16','32']);axs[1].xaxis.set_minor_locator(NullLocator());axs[1].xaxis.set_minor_formatter(NullFormatter());panel(axs[1],'B  Noiseless spatial refinement');axs[1].legend(frameon=False);save(fig,'fig08_budget_grid')
# 9 Actual post-recovery constellation from a fixed primary frame.
Z=np.load(D/'constellation_snapshots.npz');fig,axs=plt.subplots(2,2,figsize=(6.6,5.3));target=Z['PhaseOffset_8_target'][0];keep=np.arange(128,1920);keep=keep[(keep-128)%16!=0]
for ax,m,lab in zip(axs.flat,['CDC','RawMLP','CanonicalNN','DBP4'],['A','B','C','D']):
 v=Z['PhaseOffset_8_'+m][0,keep];ax.scatter(v.real,v.imag,s=1.1,color=C[m],alpha=.36,rasterized=True);ax.scatter(np.unique(target).real,np.unique(target).imag,s=14,facecolors='none',edgecolors='black',marker='s',lw=.6);ax.set(xlim=(-1.55,1.55),ylim=(-1.55,1.55),xlabel='In-phase component',ylabel='Quadrature component');ax.set_aspect('equal');panel(ax,f'{lab}  {N[m]}')
save(fig,'fig09_constellations');(F/'figure_metadata.json').write_text(json.dumps(metadata,indent=2));print('Created',len(metadata),'figures; font',FONT)
if __name__=='__main__':pass
