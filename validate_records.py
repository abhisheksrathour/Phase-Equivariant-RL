"""Audit data partitions, model dimensions, physical normalization and artifacts."""
from channel import *
from train import infer_net
import json,joblib,pandas as pd,zipfile,re,hashlib
from lxml import etree
from PIL import Image
from importlib.metadata import version

def main():
 d=ROOT/'data';tr=np.load(d/'training_windows.npz');va=np.load(d/'validation_windows.npz');r=pd.read_csv(d/'primary_results.csv');models=joblib.load(d/'equalizers.joblib')
 assert len(r)==7560 and not r.duplicated(['case','power_dbm','seed','method']).any()
 assert set(r.seed)==set(range(1200,1224));assert len(tr['w'])==38400 and len(va['w'])==10240
 assert not (set(tr['frame'])&set(va['frame']) or set(tr['frame'])&set(r.seed) or set(va['frame'])&set(r.seed))
 assert (r.bits==13440).all() and np.allclose(r.ber,r.errors/r.bits)
 count={k:[sum(a.size for a in m['model'].coefs_)+sum(a.size for a in m['model'].intercepts_) for m in models[k]] for k in ['Raw','Augmented','Canonical']}
 assert all(n==3100 for k in count for n in count[k]);assert all(m['model'].n_features_in_==37 for k in count for m in models[k])
 rg=np.random.default_rng(3600);x,b=modulation(rg);a,h=transmit(x,4);z=receive(a,4,h,linear=False);raw_error=float(np.max(np.abs(z-x)));assert raw_error<1e-12
 pp=float(np.mean(np.sum(np.abs(a)**2,axis=0))/(1e-3*10**.4));energy=float(np.mean(np.abs(x)**2));assert abs(pp-energy)<1e-12
 # Recompute a retained primary frame directly from its stored propagation output.
 w=np.load(d/'primary_waveforms.npz');j=np.flatnonzero((w['power_seed'][:,0]==4)&(w['power_seed'][:,1]==1200))[0];out=w['received'][j,0];x=w['symbols'][j];b=w['bits'][j];theta=np.random.default_rng(500000+1200+4*100).uniform(-np.pi,np.pi);a=add_lo(out,np.random.default_rng(600000+1200+4*100),linewidth=0,theta=theta);_,h=transmit(x,4);z=receive(a,4,h);met,_=pilot_recover(infer_net(z,4,models['Canonical'])[0],x,b);old=r[(r.case=='PhaseOffset')&(r.power_dbm==4)&(r.seed==1200)&(r.method=='CanonicalNN')].iloc[0];assert abs(met['snr_db']-old.snr_db)<1e-10 and met['errors']==old.errors
 out={'primary_rows':len(r),'unique_primary_seeds':24,'training_frames':len(set(tr['frame'])),'validation_frames':len(set(va['frame'])),'partition_intersections':0,'parameters_per_head':count,'raw_zero_channel_max_symbol_error':raw_error,'measured_launch_power_ratio':pp,'transmitted_symbol_mean_energy':energy,'retained_frame_replay_snr_error':float(met['snr_db']-old.snr_db),'retained_frame_replay_errors':met['errors'],'environment':{k:version(k) for k in ['numpy','scipy','scikit-learn','pandas','matplotlib','python-docx','joblib','lxml','Pillow']}}
 for f in sorted((ROOT/'figures').glob('fig*.png')):
  im=Image.open(f);dpi=im.info.get('dpi');assert dpi and min(dpi)>599
 out['figure_count']=len(list((ROOT/'figures').glob('fig*.png')));assert out['figure_count']==9
 content=json.loads((ROOT/'notes/manuscript_content.json').read_text());text='\n'.join(b.get('text','') for b in content['blocks'] if b['type']!='reference');seen=set()
 for group in re.findall(r'\[([0-9,– -]+)\]',text):
  for seg in group.split(','):
   if re.search('[–-]',seg):
    q=re.split('[–-]',seg);seen.update(range(int(q[0]),int(q[1])+1))
   else:seen.add(int(seg))
 assert set(range(1,26))<=seen;out['references_cited']=list(range(1,26));out['abstract_words']=len(content['abstract'].split())
 for f in (ROOT/'output').glob('*.docx'):
  with zipfile.ZipFile(f) as zz:
   root=etree.fromstring(zz.read('word/document.xml'));maths=len(root.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'));draw=len(root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'))
   if f.name in ['JOC_Second_Paper.docx','JOC_Second_Paper_Highlighted.docx']:assert maths==26 and draw==9
 (d/'validation_report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
