"""Run synthetic scientific analyses, followed optionally by editable documents."""
from pathlib import Path
import os,subprocess,sys,argparse
p=argparse.ArgumentParser();p.add_argument('--documents',action='store_true');args=p.parse_args();root=Path(__file__).resolve().parent
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
steps=['train.py','evaluate.py','diagnostics.py','grid_transfer.py','assemble_references.py','figures.py']
if args.documents:steps+=['build_documents.py','validate_records.py']
for name in steps:
 print('Running',name,flush=True);subprocess.run([sys.executable,str(root/name)],env=env,check=True)
print('Completed synthetic analyses. See README for PDF rendering and review.')
