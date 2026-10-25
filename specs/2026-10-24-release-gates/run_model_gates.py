import subprocess,json,sys,time
from pathlib import Path
import numpy as np
root=Path('/tmp/nert-release-gates')
dev=sys.argv[1]
runner=root/('gate_'+dev)
failed=False
for name in (sys.argv[2:] or ['ecdet_s','ecseg_s','ecpose_s','rfdetr_det_small','rfdetr_seg_medium']):
 d=root/name;meta=json.loads((d/'provenance.json').read_text());prefix=d/dev
 cmd=[str(runner),meta['model'],dev,str(prefix)]+[k+'='+str(d/(k+'.bin')) for k in meta['inputs']]
 start=time.monotonic()
 with (d/(dev+'.log')).open('w') as log:
  try: code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
  except subprocess.TimeoutExpired: code=124
 result={'process_exit':code,'gate_exit':code,'strict_pass':False,'seconds':time.monotonic()-start,'outputs':{}}
 if code==0:
  for refpath in sorted(d.glob('ref_*.npy')):
   key=refpath.stem[4:];ref=np.load(refpath);got=np.fromfile(str(prefix)+'_'+key+'.bin',dtype=ref.dtype)
   if got.size!=ref.size:result['gate_exit']=1;continue
   got=got.reshape(ref.shape)
   if ref.dtype.kind=='f':
    delta=np.abs(got.astype(np.float64)-ref.astype(np.float64));ratio=delta/(1e-3*(1+np.abs(ref.astype(np.float64))))
    count=int(np.count_nonzero((ratio>1)|~np.isfinite(got)|~np.isfinite(ref)))
    result['outputs'][key]={'over_tol':count,'elements':ref.size,'max_abs':float(delta.max()),'max_ratio':float(ratio.max())}
   else:
    count=int(np.count_nonzero(got!=ref));result['outputs'][key]={'mismatch':count,'elements':ref.size}
   if count:result['gate_exit']=1
 else: print((d/(dev+'.log')).read_text()[-1600:],flush=True)
 if code == 0:
  checker = Path(__file__).resolve().parents[2] / 'tools/check_model_outputs.py'
  strict = subprocess.run([sys.executable,str(checker),str(d),str(prefix)],capture_output=True,text=True)
  (d/(dev+'_strict.json')).write_text(strict.stdout)
  result['gate_exit'] = strict.returncode
  result['strict_pass'] = strict.returncode == 0
 (d/(dev+'_metrics.json')).write_text(json.dumps(result,indent=2));print(name,dev,result,flush=True)
 failed|=not result['strict_pass']
sys.exit(int(failed))
