#!/usr/bin/env python3
"""Recreate the exact five canonical frozen inputs and ORT references; no downloads."""
from pathlib import Path
import json,hashlib
import numpy as np
import onnxruntime as ort
from PIL import Image
root=Path('/tmp/nert-release-gates')
root.mkdir(parents=True, exist_ok=True)
img=Image.open('/home/oli/repos/neuriplo/build-parity/native_yolo_parity_fixture/bus.jpg').convert('RGB')
models={m: f'/tmp/opencode/exports/{m}/{m}.onnx' for m in ('ecdet_s','ecseg_s','ecpose_s')}
models.update(rfdetr_det_small='/tmp/opencode/exports/rfdetr_det_small/rfdetr-small-sim.onnx',rfdetr_seg_medium='/tmp/opencode/exports/rfdetr_seg_medium/rfdetr-seg-medium-sim.onnx')
for name,path in models.items():
 d=root/name;d.mkdir(parents=True, exist_ok=True)
 so=ort.SessionOptions();so.intra_op_num_threads=2;so.log_severity_level=3
 sess=ort.InferenceSession(path,so,providers=['CPUExecutionProvider'])
 feed={}
 for inp in sess.get_inputs():
  shape=[v if isinstance(v,int) and v>0 else 1 for v in inp.shape]
  if inp.type=='tensor(int64)': x=np.array([[img.height,img.width]],dtype=np.int64).reshape(shape)
  else:
   h,w=shape[-2:];x=np.asarray(img.resize((w,h),Image.Resampling.BILINEAR),dtype=np.float32)/np.float32(255)
   x=np.ascontiguousarray(x.transpose(2,0,1)[None])
   if name.startswith('ec'): x=(x-np.array([.485,.456,.406],dtype=np.float32)[None,:,None,None])/np.array([.229,.224,.225],dtype=np.float32)[None,:,None,None]
  feed[inp.name]=x;np.save(d/(inp.name+'.npy'),x);x.tofile(d/(inp.name+'.bin'))
 for out,value in zip(sess.get_outputs(),sess.run(None,feed)):np.save(d/('ref_'+out.name+'.npy'),value)
 meta={'model':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'inputs':{k:list(v.shape) for k,v in feed.items()},'ort':ort.__version__, 'numpy':np.__version__, 'pillow':__import__('PIL').__version__, 'source_image':'/home/oli/repos/neuriplo/build-parity/native_yolo_parity_fixture/bus.jpg', 'source_image_sha256':hashlib.sha256(Path('/home/oli/repos/neuriplo/build-parity/native_yolo_parity_fixture/bus.jpg').read_bytes()).hexdigest()}
 (d/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,meta,flush=True)
