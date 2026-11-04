"""Export dynamic ONNX and compare PyTorch with ONNX Runtime for multiple shapes.

Requires onnx, onnxruntime; NOT TensorRT certification. No training.
Use only trusted checkpoint paths. Run CPU/FP32 first to isolate graph parity.
"""
from __future__ import annotations
import argparse
import inspect
import sys
from pathlib import Path
import numpy as np
import torch
from ._pgsma_preflight_common import (load_exp,load_torch_checkpoint,
    unwrap_state_dict,normalize_state_keys,checkpoint_coverage,dump_json)


def legacy_export_options(export_fn):
    """Choose legacy dynamic_axes export without unsupported keywords."""
    try:
        signature = inspect.signature(export_fn)
    except (TypeError, ValueError):
        return {}
    return {'dynamo': False} if 'dynamo' in signature.parameters else {}


def check_onnx_dynamic_inputs(model_proto):
    """Check dynamic H/W and batch are actually encoded in exported graph."""
    if len(model_proto.graph.input) != 1:
        raise ValueError('Expected exactly one ONNX graph input: images')
    dims = model_proto.graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError('Expected rank-4 ONNX image input')
    found = {}
    for axis, name in ((0, 'batch'), (2, 'height'), (3, 'width')):
        dim = dims[axis]
        if not dim.dim_param:
            raise ValueError(f'ONNX input axis {axis} ({name}) frozen to {dim.dim_value}')
        found[name] = dim.dim_param
    return found

def parse_shapes(values):
    out=[]
    for value in values:
        h,w=map(int,value.lower().split('x'))
        if h%32 or w%32 or h<32 or w<32:raise ValueError('Shapes must be positive, divisible by 32')
        out.append((h,w))
    return out


def audit(args):
    import onnx
    import onnxruntime as ort
    exp=load_exp();model=exp.get_model()
    state_desc='random_untrained_architecture_only'
    if args.checkpoint:
        ckpt=load_torch_checkpoint(args.checkpoint)
        state,sub=unwrap_state_dict(ckpt,args.prefer_ema)
        cov=checkpoint_coverage(model,normalize_state_keys(state))
        model.load_state_dict(cov['matched'],strict=False)
        state_desc=f'{args.checkpoint} ({sub}, {len(cov["matched"])} tensors matched)'
    model=model.cpu().eval();model.set_inference_mode(args.mode)
    h,w=parse_shapes(args.shapes)[0]
    # Explicit legacy trace exporter: uses dynamic_axes. The modern dynamo
    # exporter instead requires dynamic_shapes and onnxscript.
    dummy=torch.randn(1,3,h,w)
    dst=Path(args.output);dst.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(model,dummy,str(dst),opset_version=17,
            input_names=['images'],output_names=['detections'],
            dynamic_axes={'images':{0:'batch',2:'height',3:'width'},
                          'detections':{0:'batch',1:'anchors'}},
            **legacy_export_options(torch.onnx.export))
    onnx.checker.check_model(str(dst))
    dyn=check_onnx_dynamic_inputs(onnx.load(str(dst)))
    so=ort.SessionOptions();so.log_severity_level=3
    sess=ort.InferenceSession(str(dst),sess_options=so,providers=['CPUExecutionProvider'])
    record={'onnx_path':str(dst),'mode':args.mode,'weights':state_desc,
            'torch_version':torch.__version__,'exporter':'legacy_dynamic_axes',
            'dynamic_input_axes':dyn,
            'onnx_check':'passed','providers':sess.get_providers(),
            'shapes':[],'status':'CHECKS_PASSED',
            'not_verified':['TensorRT compile/profile','FP16 TensorRT parity','NMS output parity','edge latency/VRAM']}
    for idx,(height,width) in enumerate(parse_shapes(args.shapes)):
        g=torch.Generator().manual_seed(500+idx)
        t=torch.randn(1,3,height,width,generator=g)
        with torch.inference_mode(): ref=model(t).detach().cpu().numpy()
        outputs=sess.run(None,{'images':t.numpy()})
        if len(outputs)!=1:raise ValueError('Expected single detections ONNX output')
        out=outputs[0]
        if out.shape!=ref.shape:raise ValueError(f'ONNX shape {out.shape} != PyTorch {ref.shape} for {height}x{width}')
        if not np.isfinite(out).all():raise FloatingPointError(f'ONNX nonfinite for {height}x{width}')
        maxabs=float(np.max(np.abs(ref-out)))
        ok=bool(np.allclose(ref,out,atol=args.atol,rtol=args.rtol))
        record['shapes'].append({'H':height,'W':width,'shape':list(out.shape),'max_abs':maxabs,'allclose':ok})
        if not ok:record['status']='BLOCKED'
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['native','pgsma','memory'],default='pgsma')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--prefer-ema',action='store_true')
    p.add_argument('--output',default='audit_pgsma_dynamic.onnx')
    p.add_argument('--shapes',nargs='+',default=['64x96','96x64','128x128'])
    p.add_argument('--atol',type=float,default=2e-3)
    p.add_argument('--rtol',type=float,default=2e-3)
    p.add_argument('--json',default=None)
    a=p.parse_args()
    try:res=audit(a)
    except Exception as exc:res={'status':'BLOCKED','error':f'{type(exc).__name__}: {exc}'}
    dump_json(res,a.json)
    sys.exit(2 if res['status']=='BLOCKED' else 0)

if __name__=='__main__':main()
