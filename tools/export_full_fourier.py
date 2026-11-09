"""Full detector ONNX export and optional numerical parity; no fake export claims.

Exports raw decoded-disabled YOLOX predictions [B,N,5+K].  The downstream
NMS, coordinate inverse transform and camera pipeline are separate deployment
components. TensorRT engine building must be tested on target hardware.
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from yolox.exp import get_exp


class RawPredictionGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model
    def forward(self,x):return self.model(x)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--model-exp',required=True)
    a.add_argument('--checkpoint',required=True)
    a.add_argument('--weights',choices=['raw','ema'],default='ema')
    a.add_argument('--output',required=True,type=Path)
    a.add_argument('--height',type=int,default=576)
    a.add_argument('--width',type=int,default=1024)
    a.add_argument('--dynamic',action='store_true',help='Enable dynamic batch/H/W axes')
    args=a.parse_args()
    exp=get_exp(args.model_exp,None)
    m=exp.get_model().eval().cpu()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict) or args.weights=='ema' and ck.get('ema_model') is None:
        raise RuntimeError('Missing requested EMA checkpoint weights')
    state=ck['ema_model' if args.weights=='ema' else 'model']
    state={key.removeprefix('module.'):v for key,v in state.items()}
    m.load_state_dict(state,strict=True)
    m.head.decode_in_inference=False
    graph=RawPredictionGraph(m)
    x=torch.randn(1,3,args.height,args.width)
    with torch.no_grad():ref=graph(x)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    axes={'images':{0:'batch',2:'height',3:'width'},'raw_predictions':{0:'batch',1:'anchors'}} if args.dynamic else None
    try:
        torch.onnx.export(graph,(x,),str(args.output),input_names=['images'],
                          output_names=['raw_predictions'],dynamic_axes=axes,
                          opset_version=17,dynamo=False)
    except Exception as exc:
        raise RuntimeError('Full ONNX export FAILED; no TensorRT compatibility '
                           'is established. Inspect unsupported ops and report.') from exc
    print('ONNX EXPORTED:',args.output,'raw_shape:',tuple(ref.shape))
    try:
        import onnxruntime as ort
    except ImportError:
        print('ORT parity SKIPPED (onnxruntime not installed)')
    else:
        session=ort.InferenceSession(str(args.output),providers=['CPUExecutionProvider'])
        out=session.run(None,{'images':x.numpy()})[0]
        diff=(torch.from_numpy(out)-ref).abs()
        print('ONNXRuntime max_abs_error=',float(diff.max()),'mean_abs_error=',float(diff.mean()))
        torch.testing.assert_close(torch.from_numpy(out),ref,rtol=1e-3,atol=1e-3)
        print('ORT numerical parity PASSED')

if __name__=='__main__':main()
