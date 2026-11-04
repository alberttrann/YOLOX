"""Optional real-repo ONNX export probe (requires ONNX and full YOLOX).

Raw output parity probe only. TRT inference, NMS and real edge benchmarking
are separate mandatory user-hardware gates; no latency claims here.
"""
import argparse
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['pgsma','native','memory'],default='pgsma')
    p.add_argument('--output',default='vnext3_pgsma.onnx')
    p.add_argument('--height',type=int,default=576)
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--checkpoint',required=True)
    args=p.parse_args()
    if args.height%32 or args.width%32:
        raise ValueError('Input dimensions must be multiples of 32')
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('Install ONNX first; export cannot be verified here') from exc
    from exps.default.tde_yolox_s_pgsma import Exp
    exp=Exp();model=exp.get_model()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['ema_model'] if ckpt.get('ema_model') else ckpt['model'],strict=True)
    model.eval();model.set_inference_mode(args.mode)
    inp=torch.randn(1,3,args.height,args.width)
    with torch.no_grad(): reference=model(inp)
    torch.onnx.export(model,inp,args.output,input_names=['images'],
       output_names=['detections'],dynamic_axes={'images':{0:'batch',2:'height',3:'width'},
                'detections':{0:'batch',1:'anchors'}},opset_version=17)
    onnx.checker.check_model(onnx.load(args.output))
    print('ONNX checker passed. Native PyTorch output:',tuple(reference.shape))
    print('NOT YET VERIFIED: ONNX runtime parity, TensorRT build, dynamic profiles, NMS, latency.')

if __name__=='__main__': main()
