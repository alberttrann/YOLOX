"""No-training certification of A4.1 model, effective loader and Mosaic behavior.

By default constructs the real dataset/loader but does not read any image;
--sample-mosaic forces 1 four-image Mosaic and 1 non-Mosaic example (main
process) and counts *actual* underlying pull_item calls, no optimizer step.
"""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from yolox.exp import get_exp


def inspect(exp, sample_mosaic=False, device='cpu', backward=False, height=64, width=96):
    from yolox.data import MosaicDetection
    exp._preflight()
    original_workers = exp.data_num_workers
    exp.data_num_workers = 0   # No child processes, deterministic direct dataset probe.
    try:
        loader = exp.get_data_loader(batch_size=1, is_distributed=False,
                                     no_aug=False, cache_img=False)
    finally:
        exp.data_num_workers = original_workers
    dataset = loader.dataset
    if not isinstance(dataset, MosaicDetection):
        raise RuntimeError('A4.1 certification failed: dataset is not MosaicDetection')
    raw = dataset._dataset
    if getattr(raw,'preproc','missing') is not None:
        raise RuntimeError('Raw COCODataset must have no preprocessing')
    if getattr(dataset,'preproc',None) is None:
        raise RuntimeError('MosaicDetection must own the TrainTransform')
    if not bool(loader.batch_sampler.mosaic):
        raise RuntimeError('Mosaic YoloBatchSampler flag is false before terminal phase')
    meta = {
        'experiment': exp.exp_name,
        'basis': exp.a4_filter_basis,
        'wrapper_type': type(dataset).__name__,
        'underlying_type': type(raw).__name__,
        'underlying_preproc': None,
        'wrapper_preproc': type(dataset.preproc).__name__,
        'mosaic_prob': float(dataset.mosaic_prob),
        'sampler_mosaic': bool(loader.batch_sampler.mosaic),
        'target_capacity': int(exp._capacity),
        'max_epoch': int(exp.max_epoch),
        'no_aug_epochs': int(exp.no_aug_epochs),
    }
    if sample_mosaic:
        saved_prob = dataset.mosaic_prob
        original_pull = raw.pull_item
        count = [0]
        def counting_pull(index):
            count[0] += 1
            return original_pull(index)
        raw.pull_item = counting_pull
        try:
            dataset.mosaic_prob = 1.0
            augmented = dataset[(True, 0)]
            pulls_mosaic = count[0]
            count[0] = 0
            # Same batch sampler signal, forced regular path via probability.
            dataset.mosaic_prob = 0.0
            regular = dataset[(True, 0)]
            pulls_regular = count[0]
            count[0] = 0
            dataset.mosaic_prob = 1.0
            closed = dataset[(False, 0)]
            pulls_closed = count[0]
            if pulls_mosaic != 4 or pulls_regular != 1 or pulls_closed != 1:
                raise RuntimeError('Mosaic sample-count certification failed: '
                                   +str((pulls_mosaic,pulls_regular,pulls_closed)))
            for x in (augmented,regular,closed):
                labels=x[1]
                if int(labels.shape[0]) != int(exp._capacity):
                    raise RuntimeError('Incorrect label capacity after preprocessing')
            meta['empirical_pull_items']={
                'mosaic_forced':pulls_mosaic,
                'regular_forced':pulls_regular,
                'sampler_closed':pulls_closed,
            }
        finally:
            dataset.mosaic_prob = saved_prob
            raw.pull_item = original_pull
    if backward:
        if height % 32 or width % 32:
            raise ValueError('Synthetic height/width must be multiples of 32')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        model = exp.get_model().to(device).train()
        x = torch.randn(2,3,height,width,device=device)
        y = torch.zeros(2,8,5,device=device)
        for k in range(2):
            y[k,0] = torch.tensor([float(k%exp.num_classes),width/2,height/2,
                                    min(width/4,24),min(height/4,24)],device=device)
        outputs = model(x,y)
        if not bool(torch.isfinite(outputs['total_loss'])):
            raise RuntimeError('Nonfinite A4.1 synthetic model loss')
        mem = outputs['mem_loss']
        named = dict(model.named_parameters())
        shared = next(p for n,p in named.items() if 'backbone.backbone.dark3' in n)
        proto = [p for n,p in named.items() if 'memory_banks' in n and 'prototypes' in n]
        query = [p for n,p in named.items() if 'latent_projectors' in n and p.ndim>1]
        aux_grad = torch.autograd.grad(mem,[shared]+query+proto,
                                       retain_graph=True,allow_unused=True)
        if aux_grad[0] is not None and bool(aux_grad[0].abs().sum()):
            raise RuntimeError('Prototype aux leaked into backbone')
        if not any(g is not None and bool(g.abs().sum()) for g in aux_grad[1:1+len(query)]):
            raise RuntimeError('Query projector received zero prototype-aux gradient')
        if not any(g is not None and bool(g.abs().sum()) for g in aux_grad[1+len(query):]):
            raise RuntimeError('Prototypes received zero class-grounding gradient')
        outputs['total_loss'].backward()
        families = ('spi_highway.projector','spi_highway.router',
                    'C3_p3.global_mixer','spectral_adapters','memory_banks')
        nz={}
        for f in families:
            gg=[p.grad.abs().sum() for n,p in model.named_parameters()
                if f in n and p.grad is not None]
            nz[f]=float(torch.stack(gg).sum()) if gg else 0.0
            if not nz[f]>0:
                raise RuntimeError('No detection gradient in '+f)
        meta['synthetic_cuda_or_cpu']={
            'device':device, 'total_loss':float(outputs['total_loss'].detach()),
            'proto_loss':float(mem.detach()), 'nonzero_gradient_family_l1':nz,
            'assignment':model.head.last_assignment_stats,
            'class_grounding':model.head.last_proto_diagnostic,
        }
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-exp',default='exps/default/tde_yolox_s_fourier_a41.py')
    p.add_argument('--sample-mosaic',action='store_true')
    p.add_argument('--backward',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--height',type=int,default=64)
    p.add_argument('--width',type=int,default=96)
    p.add_argument('--output',default='audits/a41_certification.json')
    a=p.parse_args()
    result=inspect(get_exp(a.model_exp,None),a.sample_mosaic,a.device,a.backward,
                   a.height,a.width)
    dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('A41_CERTIFICATION_PASS')


if __name__=='__main__':main()
