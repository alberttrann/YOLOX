"""Source-only research protocol helpers. Nothing here is used for TTT online."""
from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
import torch
from torch import Tensor
from torch.nn import functional as F
from .core import PRISMTTT,VERSION


def load_native(exp,checkpoint,device,weights='ema',seed=20261105):
    if weights not in ('ema','raw'):raise ValueError('weights must be ema/raw')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict):raise RuntimeError('Expected native YOLOX checkpoint dictionary')
    state=ck.get('ema_model' if weights=='ema' else 'model')
    if not isinstance(state,dict):raise RuntimeError('Required native checkpoint weights unavailable')
    torch.manual_seed(seed)
    m=exp.get_model().to(device)
    m.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    m.eval().requires_grad_(False)
    if hasattr(m,'set_meta_training_state'):
        m.set_meta_training_state(max(1,int(ck.get('start_epoch',1))),int(exp.max_epoch))
    else:raise RuntimeError('Expected vNext.2 meta-training state interface')
    if not hasattr(m.head,'get_losses_with_targets'):
        raise RuntimeError('Only audited vNext.2 TDE head supported')
    return m,ck


def native_sha256(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_train_example(dataset,index,device,seed):
    if not 0<=index<len(dataset):raise IndexError(index)
    torch.manual_seed(seed+index);random.seed(seed+index)
    try:
        import numpy as np
        np.random.seed((seed+index)%(2**32))
    except ImportError:pass
    pair=dataset[index]
    if not isinstance(pair,(tuple,list)) or len(pair)<2:raise RuntimeError('Unknown BDD dataset sample')
    x=torch.as_tensor(pair[0],device=device).float()
    targets=torch.as_tensor(pair[1],device=device).float()
    if x.ndim!=3 or x.shape[0]!=3 or targets.ndim!=2 or targets.shape[-1]!=5:
        raise RuntimeError('Require CHW image and max_GT x 5 COCO labels')
    if not torch.isfinite(x).all() or not torch.isfinite(targets).all():
        raise RuntimeError('Nonfinite training example')
    return x[None],targets[None]


def source_shift(x:Tensor,index:int,seed:int=20261105,kind='train'):
    """Geometry-preserving, source-predeclared photometric changes; no GT read.

    Train families: global contrast/channel gain or smooth illumination.
    Held-out family: mild resolution-preserving blur. This is NOT ACDC tuning.
    """
    if kind not in ('train','heldout'):raise ValueError('Unknown appearance family')
    rng=random.Random((seed*100003+index*1009+7)%2**63)
    if kind=='heldout':
        # Blend mild lowpass. No resize or spatial coordinate shift.
        blur=F.avg_pool2d(x,3,stride=1,padding=1)
        return x*.82+blur*.18
    if index%2==0:
        gains=x.new_tensor([rng.uniform(.78,1.22) for _ in range(3)])[None,:,None,None]
        contrast=rng.uniform(.84,1.16)
        mean=x.mean((2,3),keepdim=True)
        return ((x-mean)*contrast+mean)*gains
    else:
        h,w=x.shape[-2:]
        yy=torch.linspace(-1,1,h,device=x.device,dtype=x.dtype)[None,None,:,None]
        xx=torch.linspace(-1,1,w,device=x.device,dtype=x.dtype)[None,None,None,:]
        phase=rng.uniform(-math.pi,math.pi)
        strength=rng.uniform(.08,.20)
        field=1+strength*torch.cos(math.pi*yy+phase)*torch.cos(math.pi*xx*.5)
        return x*field


def save_sidecar(path,sidecar:PRISMTTT,*,native_hash:str,native_checkpoint:str,
                 native_weights:str,epoch:int,seed:int,source_indices:dict,metrics:dict):
    path=Path(path)
    if path.exists():raise FileExistsError('Refusing to overwrite sidecar')
    path.parent.mkdir(parents=True,exist_ok=True)
    doc={'prism_version':VERSION,'state_dict':{k:v.detach().cpu().clone() for k,v in sidecar.state_dict().items()},
         'metadata':{'native_hash':native_hash,'native_checkpoint':str(Path(native_checkpoint).resolve()),
                     'native_weights':native_weights,'native_epoch':int(epoch),'seed':int(seed),
                     'channels':sidecar.channels,'source_only':True,
                     'source_indices':source_indices,'source_validation':metrics}}
    import tempfile,os
    fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix='._prism_',suffix='.tmp');os.close(fd)
    try:
        torch.save(doc,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_sidecar(path,native_hash,channels,device,weights):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(ck,dict) or ck.get('prism_version')!=VERSION:
        raise RuntimeError('Unknown or outdated PRISM checkpoint')
    cfg=ck.get('metadata',{})
    if cfg.get('source_only') is not True or cfg.get('native_hash')!=native_hash or cfg.get('native_weights')!=weights:
        raise RuntimeError('PRISM provenance / frozen checkpoint mismatch')
    if tuple(cfg.get('channels',()))!=tuple(channels):raise RuntimeError('Pyramid channel mismatch')
    model=PRISMTTT(channels).to(device=device,dtype=torch.float32)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    return model,ck


def gate_calibrate_from_reports(sidecar:PRISMTTT,records,*,min_examples:int=16,
                                max_false_accept:float=.1,min_accepted:int=5):
    """Calibrate from disjoint source-only GT-scored reports, not OOD labels.

    reports iterable of (scores [3], benefits [3], presence [3]) lists.
    Gate remains abstaining if no threshold meets risk/coverage requirement.
    """
    if len(records)<min_examples:raise ValueError('Insufficient source scenes for gate calibration')
    thresholds=[];coverage=[]
    for i in range(3):
        pairs=[(float(sc[i]),float(benefit[i])) for sc,benefit,present in records if bool(present[i])]
        if len(pairs)<min_examples:thresholds.append(1.1);coverage.append(0);continue
        chosen=1.1;accepted_n=0
        # Choose largest acceptance with a low estimated false-accept risk.
        for t in [j/100 for j in range(50,100)]:
            selected=[ok for score,ok in pairs if score>=t]
            if len(selected)>=min_accepted and sum(1-ok for ok in selected)/len(selected)<=max_false_accept:
                chosen=t;accepted_n=len(selected);break
        thresholds.append(chosen);coverage.append(accepted_n)
    sidecar.gate.thresholds.copy_(sidecar.gate.thresholds.new_tensor(thresholds))
    sidecar.gate.calibrated.fill_(True)
    return {'thresholds':thresholds,'coverage':coverage,'source_reliability_calibrated':True,
            'all_rejected':all(x==0 for x in coverage)}
