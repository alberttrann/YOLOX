"""Training-only weak-semantic supervision for A4.1.

No segmentation labels are manufactured from boxes. Gaussian fields indicate
object-center evidence, not object silhouette. Reliability masks ignore uncertain
bbox interior corners; tests explicitly distinguish from the historical flat-top
superquadric occupancy target. All losses are computed in float32 under AMP.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def center_gaussian_targets(targets, num_classes, feature_size, image_size,
                            sigma_divisor=6.0, minimum_sigma=0.65,
                            ignore_box_corners=True, chunk_size=32):
    if targets.ndim != 3 or targets.shape[-1] != 5:
        raise ValueError('Expected YOLOX padded targets [B,N,5] = class,xc,yc,w,h')
    if sigma_divisor <= 0 or minimum_sigma <= 0 or chunk_size < 1:
        raise ValueError('Invalid gaussian target parameters')
    bsz = targets.shape[0]
    height, width = [int(x) for x in feature_size]
    img_h, img_w = [int(x) for x in image_size]
    if min(height,width,img_h,img_w) < 1:
        raise ValueError('Invalid image or feature geometry')
    stride_y, stride_x = img_h / height, img_w / width
    device = targets.device
    heat = torch.zeros((bsz,num_classes,height,width),device=device,dtype=torch.float32)
    reliable = torch.ones_like(heat)
    xx = (torch.arange(width,device=device,dtype=torch.float32) + 0.5)[None,None,:]
    yy = (torch.arange(height,device=device,dtype=torch.float32) + 0.5)[None,:,None]
    for b in range(bsz):
        box = targets[b].float()
        good = ((box[:,0]>=0)&(box[:,0]<num_classes)&(box[:,3]>0)&(box[:,4]>0)
                &torch.isfinite(box).all(dim=1))
        box = box[good]
        for first in range(0,box.shape[0],chunk_size):
            block=box[first:first+chunk_size]
            cls=block[:,0].long()
            cx=(block[:,1]/stride_x)[:,None,None]
            cy=(block[:,2]/stride_y)[:,None,None]
            w=(block[:,3]/stride_x)[:,None,None]
            h=(block[:,4]/stride_y)[:,None,None]
            sx=(w/sigma_divisor).clamp_min(minimum_sigma)
            sy=(h/sigma_divisor).clamp_min(minimum_sigma)
            dx=(xx-cx)/sx
            dy=(yy-cy)/sy
            radial=dx.square()+dy.square()
            # Explicit finite support, with negligible truncation at 4 sigma.
            blob=torch.exp((-0.5*radial).clamp_min(-80.0)) * (radial<=16).float()
            inside=((xx-cx).abs()<=w/2)&((yy-cy).abs()<=h/2)
            for c_tensor in cls.unique():
                c=int(c_tensor)
                keep=cls==c
                heat[b,c]=torch.maximum(heat[b,c],blob[keep].amax(dim=0))
                if ignore_box_corners:
                    corner=(inside[keep] & (blob[keep]<0.08)).any(dim=0)
                    reliable[b,c].masked_fill_(corner,0.0)
    # A center supported by any same-class box overrides another box's corner.
    reliable=torch.where(heat>=0.08,torch.ones_like(reliable),reliable)
    return heat,reliable


class BalancedCenterLoss(nn.Module):
    """Balanced BCE on *soft* target; optional overlap, per-active-class Dice.

    Channel-level means avoid background-canvas dilation drowning positives.
    The same original soft-target BCE is retained within each group, so this
    is not a hard conversion of fractional Gaussian values to binary labels.
    """
    def __init__(self,positive_threshold=0.03, dice_weight=0.25):
        super().__init__()
        self.positive_threshold=float(positive_threshold)
        self.dice_weight=float(dice_weight)

    def forward(self,logits,target,reliability):
        x=logits.float(); y=target.float(); weight=reliability.float()
        if x.shape!=y.shape or x.shape!=weight.shape:
            raise ValueError('Semantic logits/target/reliability shapes differ')
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite semantic logits')
        if not torch.all((y>=0)&(y<=1)):
            raise ValueError('Soft targets outside [0,1]')
        bce=F.binary_cross_entropy_with_logits(x,y,reduction='none')
        is_positive=(y>=self.positive_threshold).float()*weight
        is_negative=(y<self.positive_threshold).float()*weight
        sum_dims=(-2,-1)
        pos_sum=(bce*is_positive).sum(dim=sum_dims)
        neg_sum=(bce*is_negative).sum(dim=sum_dims)
        pos_n=is_positive.sum(dim=sum_dims)
        neg_n=is_negative.sum(dim=sum_dims)
        # Each present-class channel receives positive and negative evidence.
        # Absent channels still contribute negative evidence without dominating.
        active=(pos_n>0).float()
        mean_pos=(pos_sum/(pos_n+1e-6)*active).sum()/(active.sum()+1e-6)
        mean_neg=(neg_sum/(neg_n+1e-6)).mean()
        p=torch.sigmoid(x)
        intersection=(p*y*weight).sum(dim=sum_dims)
        mass=((p.square()+y.square())*weight).sum(dim=sum_dims)
        dice=1-(2*intersection+1)/(mass+1)
        mean_dice=(dice*active).sum()/(active.sum()+1e-6)
        return 0.5*mean_pos+0.5*mean_neg+self.dice_weight*mean_dice


def _dct_kernel_3x3():
    coords=torch.arange(3,dtype=torch.float32)
    def unit(u):
        coeff=math.sqrt(1/3) if u==0 else math.sqrt(2/3)
        return coeff*torch.cos(math.pi*(2*coords+1)*u/6)
    kernels=[]
    for u,v in ((0,1),(1,0),(2,2)):
        k=torch.outer(unit(u),unit(v))
        k=k/(k.square().sum().sqrt()+1e-12)
        kernels.append(k)
    return torch.stack(kernels).unsqueeze(1)


class LocalHarmonicConsistency(nn.Module):
    """Fixed local DCT-oriented residual matching, TRAIN ONLY.

    This is not contour recovery or a phase-Dice loss. Supervises spatial
    response geometry to weak center heatmaps; class features not reconstructed.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('kernels',_dct_kernel_3x3())
    def forward(self,logits,target,reliability):
        p=torch.sigmoid(logits.float()); t=target.float()
        c=p.shape[1]
        kernel=self.kernels.repeat(c,1,1,1)
        pred=F.conv2d(p,kernel,padding=1,groups=c)
        truth=F.conv2d(t,kernel,padding=1,groups=c)
        # Exclude uncertain corners and empty classes; apply local support halo.
        sup=F.max_pool2d((t>0.03).float(),5,stride=1,padding=2)*reliability.float()
        mask=sup.repeat_interleave(3,dim=1)
        loss=F.smooth_l1_loss(pred,truth,reduction='none',beta=0.1)
        return (loss*mask).sum()/(mask.sum()+1e-6)


class ReliabilityWeightedPhaseProbe(nn.Module):
    """EXPLORATORY global FFT phase probe, not recommended default.

    Excludes DC and unreliable low-magnitude bins. Global multi-instance phase
    is not object-contour phase. Use only as independent ablation.
    """
    def forward(self,logits,target,reliability):
        p=torch.sigmoid(logits.float())*reliability.float()
        t=target.float()*reliability.float()
        fp=torch.fft.rfft2(p,norm='ortho')
        ft=torch.fft.rfft2(t,norm='ortho')
        ap=fp.abs(); at=ft.abs()
        cutoff=0.025*at.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
        mask=(at>cutoff).float()*(ap.detach()>1e-6).float()
        mask[...,0,0]=0
        # Normalized phase cross-correlation, zero magnitude is harmless.
        cos=(fp*ft.conj()).real/(ap*at+1e-6)
        strength=(at/(at.amax(dim=(-2,-1),keepdim=True)+1e-6)).detach()
        w=strength*mask
        return ((1-cos.clamp(-1,1))*w).sum()/(w.sum()+1e-6)


class SemanticSupervision(nn.Module):
    def __init__(self,num_classes,mode='gaussian',sigma_divisor=6.0,
                 harmonic_weight=0.05):
        super().__init__()
        if mode not in ('legacy','gaussian','gaussian_harmonic','gaussian_phase_probe'):
            raise ValueError('Unsupported auxiliary semantic mode '+str(mode))
        self.mode=mode
        self.num_classes=int(num_classes)
        self.sigma_divisor=float(sigma_divisor)
        self.harmonic_weight=float(harmonic_weight)
        self.center_criterion=BalancedCenterLoss()
        self.local_harmonic=LocalHarmonicConsistency()
        self.phase_probe=ReliabilityWeightedPhaseProbe()
        self.last_stats={}

    def forward(self,logits,targets,image_size):
        feat=logits.shape[-2:]
        # Explicit FP32 island: both FFT probe and local convolutions must
        # avoid accidental half FFT / half-precision supervision under AMP.
        with torch.autocast(device_type=logits.device.type,enabled=False):
            return self._forward_float32(logits.float(),targets,image_size)

    def _forward_float32(self,logits,targets,image_size):
        feat=logits.shape[-2:]
        if self.mode=='legacy':
            # Preserve the actual historical superquadric target and loss.
            from .ttt_modules import BoxOccupancyTargetGenerator, ActiveClassFocalDiceLoss
            if image_size[0]%feat[0] or image_size[1]%feat[1]:
                raise ValueError('Legacy generator requires integer feature stride')
            sh=image_size[0]//feat[0]; sw=image_size[1]//feat[1]
            if sh!=sw:
                raise ValueError('Legacy generator requires isotropic stride')
            heat=BoxOccupancyTargetGenerator.generate(targets,self.num_classes,
                                                       feat_size=feat,stride=float(sh))
            spatial=ActiveClassFocalDiceLoss()(logits,heat)
            spectral=spatial*0
        else:
            heat,reliability=center_gaussian_targets(
                targets,self.num_classes,feat,image_size,
                sigma_divisor=self.sigma_divisor)
            spatial=self.center_criterion(logits,heat,reliability)
            if self.mode=='gaussian_harmonic':
                spectral=self.local_harmonic(logits,heat,reliability)
            elif self.mode=='gaussian_phase_probe':
                spectral=self.phase_probe(logits,heat,reliability)
            else:
                spectral=spatial*0
        raw=spatial+self.harmonic_weight*spectral
        # No GPU sync or data-dependent branching in the training fast path.
        self.last_stats={'spatial':spatial.detach(), 'spectral':spectral.detach(),
                         'raw':raw.detach()}
        return raw
