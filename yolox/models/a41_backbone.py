"""A4.1 additive SPI/Tribrid integration supporting two *verified* source ABIs.

ABI 1 (user's deployed vNext.2): C2f_Tribrid inherits CSPLayer and has
  conv1, conv2, conv3, m, global_mixer, dense_local, dense_gate.
ABI 2 (archived C2f prototype): cv1, cv2, m, global_mixer.

The two implementations do NOT have identical forward graphs. In particular,
CSPLayer uses two input-to-hidden projections and a post-fusion conv, whereas
C2f uses one projection split into two branches and a concat of intermediates.
Never use a C2f-style forward on a CSPLayer-style instance.
"""
import math
import torch
from torch import nn
from .fourier_backbone import NoClippingSPI, prepare_backbone
from .tribrid_neck import C2f_Tribrid
from .ttt_modules import UnifiedRMBConv, ContinuousSaliencyAttention


def _conv(layer, label):
    node = layer if isinstance(layer, nn.Conv2d) else getattr(layer, 'conv', None)
    if not isinstance(node, nn.Conv2d):
        raise RuntimeError('Unsupported Tribrid '+label+': expected Conv2d or BaseConv.conv; got '+str(type(layer)))
    return node


def _identify_layout(block):
    """Fail-closed validation; return (layout, hidden channels)."""
    if not isinstance(block, C2f_Tribrid):
        raise TypeError('Expected local C2f_Tribrid; got '+str(type(block)))
    if not hasattr(block, 'm') or not isinstance(block.m, (nn.Sequential, nn.ModuleList)):
        raise RuntimeError('Tribrid missing a Sequential/ModuleList of CSP blocks')
    if not isinstance(getattr(block, 'global_mixer', None), nn.Sequential):
        raise RuntimeError('Tribrid missing global_mixer Sequential')
    if not isinstance(getattr(block, 'gate', None), nn.Parameter):
        raise RuntimeError('Tribrid missing learnable global gate')
    if all(hasattr(block, n) for n in ('conv1', 'conv2', 'conv3')):
        if hasattr(block, 'cv1') or hasattr(block, 'cv2'):
            raise RuntimeError('Ambiguous Tribrid: both CSP and C2f convolution names')
        first = _conv(block.conv1, 'conv1')
        bypass = _conv(block.conv2, 'conv2')
        fused = _conv(block.conv3, 'conv3')
        c = first.out_channels
        if bypass.out_channels != c or fused.in_channels != 2*c:
            raise RuntimeError('Unsupported native CSP Tribrid hidden/concat geometry: '
                               +str((c, bypass.out_channels, fused.in_channels)))
        if first.in_channels != bypass.in_channels:
            raise RuntimeError('CSP Tribrid conv1/conv2 source-channel mismatch')
        if not isinstance(getattr(block, 'dense_local', None), nn.Module):
            raise RuntimeError('CSP Tribrid must preserve dense_local; missing module')
        if not isinstance(getattr(block, 'dense_gate', None), nn.Parameter):
            raise RuntimeError('CSP Tribrid must preserve dense_gate; missing parameter')
        layout = 'native_csp'
    elif all(hasattr(block, n) for n in ('cv1', 'cv2')):
        first = _conv(block.cv1, 'cv1')
        fused = _conv(block.cv2, 'cv2')
        split = first.out_channels
        if split % 2:
            raise RuntimeError('C2f split has odd number of channels')
        c = split//2
        expected = (2+len(block.m))*c
        if fused.in_channels != expected:
            raise RuntimeError('Unsupported C2f concat: found '+str(fused.in_channels)+' expected '+str(expected))
        if hasattr(block, 'dense_local') or hasattr(block, 'dense_gate'):
            raise RuntimeError('Unexpected dense branch in C2f ABI; refusing to discard it')
        layout = 'archived_c2f'
    else:
        raise RuntimeError('Unrecognized Tribrid ABI; expected conv1/conv2/conv3 or cv1/cv2')
    if getattr(block, 'c', c) != c:
        raise RuntimeError('Cached Tribrid hidden width conflicts with conv geometry')
    return layout, c


def _rms(x):
    return (x.float().square().mean(dim=(-2,-1), keepdim=True)+1e-12).sqrt()


class A41DetailSPI(NoClippingSPI):
    """Small positive energy-bounded SPI correction on top of native P3."""
    def __init__(self, in_channels_dark2, out_channels_p3):
        super().__init__(in_channels_dark2, out_channels_p3)
        self.detail_gate_logit = nn.Parameter(torch.tensor(-0.8))
        self.enabled = True
        self.telemetry = False
        self.last_probe = None

    def forward(self, dark2_feat, p3_feat):
        if not self.enabled:
            return p3_feat
        mixed = super().forward(dark2_feat, p3_feat)
        delta = mixed.float()-p3_feat.float()
        ratio = _rms(p3_feat)/(_rms(p3_feat)+_rms(delta))
        correction = 0.25*torch.sigmoid(self.detail_gate_logit.float())*delta*ratio
        if self.telemetry:
            self.last_probe = (p3_feat.detach(), correction.detach())
        return p3_feat+correction.to(p3_feat.dtype)


class A41Tribrid(C2f_Tribrid):
    """Scale-explicit, low-energy context correction WITHOUT changing native CSP topology.

    Preserves all trainable native modules under their original state_dict keys.
    For the deployed CSPLayer ABI it also preserves dense_local and dense_gate,
    which the previous A4.1 C2f-only wrapper accidentally discarded.
    """
    def __init__(self, original, level_name, initial_gate):
        layout, c = _identify_layout(original)
        nn.Module.__init__(self)
        self.layout = layout
        self.c = c
        self.level_name = str(level_name)
        if layout == 'native_csp':
            self.conv1 = original.conv1
            self.conv2 = original.conv2
            self.conv3 = original.conv3
            self.m = original.m
            self.dense_local = original.dense_local
            self.dense_gate = nn.Parameter(original.dense_gate.detach().clone())
            # A very small but non-zero dense-local contribution allows gradients
            # to reach its internals from step zero, instead of gate-only learning.
            if torch.count_nonzero(self.dense_gate.detach()) == 0:
                with torch.no_grad():
                    self.dense_gate.fill_(math.atanh(0.08))
            self.enable_dense_local = bool(getattr(original, 'enable_dense_local', True))
        else:
            self.cv1 = original.cv1
            self.cv2 = original.cv2
            self.m = original.m
            self.enable_dense_local = False
        self.global_mixer = original.global_mixer
        self.gate = nn.Parameter(original.gate.detach().clone())
        with torch.no_grad():
            self.gate.fill_(math.atanh(float(initial_gate)))
        self.enable_global_context = bool(getattr(original, 'enable_global_context', True))
        self.context_enabled = True
        self.telemetry = False
        self.last_probe = None
        self.last_dense_probe = None
        self.strict_nonfinite = False

        desired_dilation = 2 if level_name == 'C3_p3' else 1
        found_rmb = 0
        found_attention = 0
        for module in self.global_mixer:
            if isinstance(module, UnifiedRMBConv):
                found_rmb += 1
                dw = module.rep_dw_dilated
                if getattr(dw, 'deploy', False):
                    raise RuntimeError('Cannot change RepDW dilation after deployment')
                if not hasattr(dw,'dw_3x3') or dw.dw_3x3.kernel_size != (3,3):
                    raise RuntimeError('Unexpected RepDWConv topology')
                dw.dilation = desired_dilation
                dw.padding = desired_dilation
                dw.dw_3x3.dilation = (desired_dilation,desired_dilation)
                dw.dw_3x3.padding = (desired_dilation,desired_dilation)
            if isinstance(module, ContinuousSaliencyAttention):
                found_attention += 1
                if not isinstance(getattr(module,'proj',None), nn.Conv2d):
                    raise RuntimeError('Unexpected attention projection type')
                local = getattr(module,'local_v_conv',None)
                if not isinstance(local, nn.Conv2d):
                    raise RuntimeError('Unexpected local value projection')
                with torch.no_grad():
                    if torch.count_nonzero(module.proj.weight) == 0:
                        nn.init.kaiming_uniform_(module.proj.weight, a=math.sqrt(5))
                        module.proj.weight.mul_(0.015)
                    if module.proj.bias is not None:
                        module.proj.bias.zero_()
                    w = local.weight
                    if w.ndim != 4 or any(k % 2 != 1 for k in w.shape[-2:]):
                        raise RuntimeError('Invalid attention local-value kernel')
                    if torch.count_nonzero(w) == 0:
                        w[:,:,w.shape[-2]//2,w.shape[-1]//2] = 0.015
        if found_rmb != 1 or found_attention != 1:
            raise RuntimeError('Unsupported mixer: need one RMBConv and one ContinuousSaliencyAttention')
        self.rmb_dilation = desired_dilation

    def _add_context(self, native):
        if not (self.context_enabled and self.enable_global_context):
            return native
        # Match the local vNext.2 FP16 safety policy. Autocast is disabled
        # for the entire mixer AND residual merge; weights stay in FP32.
        with torch.autocast(device_type=native.device.type, enabled=False):
            native_fp32 = native.float()
            context = self.global_mixer(native_fp32)
            if self.strict_nonfinite and not bool(torch.isfinite(context).all()):
                raise FloatingPointError('Nonfinite global context at '+self.level_name)
            ratio = _rms(native_fp32)/(_rms(native_fp32)+_rms(context))
            residual = 0.20*torch.tanh(self.gate.float())*context.float()*ratio
            merged = native_fp32+residual
            if self.telemetry:
                self.last_probe = (native.detach(),residual.detach())
        return merged.to(native.dtype)

    def forward(self, x):
        if self.layout == 'native_csp':
            # EXACT base CSPLayer graph in the user's supplied Tribrid:
            # m(conv1(x)), conv2(x), concat, conv3, then dense_local(x).
            native = self.m(self.conv1(x))
            bypass = self.conv2(x)
            native = self._add_context(native)
            out = self.conv3(torch.cat((native,bypass),dim=1))
            if self.enable_dense_local:
                dense = self.dense_local(x)
                if dense.shape != out.shape:
                    raise RuntimeError('Dense residual/output geometry mismatch '+str((dense.shape,out.shape)))
                # Positive, small, RMS-bounded dense correction; original
                # 0.10*tanh(dense_gate) gate form is preserved.
                factor = _rms(out)/(_rms(out)+_rms(dense))
                correction = 0.10*torch.tanh(self.dense_gate.float())*dense.float()*factor
                if self.telemetry:
                    self.last_dense_probe = (out.detach(),correction.detach())
                out = out+correction.to(out.dtype)
            return out
        if self.layout == 'archived_c2f':
            ys = list(self.cv1(x).chunk(2,dim=1))
            for module in self.m:
                ys.append(module(ys[-1]))
            ys[-1] = self._add_context(ys[-1])
            return self.cv2(torch.cat(ys,dim=1))
        raise RuntimeError('Unknown saved Tribrid layout '+str(self.layout))

    def switch_to_deploy(self):
        for module in self.global_mixer:
            if hasattr(module,'switch_to_deploy'):
                module.switch_to_deploy()
        if self.layout == 'native_csp' and hasattr(self.dense_local,'switch_to_deploy'):
            self.dense_local.switch_to_deploy()


def prepare_a41_neck(neck):
    neck = prepare_backbone(neck,remove_spi_clipping=True)
    old = neck.spi_highway
    if not isinstance(old, NoClippingSPI):
        raise RuntimeError('SPI source differs from audited full-fidelity ABI')
    detail = A41DetailSPI(old.projector.conv.in_channels//4,
                          old.router[-1].out_channels//2)
    missing,unexpected = detail.load_state_dict(old.state_dict(),strict=False)
    if missing != ['detail_gate_logit'] or unexpected:
        raise RuntimeError(f'SPI migration unexpected missing={missing} unexpected={unexpected}')
    neck.spi_highway = detail
    for name,gate in [('C3_p4',0.16),('C3_p3',0.12),('C3_n3',0.18),('C3_n4',0.20)]:
        block = getattr(neck,name,None)
        if not isinstance(block,C2f_Tribrid):
            raise RuntimeError('Missing Tribrid block '+name)
        setattr(neck,name,A41Tribrid(block,name,gate))
    return neck
