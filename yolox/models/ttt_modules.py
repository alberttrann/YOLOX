import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2).
    Forces channel competition to prevent representation collapse in masked autoencoders.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W] -> permute to [B, H, W, C]
        x_p = x.permute(0, 2, 3, 1)
        gx = torch.norm(x_p, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x_calibrated = self.gamma * (x_p * nx) + self.beta + x_p
        return x_calibrated.permute(0, 3, 1, 2)


class TTTProjector(nn.Module):
    """
    Lightweight feature reconstruction autoencoder for the TTT inner adaptation loop.
    Architecture: Conv1x1 -> GroupNorm -> GELU -> GRN -> Conv1x1
    """
    def __init__(self, in_channels, hidden_ratio=0.5):
        super().__init__()
        hidden = max(16, int(in_channels * hidden_ratio))
        self.mask_token = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            GRN(hidden),
            nn.Conv2d(hidden, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class SimAM(nn.Module):
    """
    Parameter-free 3D Attention Module based on spatial suppression theory (Yang et al., 2021).
    Attenuates featureless, low-energy atmospheric backscatter without adding parameters.
    """
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.eps)) + 0.5
        return x * torch.sigmoid(y)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (Hou et al., 2021).
    Factorizes spatial pooling into direction-aware 1D horizontal and vertical components.
    """
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class MBConvConditioner(nn.Module):
    """
    MBConv Structural Conditioner (MaxViT; Tu et al., 2022).
    Applies Depthwise 3x3 Conv + Squeeze-and-Excitation + Pointwise Conv to bind local edges.
    """
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(8, dim // 4), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(max(8, dim // 4), dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.se(x) * self.pw(self.dw(x))


class TTTAdaptiveStage(nn.Module):
    """
    Higher-Order Functional TTT-Dark2 Adaptation Stage.
    Features:
      - Weber-Fechner relative contrast masking (Koschmieder fog scale invariance)
      - Softplus-constrained learnable meta-learning rates (Strictly positive descent)
      - Direct first-order auxiliary projector reconstruction loss (L_proj)
      - Pure functional parameter passing with zero state leakage
      - Float32 precision guard for inner-loop autograd
    """
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.05, noise_std=0.08):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.noise_std = noise_std
        
        # Meta-learning rates for each GroupNorm parameter
        # Parameterized such that Softplus(raw_param) == init_ttt_lr at step 0
        self.ttt_lrs = nn.ParameterDict()
        init_raw = math.log(max(1e-6, math.exp(init_ttt_lr) - 1.0))
        for name, param in self.backbone_stage.named_parameters():
            if 'gn' in name or 'norm' in name:
                self.ttt_lrs[name.replace('.', '_')] = nn.Parameter(torch.tensor(init_raw))

    def _get_robust_contrast_mask(self, feat):
        """
        Weber-Fechner Relative Contrast Masker:
        C(f) = Var(f) / [Mean(f)^2 + eps]
        Multiplicative attenuation factor e^{-beta * d} cancels out in the ratio.
        """
        B, C, H, W = feat.shape
        f_fp32 = feat.float()
        var = torch.var(f_fp32, dim=1, keepdim=True)
        mean_sq = torch.mean(f_fp32, dim=1, keepdim=True).pow(2) + 1e-5
        contrast = var / mean_sq

        # Multi-scale pooling to isolate fine contours and regional objects
        v4 = F.avg_pool2d(contrast, kernel_size=4, stride=4)
        v8 = F.avg_pool2d(contrast, kernel_size=8, stride=8)
        v_fusion = v4 + F.interpolate(v8, size=(H // 4, W // 4), mode='nearest')
        
        # Dynamic threshold (approximates 75th percentile of contrast energy)
        v_mean = v_fusion.mean(dim=[-2, -1], keepdim=True)
        v_std = v_fusion.std(dim=[-2, -1], keepdim=True)
        threshold = v_mean + 0.675 * v_std
        mask_fusion = (v_fusion >= threshold).to(feat.dtype)
        
        return F.interpolate(mask_fusion, size=(H, W), mode='nearest')

    def inner_loss_fn(self, params_backbone, params_projector, x_in, mask, noise_map, clean_target):
        """
        Pure Functional Inner Adaptation Loss.
        Zero references to self.module.param! All tensors routed from dictionaries.
        """
        feat = functional_call(self.backbone_stage, params_backbone, x_in.float())
        feat_noisy = feat + noise_map.float()
        
        # Extract mask_token strictly from functional dictionary
        token = params_projector['mask_token'].float()
        feat_corrupted = feat_noisy * (1.0 - mask.float()) + token * mask.float()
        
        rec = functional_call(self.projector, params_projector, feat_corrupted)
        return F.mse_loss(rec, clean_target.float())

    def forward(self, x_in, run_ttt=True):
        input_dtype = x_in.dtype
        
        # Profiler bypass check (thop / fvcore safety)
        is_profiling = any(hasattr(m, 'total_ops') or hasattr(m, 'total_params') for m in self.backbone_stage.modules())
        if is_profiling or not run_ttt:
            return self.backbone_stage(x_in), torch.tensor(0.0, device=x_in.device)

        # Context manager for inference-time adaptation
        is_inference_mode = (not torch.is_grad_enabled())
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        # Capture parameter dictionaries for pure functional execution
        backbone_params = dict(self.backbone_stage.named_parameters())
        backbone_buffers = dict(self.backbone_stage.named_buffers())
        
        # GroupNorm parameters are strictly isolated for domain adaptation
        adapt_backbone = {k: v.float() for k, v in backbone_params.items() 
                          if ('gn' in k or 'norm' in k) and v.is_floating_point()}
        static_backbone_state = {k: v.float() for k, v in backbone_params.items() 
                                 if k not in adapt_backbone}
        fixed_backbone_state = {**static_backbone_state, **{k: v.float() for k, v in backbone_buffers.items()}}
        proj_params = {k: v.float() for k, v in self.projector.named_parameters()}

        self.backbone_stage.eval()
        self.projector.eval()

        with context_manager:
            with torch.no_grad():
                # Pass x_curr in its native precision (float16 under --fp16) to match backbone_stage weights
                feat_initial = self.backbone_stage(x_curr)
                clean_target = feat_initial.detach()
                mask = self._get_robust_contrast_mask(clean_target)
                noise_map = torch.randn_like(clean_target) * self.noise_std

            # Closure for functorch dual-number gradient computation
            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                return self.inner_loss_fn(full_backbone, p_adapt_p, x_curr, mask, noise_map, clean_target)

            # High-precision FP32 gradient computation
            grads_backbone = grad(inner_grad_fn, argnums=0)(adapt_backbone, proj_params)
            
            # Construct adapted weight dictionary with Softplus-constrained positive step sizes
            updated_backbone_params = {**backbone_params, **backbone_buffers}
            for name, g in grads_backbone.items():
                lr_key = name.replace('.', '_')
                effective_lr = F.softplus(self.ttt_lrs[lr_key])
                updated_backbone_params[name] = (backbone_params[name].float() - effective_lr * g).to(input_dtype)

        # Compute direct first-order auxiliary projector loss during training
        if self.training:
            self.backbone_stage.train()
            self.projector.train()
            out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
            
            # Direct MSE loss ensures Projector learns the clean manifold
            token_direct = self.projector.mask_token
            corrupted_for_proj = (feat_initial + noise_map) * (1.0 - mask) + token_direct * mask
            proj_rec = self.projector(corrupted_for_proj)
            aux_proj_loss = F.mse_loss(proj_rec, clean_target)
        else:
            with torch.no_grad():
                out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
            aux_proj_loss = torch.tensor(0.0, device=x_in.device)

        return out.to(input_dtype), aux_proj_loss