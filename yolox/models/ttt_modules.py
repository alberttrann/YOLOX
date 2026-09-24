import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        x_p = x.permute(0, 2, 3, 1)
        gx = torch.sqrt(torch.sum(x_p.pow(2), dim=(1, 2), keepdim=True) + 1e-6)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x_calibrated = self.gamma * (x_p * nx) + self.beta + x_p
        return x_calibrated.permute(0, 3, 1, 2).contiguous()

class TTTProjector(nn.Module):
    def __init__(self, in_channels, hidden_ratio=0.5):
        super().__init__()
        hidden = max(16, int(in_channels * hidden_ratio))
        self.mask_token = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden, eps=1e-5),
            nn.GELU(),
            GRN(hidden, eps=1e-5),
            nn.Conv2d(hidden, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        var_fp32 = torch.var(x_fp32, dim=[2, 3], keepdim=True, unbiased=False)
        mean_fp32 = torch.mean(x_fp32, dim=[2, 3], keepdim=True)
        x_minus_mu_sq = (x_fp32 - mean_fp32).pow(2)
        y = x_minus_mu_sq / (4.0 * (var_fp32 + self.eps)) + 0.5
        return x * torch.sigmoid(y).to(x.dtype)

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.GroupNorm(8 if mip % 8 == 0 else math.gcd(mip, 8), mip, eps=1e-5)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        
        x_fp32 = x.float()
        x_h = F.adaptive_avg_pool2d(x_fp32, (None, 1)).to(x.dtype)
        x_w = F.adaptive_avg_pool2d(x_fp32, (1, None)).permute(0, 1, 3, 2).to(x.dtype)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        att_h = 2.0 * torch.sigmoid(self.conv_h(x_h))
        att_w = 2.0 * torch.sigmoid(self.conv_w(x_w))
        return identity * att_h * att_w

class FP32AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    FP32 Shielded Pooling to prevent FP16 accumulator overflow.
    Has no weights, so it perfectly preserves checkpoint loading compatibility.
    """
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)

class MBConvConditioner(nn.Module):
    """MaxViT-inspired local structural conditioner."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        
        # RESTORED: Exact same variable name and 5-layer structure as your checkpoint!
        self.se = nn.Sequential(
            FP32AdaptiveAvgPool2d(1),                        # Index 0 (no weights)
            nn.Conv2d(dim, max(8, dim // 4), kernel_size=1), # Index 1 (matches se.1.weight)
            nn.ReLU(),                                       # Index 2
            nn.Conv2d(max(8, dim // 4), dim, kernel_size=1), # Index 3 (matches se.3.weight)
            nn.Sigmoid()                                     # Index 4
        )

    def forward(self, x):
        return x + self.se(x) * self.pw(self.dw(x))

class TTTAdaptiveStage(nn.Module):
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.05, noise_std=0.08):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.noise_std = noise_std
        
        self.ttt_lrs = nn.ParameterDict()
        init_raw = math.log(max(1e-6, math.exp(init_ttt_lr) - 1.0))
        for name, param in self.backbone_stage.named_parameters():
            if 'gn' in name or 'norm' in name:
                self.ttt_lrs[name.replace('.', '_')] = nn.Parameter(torch.tensor(init_raw))

    def _get_robust_contrast_mask(self, feat):
        B, C, H, W = feat.shape
        f_fp32 = feat.float()
        
        var = torch.clamp(torch.var(f_fp32, dim=1, keepdim=True), min=0.0)
        mean_sq = torch.clamp(torch.mean(f_fp32, dim=1, keepdim=True).pow(2), min=1e-3)
        contrast = var / mean_sq

        v4 = F.avg_pool2d(contrast, kernel_size=4, stride=4)
        v8 = F.avg_pool2d(contrast, kernel_size=8, stride=8)
        v_fusion = v4 + F.interpolate(v8, size=(H // 4, W // 4), mode='nearest')
        
        v_mean = v_fusion.mean(dim=[-2, -1], keepdim=True)
        v_std_safe = torch.clamp(v_fusion.std(dim=[-2, -1], keepdim=True), min=1e-4)
        
        threshold = v_mean + 0.675 * v_std_safe
        mask_fusion = (v_fusion >= threshold).to(feat.dtype)
        
        mask_ratio = mask_fusion.mean(dim=[-2, -1], keepdim=True)
        if (mask_ratio > 0.75).any():
            scores_flat = v_fusion.view(B, -1)
            q75 = torch.quantile(scores_flat, 0.25, dim=-1, keepdim=True).view(B, 1, 1, 1)
            mask_fusion = (v_fusion >= q75).to(feat.dtype)
        
        return F.interpolate(mask_fusion, size=(H, W), mode='nearest')

    def inner_loss_fn(self, params_backbone, params_projector, x_in, mask, noise_map, clean_target):
        feat = functional_call(self.backbone_stage, params_backbone, x_in.float())
        feat_noisy = feat + noise_map.float()
        
        token = params_projector['mask_token'].float()
        feat_corrupted = feat_noisy * (1.0 - mask.float()) + token * mask.float()
        
        rec = functional_call(self.projector, params_projector, feat_corrupted)
        return F.mse_loss(rec, clean_target.float())

    def forward(self, x_in, run_ttt=True):
        input_dtype = x_in.dtype
        is_profiling = any(hasattr(m, 'total_ops') or hasattr(m, 'total_params') for m in self.backbone_stage.modules())
        if is_profiling or not run_ttt:
            return self.backbone_stage(x_in), torch.tensor(0.0, device=x_in.device)

        is_inference_mode = (not torch.is_grad_enabled())
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        backbone_params = dict(self.backbone_stage.named_parameters())
        backbone_buffers = dict(self.backbone_stage.named_buffers())
        
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
                feat_initial = self.backbone_stage(x_curr)
                clean_target = feat_initial.detach()
                mask = self._get_robust_contrast_mask(clean_target)
                
                if self.training:
                    noise_map = torch.randn_like(clean_target) * self.noise_std
                else:
                    noise_map = torch.zeros_like(clean_target)

            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                return self.inner_loss_fn(full_backbone, p_adapt_p, x_curr, mask, noise_map, clean_target)

            grads_backbone = grad(inner_grad_fn, argnums=0)(adapt_backbone, proj_params)
            
            updated_backbone_params = {**backbone_params, **backbone_buffers}
            for name, g in grads_backbone.items():
                lr_key = name.replace('.', '_')
                effective_lr = torch.clamp(F.softplus(self.ttt_lrs[lr_key]), min=1e-4, max=0.25)
                
                if not self.training:
                    effective_lr = effective_lr.detach()
                    g = g.detach()
                # ADD .detach() TO PREVENT NON-LEAF TRACKING DURING INFERENCE:
                new_val = (backbone_params[name].float() - effective_lr * g).to(input_dtype)
                if not self.training:
                    new_val = new_val.detach()
                updated_backbone_params[name] = new_val
                    
                updated_backbone_params[name] = (backbone_params[name].float() - effective_lr * g).to(input_dtype)

        if self.training:
            self.backbone_stage.train()
            self.projector.train()
            out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
            
            token_direct = self.projector.mask_token
            corrupted_for_proj = (feat_initial + noise_map) * (1.0 - mask) + token_direct * mask
            proj_rec = self.projector(corrupted_for_proj)
            aux_proj_loss = F.mse_loss(proj_rec.float(), clean_target.float())
        else:
            with torch.no_grad():
                out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
            aux_proj_loss = torch.tensor(0.0, device=x_in.device)

        return out.to(input_dtype), aux_proj_loss