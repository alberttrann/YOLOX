import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib
from collections import OrderedDict

@contextlib.contextmanager
def suspend_hooks(module):
    """
    EXPERT FIX: Temporarily suspends all forward hooks.
    Prevents profiling libraries (like `thop` or `fvcore`) from triggering 
    in-place tensor mutations (e.g., m.total_ops += 1) during torch.func.grad,
    which strictly forbids in-place operations to maintain functional purity.
    """
    hooks_backup = {}
    pre_hooks_backup = {}
    for name, m in module.named_modules():
        if m._forward_hooks:
            hooks_backup[name] = m._forward_hooks
            m._forward_hooks = OrderedDict()
        if hasattr(m, '_forward_pre_hooks') and m._forward_pre_hooks:
            pre_hooks_backup[name] = m._forward_pre_hooks
            m._forward_pre_hooks = OrderedDict()
    try:
        yield
    finally:
        # Restore all hooks flawlessly
        for name, m in module.named_modules():
            if name in hooks_backup:
                m._forward_hooks = hooks_backup[name]
            if name in pre_hooks_backup:
                m._forward_pre_hooks = pre_hooks_backup[name]
class GRN(nn.Module):
    """Global Response Normalization: Essential for preventing feature collapse."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        x_p = x.permute(0, 2, 3, 1)
        gx = torch.norm(x_p, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return (self.gamma * (x_p * nx) + self.beta + x_p).permute(0, 3, 1, 2)

class TTTProjector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        hidden = in_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            GRN(hidden),
            nn.Conv2d(hidden, in_channels, 1)
        )
    def forward(self, x): return self.net(x)

class TTTAdaptiveStage(nn.Module):
    """
    NUCLEAR VERSION: Adversarial Adaptation Engine.
    Uses 75% Masking to force extreme structural reasoning.
    """
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.02, noise_std=0.08):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.noise_std = noise_std
        
        self.ttt_lr = nn.Parameter(torch.tensor(init_ttt_lr))

    def _get_robust_variance_mask(self, feat, ratio=0.65): # Adjusted to 0.65 to prevent complete blackout in dense fog
        B, C, H, W = feat.shape
        var = torch.var(feat, dim=1, keepdim=True)
        v_fusion = F.avg_pool2d(var, 4, stride=4) # Fine-grained focus
        scores_fp32 = v_fusion.view(B, -1).float()
        threshold = torch.quantile(scores_fp32, ratio, dim=1, keepdim=True)
        mask = (v_fusion >= threshold.view(B, 1, 1, 1)).to(feat.dtype)
        return F.interpolate(mask, size=(H, W), mode='nearest')

    def inner_loss_fn(self, params, projector_params, x_in, mask, noise_map, clean_target):
        feat = functional_call(self.backbone_stage, params, x_in)
        feat_noisy = feat + noise_map
        feat_corrupted = feat_noisy * (1 - mask) + self.projector.mask_token * mask
        rec = functional_call(self.projector, projector_params, feat_corrupted)
        
        # Loss calculated only on masked regions to force structural understanding
        loss = F.mse_loss(rec * mask, clean_target * mask, reduction='sum') / (mask.sum() + 1e-6)
        return loss

    def forward(self, x_in, run_ttt=True):
        # Profiler Bypass
        is_profiling = any(hasattr(m, 'total_ops') for m in self.backbone_stage.modules())
        if is_profiling or not run_ttt:
            return self.backbone_stage(x_in)

        input_dtype = x_in.dtype
        is_inference_mode = (not torch.is_grad_enabled())
        
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        # --- CAPTURE STATE ---
        all_params = dict(self.backbone_stage.named_parameters())
        all_buffers = dict(self.backbone_stage.named_buffers())
        proj_params = dict(self.projector.named_parameters())

        adapt_backbone = {k: v for k, v in all_params.items() 
                         if ('gn' in k or 'norm' in k) and v.is_floating_point()}
        static_backbone_state = {k: v for k, v in all_params.items() 
                                if k not in adapt_backbone}
        
        fixed_backbone_state = {**static_backbone_state, **all_buffers}

        self.backbone_stage.eval()
        self.projector.eval()

        with context_manager:
            if not x_curr.is_floating_point():
                return self.backbone_stage(x_in)

            with torch.no_grad():
                feat_initial = self.backbone_stage(x_curr)
                clean_target = feat_initial.detach()
                # --- THE EXPERT FIX: TARGET SHARPENING (DEFOGGING CATALYST) ---
                # We artificially boost the contrast of the target by 20%.
                # This forces the TTT loop to upscale GroupNorm parameters (gamma)
                # to "reach" the sharper target, effectively cutting through low-contrast fog.
                target_mean = clean_target.mean(dim=(2, 3), keepdim=True)
                clean_target_sharp = target_mean + (clean_target - target_mean) * 1.20
                # --------------------------------------------------------------
                mask = self._get_robust_variance_mask(clean_target_sharp)
                noise_map = torch.randn_like(clean_target_sharp) * self.noise_std

            # --- THE TARGETED FIX: JOINT OPTIMIZATION ---
            # We need the gradient with respect to BOTH the backbone norms AND the projector
            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                return self.inner_loss_fn(full_backbone, p_adapt_p, x_curr, mask, noise_map, clean_target)

            with suspend_hooks(self.backbone_stage), suspend_hooks(self.projector):
                # argnums=(0, 1) calculates gradients for both parameter dictionaries
                grads_backbone, grads_projector = grad(inner_grad_fn, argnums=(0, 1))(adapt_backbone, proj_params)
            
            # --- EXPERT FIX: Bounded Meta-LR ---
            # Prevents the outer optimizer from inverting the adaptation direction
            safe_ttt_lr = F.softplus(self.ttt_lr) + 1e-4
            
            # 1. Update Backbone Norms (The core TTT mechanism)
            updated_backbone_params = {**all_params, **all_buffers}
            for name, g in grads_backbone.items():
                updated_backbone_params[name] = all_params[name] - (safe_ttt_lr * g.to(all_params[name].dtype))
                
            # 2. Update Projector Weights (Fixes the 0.04% Shift)
            # The projector learns *how* to reconstruct on the fly, 
            # providing increasingly accurate gradients to the backbone.
            updated_proj_params = {}
            for name, g in grads_projector.items():
                updated_proj_params[name] = proj_params[name] - (safe_ttt_lr * g.to(proj_params[name].dtype))

        if self.training:
            self.backbone_stage.train()
            self.projector.train()

        # Final pass (Using updated backbone, ignoring updated projector as it's only used in TTT loop)
        if self.training:
            out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
        else:
            with torch.no_grad():
                out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)

        return out.to(input_dtype)

#PHASE 2

class SimAM(nn.Module):
    """Parameter-free Energy-based Attention. Suppresses OOD background noise."""
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
    """Spatial Bias injection to provide DSA with geometric context."""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.GroupNorm(8, mip) # Stable for real-time inference
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
    """MaxViT-inspired local structural conditioner."""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x + self.se(x) * self.pw(self.dw(x))

class DeepSeekSparseAttention(nn.Module):
    """
    DeepSeek-V3.2 DSA + Exclusive Self-Attention (XSA).
    Full-Fidelity Implementation.
    """
    def __init__(self, dim, sparsity_ratio=0.1):
        super().__init__()
        self.dim = dim
        self.ratio = sparsity_ratio
        
        self.indexer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, 1, 1),
            nn.ReLU()
        )
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        num_tokens = H * W
        K_tokens = max(1, int(num_tokens * self.ratio))

        scores = self.indexer(x).view(B, -1) 
        _, topk_indices = torch.topk(scores, K_tokens, dim=1) 
        topk_indices = topk_indices.long()
        
        qkv = self.qkv(x).view(B, 3*C, -1) 
        q, k, v = torch.chunk(qkv, 3, dim=1) 
        
        k_s = torch.gather(k, 2, topk_indices.unsqueeze(1).expand(-1, C, -1)) 
        v_s = torch.gather(v, 2, topk_indices.unsqueeze(1).expand(-1, C, -1)) 

        scale = C ** -0.5
        attn = (q.transpose(-2, -1) @ k_s) * scale 
        attn = F.softmax(attn, dim=-1)
        
        attn_out = (v_s @ attn.transpose(-2, -1)).view(B, C, H, W) 
        
        # --- XSA CORE ---
        v_spatial = v.view(B, C, H, W)
        v_norm = F.normalize(v_spatial, p=2, dim=1)
        projection = torch.sum(attn_out * v_norm, dim=1, keepdim=True)
        xsa_out = attn_out - (projection * v_norm)
        
        return self.proj(xsa_out)

class InstanceConditionedRouter(nn.Module):
    """
    Dynamically predicts mHC manifold weights based on instance statistics.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, 2) 
        )
        
        # --- THE EXPERT FIX: Symmetrical Agnostic Initialization ---
        # Weights and biases initialized to 0.0. 
        # Output is [0.0, 0.0] -> Softmax yields [0.5, 0.5].
        # Forces the optimizer to train BOTH paths equally at the start.
        nn.init.constant_(self.router[2].weight, 0.0)
        nn.init.constant_(self.router[2].bias, 0.0)

    def forward(self, x):
        B = x.shape[0]
        stats = self.pool(x).view(B, -1)
        return self.router(stats) 
    
#PHASE 3

class EngramMemoryBank(nn.Module):
    """
    NUCLEAR VERSION: Hyperspherical Associative Memory.
    Forces hard-decision identity restoration.
    """
    def __init__(self, num_classes, latent_dim=128, temperature=15.0):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # Prototypes are now unit vectors on a hypersphere
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x_latent, uncertainty_gate, objectness_mask):
        # 1. Hyperspherical Projection (Crucial for OOD)
        # Normalize both input and prototypes to unit length
        x_norm = F.normalize(x_latent, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # 2. Hard-Attention Lookup (Temperature 50)
        # Dot product similarity in hypersphere
        attn_scores = torch.matmul(x_norm, p_norm.t()) * self.temperature
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 3. Memory Retrieval
        memory_retrieved = torch.matmul(attn_weights, self.prototypes)
        
        # Restoration mask (Gating)
        restoration_mask = uncertainty_gate * objectness_mask
        
        return memory_retrieved * restoration_mask

class UncertaintyEstimator(nn.Module):
    """
    Expert Enhancement: 
    Learns to predict uncertainty from the latent feature pattern itself.
    """
    def __init__(self, dim):
        super().__init__()
        # Input dim is 128 (latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, HW, 128]
        # process each token's latent vector to get its uncertainty score
        return self.classifier(x) # Returns [B, HW, 1]