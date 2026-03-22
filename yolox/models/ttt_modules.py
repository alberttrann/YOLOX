import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib

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
    Uses 85% Masking to force extreme structural reasoning.
    """
    def __init__(self, stage_module, in_channels, init_ttt_lr=0.08, noise_std=0.12):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.noise_std = noise_std
        
        self.ttt_lrs = nn.ParameterDict()
        for name, param in self.backbone_stage.named_parameters():
            if 'gn' in name or 'norm' in name:
                # Meta-LR is now significantly more aggressive (0.08)
                self.ttt_lrs[name.replace('.', '_')] = nn.Parameter(torch.tensor(init_ttt_lr))

    def _get_robust_variance_mask(self, feat, ratio=0.85): # Extreme 85% Mask
        B, C, H, W = feat.shape
        var = torch.var(feat, dim=1, keepdim=True)
        v_fusion = F.avg_pool2d(var, 4, stride=4) # Fine-grained focus
        scores_fp32 = v_fusion.view(B, -1).float()
        threshold = torch.quantile(scores_fp32, ratio, dim=1, keepdim=True)
        mask = (v_fusion >= threshold.view(B, 1, 1, 1)).to(feat.dtype)
        return F.interpolate(mask, size=(H, W), mode='nearest')

    def inner_loss_fn(self, params, projector_params, x_in, mask, noise_map, clean_target):
        feat = functional_call(self.backbone_stage, params, x_in)
        # Apply Denoising + Masking
        feat_noisy = feat + noise_map
        feat_corrupted = feat_noisy * (1 - mask) + self.projector.mask_token * mask
        rec = functional_call(self.projector, projector_params, feat_corrupted)
        return F.mse_loss(rec, clean_target)

    def forward(self, x_in, run_ttt=True):
        # --- PROFILER BYPASS (Fixes thop collision) ---
        # 1. Check if we are in a 'thop' profiling context (Total Ops tracking)
        # 2. Check if we are in a 'fvcore' or other common profiler context
        is_profiling = False
        for m in self.backbone_stage.modules():
            if hasattr(m, 'total_ops') or hasattr(m, 'total_params'):
                is_profiling = True
                break
        
        if is_profiling or not run_ttt:
            return self.backbone_stage(x_in)

        # Capture precision to prevent DType Mismatch in Neck
        input_dtype = x_in.dtype

        # --- SETUP GRADIENT CONTEXT ---
        # TTT during inference often runs in no_grad mode. must force grad enablement.
        is_inference_mode = (not torch.is_grad_enabled())
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        # --- CAPTURE & PARTITION STATE ---
        backbone_params = dict(self.backbone_stage.named_parameters())
        backbone_buffers = dict(self.backbone_stage.named_buffers())
        proj_params = dict(self.projector.named_parameters())

        # Isolate adaptable Norm parameters (GroupNorm has no buffers)
        adapt_backbone = {k: v for k, v in backbone_params.items() 
                         if ('gn' in k or 'norm' in k) and v.is_floating_point()}
        static_backbone_state = {k: v for k, v in backbone_params.items() 
                                if k not in adapt_backbone}
        
        fixed_backbone_state = {**static_backbone_state, **backbone_buffers}

        # Force Eval mode for the inner loop to prevent batch stat updates
        self.backbone_stage.eval()
        self.projector.eval()

        with context_manager:
            # Generate Adaptation Targets
            with torch.no_grad():
                feat_initial = self.backbone_stage(x_curr)
                clean_target = feat_initial.detach()
                mask = self._get_robust_variance_mask(clean_target)
                
                # --- PATCH: CURRICULUM NOISE ---
                epoch = getattr(self, "current_epoch", 10) # Default to 10 if not found
                curr_noise = self.noise_std * min(1.0, epoch / 10.0)
                noise_map = torch.randn_like(clean_target) * curr_noise

            # --- CORE FUNCTIONAL TTT STEP ---
            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                return self.inner_loss_fn(full_backbone, p_adapt_p, x_curr, mask, noise_map, clean_target)

            # Compute d_Loss / d_BackboneNorms
            # Note: also pass proj_params to grad to allow joint optimization
            grads_backbone = grad(inner_grad_fn, argnums=0)(adapt_backbone, proj_params)
            
            # --- CONSTRUCT ADAPTED WEIGHT SPACE ---
            updated_backbone_params = {**backbone_params, **backbone_buffers}
            for name, g in grads_backbone.items():
                lr_key = name.replace('.', '_')
                lr = self.ttt_lrs[lr_key]
                # Cast grad to param type for safe functional update
                updated_backbone_params[name] = updated_backbone_params[name] - lr * g.to(updated_backbone_params[name].dtype)

        # --- RESTORE TRAINING STATE ---
        if self.training:
            self.backbone_stage.train()
            self.projector.train()

        # --- FINAL ADAPTED FORWARD ---
        # uses the 'updated_backbone_params' dict locally for this image only
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
    DeepSeek-V3.2 DSA for Object Detection.
    Uses Augmented Lightning Indexer for saliency selection.
    """
    def __init__(self, dim, sparsity_ratio=0.1):
        super().__init__()
        self.dim = dim
        self.ratio = sparsity_ratio
        # Augmented Lightning Indexer (Saliency Head)
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
        K = max(1, int(num_tokens * self.ratio))

        # 1. Saliency Scoring (Lightning Indexer)
        scores = self.indexer(x).view(B, -1) # [B, HW]
        
        # 2. Top-K Token Selection
        _, topk_indices = torch.topk(scores, K, dim=1) # [B, K]
        topk_indices = topk_indices.long()
        # 3. Sparse Projection
        qkv = self.qkv(x).view(B, 3*C, -1) # [B, 3C, HW]
        q, k, v = torch.chunk(qkv, 3, dim=1) # [B, C, HW]
        
        # Select sparse keys and values based on saliency
        # forces global context to focus only on informative regions
        k_s = torch.gather(k, 2, topk_indices.unsqueeze(1).expand(-1, C, -1)) # [B, C, K]
        v_s = torch.gather(v, 2, topk_indices.unsqueeze(1).expand(-1, C, -1)) # [B, C, K]

        # 4. Sparse Cross-Attention (Query is full-map, K/V are sparse)
        # Scaling factor
        scale = C ** -0.5
        attn = (q.transpose(-2, -1) @ k_s) * scale # [B, HW, K]
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate context
        context = (v_s @ attn.transpose(-2, -1)).view(B, C, H, W) # [B, C, H, W]
        return self.proj(context)
    
#PHASE 3

class EngramMemoryBank(nn.Module):
    def __init__(self, num_classes, latent_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.temperature = 5.0 
        self.prototype_layer = nn.Linear(latent_dim, num_classes, bias=False)
        nn.init.orthogonal_(self.prototype_layer.weight)

    # REMOVED objectness_mask argument
    def forward(self, x_latent, uncertainty_gate): 
        x_norm = F.normalize(x_latent, p=2, dim=-1)
        p_norm = F.normalize(self.prototype_layer.weight, p=2, dim=-1)
        
        attn_scores = torch.matmul(x_norm, p_norm.t()) * self.temperature
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        memory_retrieved = torch.matmul(attn_weights, self.prototype_layer.weight)
        
        # Only gate by uncertainty
        return memory_retrieved * uncertainty_gate

class UncertaintyEstimator(nn.Module):
    """
    NUCLEAR VERSION: Full-Pattern Uncertainty Estimation.
    Learns to detect OOD features by analyzing the 128D latent distribution.
    """
    def __init__(self, dim):
        super().__init__()
        # dim = 128
        self.gate_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        # Initialize bias to a high value (e.g., 2.0) so sigmoid(2.0) ~ 0.88
        nn.init.constant_(self.gate_fc[-2].bias, 2.0)

    def forward(self, x):
        # x: [B, HW, 128]
        # We pass the full latent vector into the MLP.
        # This allows the model to learn complex 'identity confusion' patterns.
        return self.gate_fc(x) # [B, HW, 1]