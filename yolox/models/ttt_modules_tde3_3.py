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

        self._is_profiling_mode = False

    def _get_robust_variance_mask(self, feat, ratio=0.85):
        """TDE 2.0 Single-Scale Aggressive Masking"""
        B, C, H, W = feat.shape
        var = torch.var(feat, dim=1, keepdim=True)
        v_fusion = F.avg_pool2d(var, 4, stride=4)
        scores_fp32 = v_fusion.view(B, -1).float()
        threshold = torch.quantile(scores_fp32, ratio, dim=1, keepdim=True)
        mask = (v_fusion >= threshold.view(B, 1, 1, 1)).to(feat.dtype)
        return F.interpolate(mask, size=(H, W), mode='nearest')

    def inner_loss_fn(self, params, projector_params, x_in, mask, clean_target):
        feat = functional_call(self.backbone_stage, params, x_in)
        
        # SAFETY LOCK 1: Bounded Stochastic Fog
        # Prevents the model from memorizing a constant inverse scalar.
        # Forces true adaptive contrast restoration.
        alpha = 0.5 + 0.5 * torch.rand(1, device=feat.device, dtype=feat.dtype)
        feat_foggy = feat * alpha 
        
        # Gaussian Noise (Simulate Rain/Snow)
        noise_map = torch.randn_like(feat) * self.noise_std
        feat_noisy = feat_foggy + noise_map
        
        # Masking
        feat_corrupted = feat_noisy * (1 - mask) + self.projector.mask_token * mask
        rec = functional_call(self.projector, projector_params, feat_corrupted)
        
        # TDE 2.0 Pure MSE
        return F.mse_loss(rec, clean_target)

    def forward(self, x_in, run_ttt=True):
        # 0. EXPLICIT Profiler Bypass
        # We only bypass if the flag is explicitly flipped.
        # This prevents accidental bypassing due to lingering 'total_ops' attributes.
        if getattr(self, '_is_profiling_mode', False) or not run_ttt:
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

            # --- CORE FUNCTIONAL TTT STEP ---
            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                return self.inner_loss_fn(full_backbone, p_adapt_p, x_curr, mask, clean_target)

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
    """
    NUCLEAR VERSION: Hyperspherical Associative Memory.
    Forces hard-decision identity restoration.
    """
    def __init__(self, num_classes, latent_dim=128, temperature=50.0):
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
    TDE 3.0 DUAL-HEAD GATE + SAFETY LOCK:
    Independently evaluates Semantic Anomalies and Physical Anomalies.
    Guarantees a minimum gradient flow to prevent Memory Bank death.
    """
    def __init__(self, in_channels, latent_dim=128):
        super().__init__()
        self.semantic_critic = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.physical_critic = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_feat, x_latent):
        B, C, H, W = x_feat.shape
        
        gate_semantic = self.semantic_critic(x_latent) 
        
        variance = torch.var(x_feat, dim=1, keepdim=True)
        spatial_variance = F.max_pool2d(variance, kernel_size=3, stride=1, padding=1)
        
        magnitude = torch.norm(x_feat, p=2, dim=1, keepdim=True)
        spatial_magnitude = F.avg_pool2d(magnitude, kernel_size=3, stride=1, padding=1)
        
        stats = torch.cat([spatial_variance, spatial_magnitude], dim=1)
        stats_flat = stats.view(B, 2, H * W).permute(0, 2, 1)
        
        gate_physical = self.physical_critic(stats_flat) 
        
        # Probabilistic OR
        gate_final = gate_semantic + gate_physical - (gate_semantic * gate_physical)
        
        # SAFETY LOCK 2: Minimum Information Flow
        # Ensures the Memory Bank always receives a 1% gradient trickle during training
        # to prevent prototype collapse in early epochs.
        if self.training:
            gate_final = torch.clamp(gate_final, min=0.01)
            
        return gate_final