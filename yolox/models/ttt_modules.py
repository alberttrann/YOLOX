import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib

#PHASE 1
class GRN(nn.Module):
    """Global Response Normalization: Prevents feature collapse."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        x_p = x.permute(0, 2, 3, 1) # B, H, W, C
        gx = torch.norm(x_p, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return (self.gamma * (x_p * nx) + self.beta + x_p).permute(0, 3, 1, 2)

class AugmentedLightningIndexer(nn.Module):
    """DSA-inspired learnable saliency head."""
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pw_conv = nn.Conv2d(in_channels, num_heads, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return torch.mean(self.act(self.pw_conv(self.dw_conv(x))), dim=1, keepdim=True)

class TTTProjector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # mask_token here so it is tracked in named_parameters()
        self.mask_token = nn.Parameter(torch.randn(1, in_channels, 1, 1))
        hidden = in_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            GRN(hidden),
            nn.Conv2d(hidden, in_channels, 1)
        )
    def forward(self, x): return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

class TTTAdaptiveStage(nn.Module):
    def __init__(self, stage_module, in_channels, ttt_lr=0.005, noise_std=0.05):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.lightning_indexer = AugmentedLightningIndexer(in_channels)
        self.ttt_lr = ttt_lr
        self.noise_std = noise_std

    def _get_active_mask(self, feat, ratio=0.75):
        saliency_map = self.lightning_indexer(feat) 
        scores_blocked = F.avg_pool2d(saliency_map, 8, stride=8)
        B = feat.shape[0]
        threshold = torch.quantile(scores_blocked.view(B, -1), ratio, dim=1, keepdim=True)
        mask_blocked = (scores_blocked >= threshold.view(B, 1, 1, 1)).float()
        return F.interpolate(mask_blocked, size=(feat.shape[2:]), mode='nearest')

    import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad
import contextlib # <--- New Import for Windows/PyTorch compatibility

# ... (GRN, AugmentedLightningIndexer, TTTProjector classes stay the same) ...

class TTTAdaptiveStage(nn.Module):
    def __init__(self, stage_module, in_channels, ttt_lr=0.005, noise_std=0.05):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.lightning_indexer = AugmentedLightningIndexer(in_channels)
        self.ttt_lr = ttt_lr
        self.noise_std = noise_std

    def _get_active_mask(self, feat, ratio=0.75):
        B, C, H, W = feat.shape
        
        # 1. Lightning Indexer 
        # (Indexer weights are usually float16 in --fp16 mode)
        saliency_map = self.lightning_indexer(feat.detach()) 
        
        # 2. Block-wise Aggregation
        scores_blocked = F.avg_pool2d(saliency_map, 8, stride=8)
        
        # --- PRECISION BRIDGE START ---
        # 3. Top-K Selection 
        # Move to float32 for quantile, as float16 is not supported
        scores_flattened = scores_blocked.view(B, -1).float() 
        
        threshold = torch.quantile(scores_flattened, ratio, dim=1, keepdim=True)
        
        # Perform comparison in float32, then cast back to the original dtype 
        # ensures the mask matches the precision of the backbone features
        mask_blocked = (scores_blocked >= threshold.view(B, 1, 1, 1)).to(feat.dtype)
        
        return F.interpolate(mask_blocked, size=(H, W), mode='nearest')

    def inner_loss_fn(self, params, projector_params, x_in, mask, noise_map, clean_target):
        """
        AMP-Optimized Inner Loss.
        Uses autocast to speed up functional forward passes on RTX 30-series GPUs.
        """
        # Enable Mixed Precision inside the functional scope
        with torch.cuda.amp.autocast(enabled=True):
            # 1. Functional Forward through backbone
            feat = functional_call(self.backbone_stage, params, x_in)
            
            # 2. Corruption logic
            feat_noisy = feat + noise_map
            feat_corrupted = feat_noisy * (1 - mask) + self.projector.mask_token * mask
            
            # 3. Reconstruction through projector
            rec = functional_call(self.projector, projector_params, feat_corrupted)
            
            # 4. MSE calculation (autocast handles the scaling internally)
            return F.mse_loss(rec, clean_target)


class TTTAdaptiveStage(nn.Module):
    def __init__(self, stage_module, in_channels, ttt_lr=0.005, noise_std=0.05):
        super().__init__()
        self.backbone_stage = stage_module
        self.projector = TTTProjector(in_channels)
        self.lightning_indexer = AugmentedLightningIndexer(in_channels)
        self.ttt_lr = ttt_lr
        self.noise_std = noise_std

    def _get_active_mask(self, feat, ratio=0.75):
        # Universal DType Alignment
        indexer_params = dict(self.lightning_indexer.named_parameters())
        
        with torch.no_grad():
            # functional_call to force the indexer to match feat.dtype
            saliency_map = functional_call(self.lightning_indexer, indexer_params, feat.detach())
            
            # 8x8 Block Aggregation
            scores_blocked = F.avg_pool2d(saliency_map, 8, stride=8)
            
            # Precision Bridge for Quantile
            scores_fp32 = scores_blocked.view(feat.shape[0], -1).float()
            threshold = torch.quantile(scores_fp32, ratio, dim=1, keepdim=True)
            
            # Final Mask construction
            mask_blocked = (scores_blocked >= threshold.view(feat.shape[0], 1, 1, 1)).to(feat.dtype)
            
        return F.interpolate(mask_blocked, size=(feat.shape[2:]), mode='nearest')
    def inner_loss_fn(self, params, projector_params, x_in, mask, noise_map, clean_target):
        # We run the inner loop in the precision provided by the caller (Training or Eval)
        # 1. Functional Forward through backbone
        feat = functional_call(self.backbone_stage, params, x_in)
        # 2. Denoising Logic
        feat_noisy = feat + noise_map
        # 3. Masked Modeling Logic
        feat_corrupted = feat_noisy * (1 - mask) + self.projector.mask_token * mask
        # 4. Reconstruction
        rec = functional_call(self.projector, projector_params, feat_corrupted)
        return F.mse_loss(rec, clean_target)

    def forward(self, x_in, run_ttt=True):
        # 0. Profiler Bypass
        is_profiling = any(hasattr(m, 'total_ops') for m in self.backbone_stage.modules())
        if is_profiling or not run_ttt:
            return self.backbone_stage(x_in)

        # Capture input precision to ensure output matches
        input_dtype = x_in.dtype

        # 1. SETUP GRADIENT CONTEXT
        is_inference_mode = (not torch.is_grad_enabled())
        if is_inference_mode:
            x_curr = x_in.detach()
            x_curr.requires_grad = True
            context_manager = torch.enable_grad()
        else:
            x_curr = x_in
            context_manager = contextlib.nullcontext()

        # 2. CAPTURE & PARTITION STATE
        backbone_params = dict(self.backbone_stage.named_parameters())
        backbone_buffers = dict(self.backbone_stage.named_buffers())
        proj_params = dict(self.projector.named_parameters())
        proj_buffers = dict(self.projector.named_buffers())

        adapt_backbone = {k: v for k, v in backbone_params.items() 
                         if ('bn' in k or 'norm' in k) and v.is_floating_point()}
        static_backbone_state = {k: v for k, v in backbone_params.items() 
                                if k not in adapt_backbone}
        fixed_backbone_state = {**static_backbone_state, **backbone_buffers}

        # 3. FREEZE MODULE STATE 
        self.backbone_stage.eval()
        self.projector.eval()

        with context_manager:
            if not x_curr.is_floating_point():
                return self.backbone_stage(x_in)

            # Generate Target
            with torch.no_grad():
                feat_initial = self.backbone_stage(x_curr)
                clean_target = feat_initial.detach()
                mask = self._get_active_mask(clean_target)
                noise_map = torch.randn_like(clean_target) * self.noise_std

            # 4. THE CORE META-LEARNING STEP
            def inner_grad_fn(p_adapt_b, p_adapt_p):
                full_backbone = {**p_adapt_b, **fixed_backbone_state}
                # Projector buffers (fixed_proj_state) is empty but included for safety
                return self.inner_loss_fn(full_backbone, p_adapt_p, x_curr, mask, noise_map, clean_target)

            grads_backbone, grads_proj = grad(inner_grad_fn, argnums=(0, 1))(adapt_backbone, proj_params)
            
            # 5. CONSTRUCT ADAPTED WEIGHT SPACE
            updated_backbone_params = {**backbone_params, **backbone_buffers}
            for name, g in grads_backbone.items():
                updated_backbone_params[name] = updated_backbone_params[name] - self.ttt_lr * g

        # 6. RESTORE ORIGINAL STATE
        if self.training:
            self.backbone_stage.train()
            self.projector.train()

        # 7. FINAL ADAPTED FORWARD 
        if self.training:
            out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)
        else:
            with torch.no_grad():
                out = functional_call(self.backbone_stage, updated_backbone_params, x_curr)

        # Force cast to input_dtype to be safe for deeper layers
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
    Differentiable Associative Memory Bank.
    Stores 'Canonical Prototypes' of object classes.
    """
    def __init__(self, num_classes, latent_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Memory Bank: Prototypes for each class
        # Initialized with orthogonal vectors to maximize identity separation
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x_latent, uncertainty_gate, objectness_mask):
        """
        x_latent: [B, HW, latent_dim] - Observed features
        uncertainty_gate: [B, HW, 1] - beta value based on entropy
        objectness_mask: [B, HW, 1] - Gating based on physical presence
        """
        # 1. Similarity Scoring (Soft-Attention Lookup)
        # compare observed latent vectors to all canonical prototypes
        # [B, HW, latent_dim] @ [latent_dim, num_classes] -> [B, HW, num_classes]
        attn_scores = torch.matmul(x_latent, self.prototypes.t())
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 2. Retrieval: Weighted sum of prototypes
        # [B, HW, num_classes] @ [num_classes, latent_dim] -> [B, HW, latent_dim]
        memory_retrieved = torch.matmul(attn_weights, self.prototypes)
        
        # 3. Closed-Loop Restoration
        # Memory is only inlayed if:
        # a) Uncertainty is high (gate)
        # b) An object is physically present (objectness)
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