import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad

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

    def forward(self, x_in, run_ttt=True):
        if not run_ttt:
            return self.backbone_stage(x_in)

        # 1. Setup Param Groups
        # manually separate params to use functional_call for the BACKBONE only
        all_params = dict(self.backbone_stage.named_parameters())
        
        # Identify which params will be updated (Norms)
        adapt_names = [n for n, p in self.backbone_stage.named_parameters() 
                       if ('bn' in n or 'norm' in n) and p.requires_grad]
        
        # 2. Meta-Forward (Building the Graph)
        # Standard forward to get features (graph connected to x_in and backbone weights)
        feat = self.backbone_stage(x_in)
        
        # Generate Corruption
        mask = self._get_active_mask(feat)
        noise = torch.randn_like(feat) * self.noise_std
        feat_corrupted = (feat + noise) * (1 - mask) + self.projector.mask_token * mask
        
        # Reconstruction (Standard call keeps Projector in the graph)
        rec = self.projector(feat_corrupted)
        
        # Inner Loss
        # Target must be detached to prevent "collapse to zero" solution
        inner_loss = F.mse_loss(rec, feat.detach())

        # 3. Meta-Gradient Calculation
        # need gradients of inner_loss w.r.t. the Norm parameters of self.backbone_stage
        adapt_tensors = [all_params[n] for n in adapt_names]
        
        grads = torch.autograd.grad(
            inner_loss, 
            adapt_tensors, 
            create_graph=True, 
            retain_graph=True
        )
        
        # 4. Functional Update
        # Create the "fast weights" dictionary
        fast_params = {}
        grad_idx = 0
        for name, p in all_params.items():
            if name in adapt_names:
                # The meta-learning update step
                fast_params[name] = p - self.ttt_lr * grads[grad_idx]
                grad_idx += 1
            else:
                fast_params[name] = p

        # 5. Final Forward with Fast Weights
        # functional_call applies the fast_params to the backbone structure
        out = functional_call(self.backbone_stage, fast_params, x_in)

        return out

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