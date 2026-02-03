import torch
from torch.func import grad
print(f"CUDA Available: {torch.cuda.is_available()}")
# Check if functional gradients work (Phase 1 logic)
x = torch.randn(1, requires_grad=True)
loss = (x**2).sum()
print(f"Meta-grad test: {grad(lambda x: (x**2).sum())(x)}")