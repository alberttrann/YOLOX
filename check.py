import torch
from torch.func import grad
print(f"CUDA Available: {torch.cuda.is_available()}")
x = torch.randn(1, requires_grad=True)
loss = (x**2).sum()
print(f"Meta-grad test: {grad(lambda x: (x**2).sum())(x)}")