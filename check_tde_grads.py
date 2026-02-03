import torch
from yolox.exp import get_exp

def check_meta_gradients():
    print("Initializing TDE-YOLOX...")
    # Load exp
    exp = get_exp("exps/default/tde_yolox_s.py", None)
    model = exp.get_model()
    model.train()
    
    # Mock Input
    x = torch.randn(2, 3, 640, 640) # Batch size 2
    # Mock Targets (YOLOX expects list of tensors)
    targets = torch.zeros(2, 50, 5) # [batch, max_objs, 5]
    
    print("Running Forward Pass with Meta-TTT...")
    try:
        outputs = model(x, targets)
        loss = outputs["total_loss"]
        print(f"Forward Pass Successful. Loss: {loss.item()}")
    except Exception as e:
        print(f"Forward Failed: {e}")
        return

    print("Running Backward (Meta-Update)...")
    loss.backward()
    
    # CHECK 1: Did gradients reach the TTT Projector?
    # confirms Inner Loop is connected to Outer Loop
    projector_grad = model.backbone.backbone.dark2.projector.net[0].weight.grad
    if projector_grad is not None:
        print("SUCCESS: Gradients reached TTT Projector (Meta-Learning Active)")
        print(f"   Grad Norm: {projector_grad.norm().item()}")
    else:
        print("FAILURE: TTT Projector has no gradients. Graph broken.")

    # CHECK 2: Did gradients reach the Engram Memory?
    engram_grad = model.head.memory_banks[0].prototypes.grad
    if engram_grad is not None:
         print("SUCCESS: Gradients reached Engram Memory.")
    else:
         print("FAILURE: Engram Memory has no gradients.")

if __name__ == "__main__":
    check_meta_gradients()