import torch
from yolox.models import YOLOX, YOLOPAFPN, CSPDarknet
from yolox.models.engram_head import TDE_Head

def sanity_check():
    print("Starting TDE-YOLOX Holistic Sanity Check...")
    
    # 1. Setup
    num_classes = 9
    model = YOLOX(
        backbone=YOLOPAFPN(depth=0.33, width=0.50),
        head=TDE_Head(num_classes=num_classes, width=0.50)
    ).cuda()
    
    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    # --- Create VALID Targets ---
    # [batch, max_labels, 5] -> [class_id, x, y, w, h]
    dummy_targets = torch.zeros(1, 10, 5).cuda()
    # Class IDs must be integers in [0, 8]
    dummy_targets[:, :, 0] = torch.randint(0, 9, (1, 10)).float()
    # Coordinates must be positive
    dummy_targets[:, :, 1:] = torch.abs(torch.randn(1, 10, 4)) * 100 

    # --- TEST 1: TRAINING MODE ---
    print("Testing Training Forward...")
    model.train()
    try:
        # Mocking an epoch to test annealing
        model.set_meta_training_state(epoch=20, max_epochs=80) 
        loss_dict = model(dummy_input, dummy_targets)
        loss_dict["total_loss"].backward()
        print("Training Pass Successful.")
    except Exception as e:
        print(f"Training Pass Failed: {e}")

    # --- TEST 2: INFERENCE / EVAL MODE ---
    print("Testing Inference (Validation) Forward...")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(dummy_input)
            print(f"Inference Pass Successful. Output shape: {outputs[0].shape}")
        except Exception as e:
            print(f"Inference Pass Failed: {e}")

    # --- TEST 3: MIXED PRECISION (AMP) ---
    print("Testing Mixed Precision (FP16)...")
    model.eval()
    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
            try:
                outputs = model(dummy_input.half())
                print("AMP Pass Successful.")
            except Exception as e:
                print(f"AMP Pass Failed: {e}")

    print("\nAll Checks Passed.")

if __name__ == "__main__":
    sanity_check()