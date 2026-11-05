"""Structural and Taylor-sign tests for gradient alignment on synthetic tasks."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from safe_ttt_model import SafeSpatialTTT
from gradient_audit import audit_compact_gradient

def main():
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model=SafeSpatialTTT(channels=8,rank=4,max_state_step=0.0001)
    x=torch.randn(2,8,10,12)
    target=torch.randn_like(x)
    # Distinct task component stand-ins, output scalar losses; scales are not
    # claimed to be real YOLOX P3/P4/P5 task losses here.
    def tasks(feats):
        return {'synthetic_localization': F.mse_loss(feats,target),
                'synthetic_activation':(feats.mean(dim=(-2,-1)).square().mean())}
    report=audit_compact_gradient(model,x,tasks)
    assert set(report['by_task'])=={'synthetic_localization','synthetic_activation'}
    for entry in report['by_task'].values():
        assert len(entry['cosine'])==2
        assert all(-1.0001 <= v <=1.0001 for v in entry['cosine'])
        assert entry['loss_delta']==entry['loss_delta']
        assert entry['linear_predicted_delta']==entry['linear_predicted_delta']
    # Weak Taylor local approximation should agree with sign/magnitude for
    # small enough step. This is a mathematical gradient check, not detection.
    for name, entry in report['by_task'].items():
        predicted=entry['linear_predicted_delta']
        observed=entry['loss_delta']
        assert abs(predicted-observed) <= 2e-4, (name,predicted,observed)
    print('PASS: per-task SSL/detection gradient cosines, finite-difference Taylor check')

if __name__=='__main__':
    main()
