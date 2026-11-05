"""Small stdlib/PyTorch tests. Requires the audit Python file installed under tools/."""
import importlib.util
from pathlib import Path
import math
import torch

path=Path(__file__).resolve().parents[1]/'tools'/'audit_ttt_training_attribution.py'
# Loading this module requires the already-installed prior YOLOX audit tools.
# Copying over all 5 dependencies from the previous packaged bundle isn't needed
# for host tests, but a synthetic module import path is provided in CI.

def _load_module():
    import sys
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('attribution',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def run_tests(m):
    keys=('adapter.conv.weight','predictor.conv.weight','mask_token','step_logit')
    i={k:torch.zeros(2,2) for k in keys}
    t={k:torch.full((2,2),j+1.) for j,k in enumerate(keys)}
    changes=m.check_states(i,t)
    assert all(changes[g]['changed_tensors']==1 for g in m.GROUPS)
    assert len(m.subsets())==16
    for combo in m.subsets():
        h=m.combine(i,t,combo)
        for k in keys:
            assert torch.equal(h[k],t[k] if m.family(k) in combo else i[k])
    d={frozenset(S):sum((m.GROUPS.index(s)+1) for s in S) for S in m.subsets()}
    phi=m.shapley(d)
    assert all(abs(phi[g]-(j+1))<1e-12 for j,g in enumerate(m.GROUPS))
    d2={frozenset(S): (10 if set(m.GROUPS[:2])<=set(S) else 0)+len(S) for S in m.subsets()}
    phi2=m.shapley(d2)
    assert abs(sum(phi2.values())-(d2[frozenset(m.GROUPS)]-d2[frozenset()]))<1e-12
    assert abs(phi2['adapter']-6.0)<1e-12 and abs(phi2['predictor']-6.0)<1e-12
    try:
        m.family('new_state.unknown')
    except ValueError:pass
    else:raise AssertionError('Unknown source state accepted')
    try:
        m.combine(i,dict(t,unlisted=torch.zeros(1)),[])
    except RuntimeError:pass
    else:raise AssertionError('Mismatched state accepted')
    print('PASS: 16 strict hybrids, group-state changes, Shapley interactions, key guards')

if __name__=='__main__':
    run_tests(_load_module())
