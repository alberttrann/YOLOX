from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import shutil
import hashlib
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('clip_installer',ROOT/'apply_clip_safety.py')
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_patch_guard_applies_and_is_idempotent():
    # self-contained without /tmp: reconstruct baseline via reverse patch changes
    modified=(ROOT/'trainer_patched.py').read_text(encoding='utf-8')
    replacements=[
        ("""        # Semantic gradients belong to the semantic optimizer, whose step already
        # completed above. Clear them before the native backward and telemetry.
        # Native gradient clipping must NEVER include semantic parameters.
        self.semantic_optimizer.zero_grad(set_to_none=True)

""",''),
        ("""        # Parameter ownership is defined by the native optimizer's param groups,
        # not by model.parameters(): PG-SMA parameters are in the same module.
        native_params = [p for group in self.optimizer.param_groups
                         for p in group['params'] if p.requires_grad]
        native_ids = {id(p) for p in native_params}
        named_grads = [(n, p.grad) for n, p in self.model.named_parameters()
                       if id(p) in native_ids and p.grad is not None]
        invalid_names = [n for n, g in named_grads if not torch.isfinite(g).all().item()]
""",
         """        named_grads = [(n, p.grad) for n, p in self.model.named_parameters() if p.grad is not None]
        invalid_names = [n for n, g in named_grads if not torch.isfinite(g).all().item()]
"""),
        ("""                native_params, max_norm=10.0, error_if_nonfinite=True""",
         """                self.model.parameters(), max_norm=10.0, error_if_nonfinite=True"""),
    ]
    for a,b in replacements:
        assert modified.count(a)==1
        modified=modified.replace(a,b)
    assert hashlib.sha256(modified.encode('utf-8')).hexdigest()==installer.MANIFEST['baseline_sha256']
    with tempfile.TemporaryDirectory() as tmp:
        trainer=Path(tmp)/installer.MANIFEST['path']
        trainer.parent.mkdir(parents=True)
        trainer.write_text(modified,encoding='utf-8')
        assert installer.apply(Path(tmp),False)['status']=='READY_TO_APPLY'
        assert installer.apply(Path(tmp),True)['status']=='PATCHED'
        assert installer.apply(Path(tmp),False)['status']=='ALREADY_PATCHED'
        trainer.write_text('local changed file',encoding='utf-8')
        import pytest
        with pytest.raises(RuntimeError, match='Unrecognized trainer'):
            installer.apply(Path(tmp),True)


def test_native_clip_excludes_semantic_grads():
    n=torch.nn.Parameter(torch.tensor([1.,2.]))
    s=torch.nn.Parameter(torch.tensor([1.,2.]))
    optim=torch.optim.SGD([n],lr=.1)
    n.grad=torch.tensor([12.,16.])
    s.grad=torch.tensor([600000.,800000.])
    native_params=[p for group in optim.param_groups for p in group['params']]
    original=torch.nn.utils.clip_grad_norm_(native_params,10.0,error_if_nonfinite=True)
    assert float(original)==20.0
    assert torch.allclose(n.grad,torch.tensor([6.,8.]),atol=1e-5)
    assert s.grad.abs().max().item()==800000


def test_reject_semantic_nan_from_native_scope():
    n=torch.nn.Parameter(torch.ones(2))
    s=torch.nn.Parameter(torch.ones(2))
    opt=torch.optim.SGD([n],lr=.01)
    n.grad=torch.ones(2)
    s.grad=torch.full((2,),float('nan'))
    result=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],10.,error_if_nonfinite=True)
    assert bool(torch.isfinite(result))
