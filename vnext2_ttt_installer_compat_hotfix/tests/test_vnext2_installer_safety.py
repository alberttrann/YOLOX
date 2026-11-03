"""Portable source-level sanity check after installing vNext.2 per-image TTT.
Run from YOLOX repository root: python tests/test_vnext2_installer_safety.py
"""
import ast
from pathlib import Path
src = Path('yolox/models/ttt_modules.py').read_text(encoding='utf-8')
t = ast.parse(src)
cls = [n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='TTTAdaptiveStage']
assert len(cls)==1
fn = [n for n in cls[0].body if isinstance(n,ast.FunctionDef) and n.name=='_forward_residual_ttt']
helper = [n for n in cls[0].body if isinstance(n,ast.FunctionDef) and n.name=='_forward_residual_ttt_per_image_eval']
assert len(fn)==1 and len(helper)==1, 'Missing/duplicate TTT method'
assert sum('_forward_residual_ttt_per_image_eval(base_feat)' in ast.unparse(n) for n in fn[0].body)==1, 'Missing/duplicate dispatch'
assert 'if not run_ttt' in ast.unparse(fn[0])
assert 'if training_meta:' in ast.unparse(fn[0]), 'Expected frozen-EMA training path absent'
assert 'self.current_rec_loss' in ast.unparse(fn[0]), 'Training reconstruction loss absent'
assert 'torch.autograd.grad' in ast.unparse(helper[0]), 'No inner autograd in new helper'
print('PASS: one helper + one eval-only dispatch; original meta-training path retained')
