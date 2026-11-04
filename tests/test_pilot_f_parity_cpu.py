"""CPU contract checks for parity forensic utilities. Requires torch and pytest."""
import importlib.util
import json
from pathlib import Path
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('parity_compare',ROOT/'tools'/'compare_pgsma_native_parity.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_diff_location(tmp_path):
    a=tmp_path/'a.jsonl';b=tmp_path/'b.jsonl'
    rows=[{'step':0,'phase':'post_initial_validation','model':{'all':'x'}},
          {'step':1,'phase':'native_update','native_pre_step':{'all':'x'},
           'native_post_step':{'all':'y'},'native_loss':1.0,'native_grad_after_clip':'g'}]
    a.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rows[1]['native_grad_after_clip']='h'
    b.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=mod.compare(a,b)
    assert result['first_divergence']['step']==1
    assert 'native_grad_after_clip' in result['first_divergence']['changed']
    assert not result['full_training_authorized']


def test_identical_trace(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    row='{"step":0,"phase":"post_initial_validation","model":{"all":"x"}}\n'
    a.write_text(row);b.write_text(row)
    assert mod.compare(a,b)['status']=='PARITY_MATCH'


def test_duplicate_rejected(tmp_path):
    a=tmp_path/'a';a.write_text('{"step":0}\n{"step":0}\n')
    with pytest.raises(ValueError,match='Duplicate'):
        mod.read_trace(a)
