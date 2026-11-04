from pathlib import Path
import importlib.util
import sys
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'install_bounded_pilot.py'
s=importlib.util.spec_from_file_location('pilot_installer',SCRIPT)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)

@pytest.fixture
def repo(tmp_path):
    for path in ('exps/default/tde_yolox_s_pgsma.py','yolox/models/pgsma.py',
                 'yolox/models/pgsma_head.py','yolox/core/pgsma_step.py','yolox/core/trainer.py'):
        target=tmp_path/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('allow_full_training = False' if path.startswith('exps') else
            'for group in self.optimizer.param_groups:\n clip_grad_norm_()' if path.endswith('trainer.py') else 'fake',encoding='utf8')
    (tmp_path/'tools').mkdir()
    return tmp_path

def test_inspect_and_apply_idempotent(repo):
    a=m.run(repo,False)
    assert set(a['actions'].values())=={'ADD'}
    a=m.run(repo,True)
    assert a['status']=='APPLIED'
    assert set(m.run(repo,False)['actions'].values())=={'ALREADY_INSTALLED'}

def test_does_not_overwrite_existing(repo):
    (repo/'tools'/'pgsma_bounded_pilot.py').write_text('user code',encoding='utf8')
    with pytest.raises(RuntimeError):m.run(repo,True)

def test_enforces_full_training_guard(repo):
    p=repo/'exps/default/tde_yolox_s_pgsma.py'
    p.write_text('allow_full_training = True')
    with pytest.raises(RuntimeError):m.run(repo,True)
