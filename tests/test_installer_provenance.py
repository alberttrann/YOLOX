"""Reproducibility/installation tests; no model files are changed in user tree."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import types

ROOT=Path(__file__).resolve().parents[1]

def load_install():
    spec=importlib.util.spec_from_file_location('install_a41_test',ROOT/'install_a41.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def load_provenance():
    for name in ('yolox','yolox.core'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    ft=types.ModuleType('yolox.core.fourier_trainer');ft.FourierTrainer=type('FourierTrainer',(object,),{})
    sys.modules[ft.__name__]=ft
    sched=types.ModuleType('yolox.core.a4_schedule');sched.before_epoch_with_native_finish=lambda self:None
    sys.modules[sched.__name__]=sched
    spec=importlib.util.spec_from_file_location('a41_provenance_test',ROOT/'overlay/yolox/core/a41_trainer.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_installer():
    mod=load_install()
    with tempfile.TemporaryDirectory() as td:
        repo=Path(td)
        for rel in mod.REQUIRED:
            p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original A4 placeholder',encoding='utf-8')
        original={rel:(repo/rel).read_bytes() for rel in mod.REQUIRED}
        ops=mod.install(repo,apply=False)
        assert not (repo/'yolox/models/a41_head.py').exists()
        assert any(action=='create' for action,*_ in ops)
        mod.install(repo,apply=True)
        mod.install(repo,apply=True)
        for rel,contents in original.items():assert (repo/rel).read_bytes()==contents
        dest=repo/'yolox/models/a41_head.py'
        dest.write_text('modified by user',encoding='utf-8')
        try:mod.install(repo,apply=True)
        except RuntimeError:pass
        else:raise AssertionError('Installer overwrote unexpected local changes')
        assert dest.read_text()=='modified by user'
        print('PASS additive installation, dry-run, idempotence, no A4 edits, conflict refusal')


def test_transitive_hashes():
    mod=load_provenance()
    with tempfile.TemporaryDirectory() as td:
        repo=Path(td)
        for rel in mod.A41_SOURCE_FILES:
            p=repo/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source '+rel)
        hash_a=mod.a41_source_hashes(repo)
        target=repo/mod.A41_SOURCE_FILES[0];target.write_text('modified')
        hash_b=mod.a41_source_hashes(repo)
        assert hash_a!=hash_b
        before={'schema':'a','source_sha256':hash_a,'basis':'dct','batch_size':8}
        after={'schema':'a','source_sha256':hash_b,'basis':'dct','batch_size':8}
        try:mod.verify_a41_provenance(before,after)
        except RuntimeError:pass
        else:raise AssertionError('Provenance drift not refused')
        try:(repo/mod.A41_SOURCE_FILES[-1]).unlink();mod.a41_source_hashes(repo)
        except RuntimeError:pass
        else:raise AssertionError('Missing code hash silently accepted')
        print('PASS all-source hashing, changed-source refusal, missing-source fail-closed')


if __name__=='__main__':
    test_installer();test_transitive_hashes()
