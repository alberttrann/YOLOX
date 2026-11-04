from pathlib import Path
import importlib.util
import sys
import types
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ann=load('test_annotation_survival',ROOT/'tools/audit_pgsma_annotation_survival.py')
inst=load('test_installer',ROOT/'install_fidelity_audit.py')


def test_match_preserved_boxes():
    before=np.array([[10,20,30,50,2],[0,0,3,3,1]],dtype=float)
    # Scale 2x into 100x100, target output class/cxcywh
    after=np.array([[2,40,70,40,60],[1,3,3,6,6],[0,0,0,0,0]],dtype=float)
    result=ann.box_matches_pre_to_post(before,after,(100,100),(50,50))
    assert result['matched_count']==2
    assert result['unmatched_pre_indices']==[]
    assert result['post_valid_count']==2


def test_match_detects_tiny_target_removed():
    before=np.array([[10,20,30,50,2],[0,0,1,1,1]],dtype=float)
    after=np.array([[2,40,70,40,60],[0,0,0,0,0]],dtype=float)
    result=ann.box_matches_pre_to_post(before,after,(100,100),(50,50))
    assert result['matched_count']==1
    assert result['unmatched_pre_indices']==[1]
    assert ann.size_bin(1,5)=='<4px'


def test_no_match_wrong_class():
    before=np.array([[10,20,30,50,2]])
    after=np.array([[3,40,70,40,60]])
    assert ann.box_matches_pre_to_post(before,after,(100,100),(50,50))['matched_count']==0


def test_manifest_guard_preserves_unknown_modification(tmp_path):
    repo=tmp_path/'repo';(repo/'tools').mkdir(parents=True)
    (repo/'tools/audit_pgsma_real_amp.py').write_text('user local edits')
    res=inst.run(repo,apply=True)
    assert res['status']=='BLOCKED'
    assert (repo/'tools/audit_pgsma_real_amp.py').read_text()=='user local edits'
    assert not (repo/'tools/audit_pgsma_padding_mask.py').exists()


def test_installer_check_apply_idempotent(tmp_path):
    repo=tmp_path/'repo'
    current=repo/'tools/audit_pgsma_real_amp.py'
    current.parent.mkdir(parents=True)
    # Use the exact expected prior script from our archived data patch.
    previous=ROOT/'test_support/prior_amp_audit.py'
    if not previous.is_file():pytest.skip('Archived prior script not available')
    current.write_bytes(previous.read_bytes())
    assert inst.run(repo,apply=False)['status']=='READY'
    r=inst.run(repo,apply=True)
    assert r['status']=='INSTALLED'
    assert inst.run(repo,apply=False)['status']=='READY'
    assert all(x['action']=='already_installed' for x in inst.run(repo,apply=False)['files'])


def test_geometry_stress_uses_nontrivial_masks(monkeypatch):
    # Mock only YOLOX TrainTransform dependency; use actual validity geometry.
    yolox=types.ModuleType('yolox');yolox.__path__=[]
    data=types.ModuleType('yolox.data');data.__path__=[]
    models=types.ModuleType('yolox.models');models.__path__=[]
    class FakeTrainTransform:
        def __init__(self,max_labels,flip_prob,hsv_prob):self.max_labels=max_labels
        def __call__(self,image,targets,input_dim):
            h,w=input_dim
            return np.zeros((3,h,w),dtype=np.float32),np.zeros((self.max_labels,5),dtype=np.float32)
    data.TrainTransform=FakeTrainTransform
    for name,mod in [('yolox',yolox),('yolox.data',data),('yolox.models',models)]:monkeypatch.setitem(sys.modules,name,mod)
    trans=load('yolox.data.pgsma_train_transform',ROOT/'test_support/pgsma_train_transform.py')
    validity=load('yolox.models.pgsma_validity',ROOT/'test_support/pgsma_validity.py')
    monkeypatch.setitem(sys.modules,'yolox.data.pgsma_train_transform',trans)
    monkeypatch.setitem(sys.modules,'yolox.models.pgsma_validity',validity)
    pad=load('pad_audit',ROOT/'tools/audit_pgsma_padding_mask.py')
    result=pad.audit()
    assert result['status']=='CHECKS_PASSED'
    fractions={x['case']:x['valid_pixel_fraction'] for x in result['cases']}
    assert fractions['reference_16_9']==1.0
    assert fractions['square']<1.0
    assert fractions['portrait']<1.0
