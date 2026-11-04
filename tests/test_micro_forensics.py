from pathlib import Path
import importlib.util
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'tests'/'support'/'audit_survival_snapshot.py'
# Standalone import without YOLOX data: existing audit code only imports YOLOX in run().
spec=importlib.util.spec_from_file_location('tools.audit_pgsma_annotation_survival',PREVIOUS)
prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
tools=types.ModuleType('tools');tools.__path__=[]
sys.modules['tools']=tools
sys.modules['tools.audit_pgsma_annotation_survival']=prev
spec2=importlib.util.spec_from_file_location('tools.audit_pgsma_micro_forensics',ROOT/'tools/audit_pgsma_micro_forensics.py')
mod=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(mod)


def fixture_trace(pre,post,input_size=(100,100),original_shape=(100,100)):
    b=np.asarray(pre,dtype=float).reshape(-1,5)
    a=np.asarray(post,dtype=float).reshape(-1,5)
    matching=prev.box_matches_pre_to_post(b,a,input_size,original_shape)
    matching['before_boxes']=b.tolist()
    matching['after_labels']=a.tolist()
    return matching


def test_one_pixel_filter_drop():
    tr=fixture_trace([[1,1,1.8,3,0],[10,10,20,20,1]],[[1,15,15,10,10]])
    s=mod.classify_trace(tr)
    assert s['valid_pre_boxes']==2
    assert s['post_valid_boxes']==1
    assert s['predicted_filter_by_min_side']==1
    assert s['unmatched_explained_by_min_side_filter']==1
    assert s['unmatched_not_explained_by_min_side_filter']==0
    assert s['post_count_agrees_with_filter_prediction']


def test_unexplained_drop_detected():
    tr=fixture_trace([[1,1,10,10,0]],[])
    s=mod.classify_trace(tr)
    assert s['unmatched_not_explained_by_min_side_filter']==1
    assert not s['post_count_agrees_with_filter_prediction']


def test_an_invalid_crop_box_not_counted_as_drop():
    tr=fixture_trace([[10,10,10,14,0],[20,20,30,30,2]],[[2,25,25,10,10]])
    s=mod.classify_trace(tr)
    assert s['invalid_pre_boxes']==1
    assert s['valid_pre_boxes']==1
    assert s['unmatched_valid_boxes']==0


def test_scale_sensitive_side_length():
    tr=fixture_trace([[1,1,2.4,5,0]],[],input_size=(50,50),original_shape=(100,100))
    s=mod.classify_trace(tr)
    assert s['predicted_filter_by_min_side']==1
    assert s['missing'][0]['min_side_px'] == 0.7


def test_filter_no_loss():
    tr=fixture_trace([[0,0,2,3,0]],[[0,1,1.5,2,3]])
    s=mod.classify_trace(tr)
    assert s['post_count_agrees_with_filter_prediction']
    assert s['unmatched_valid_boxes']==0
