import importlib
import os
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import pgsma_bounded_pilot as pilot
from tools import compare_pgsma_pilots as comp
from tools._pgsma_pilot_core import atomic_json


def test_rng_capture_restore_cpu():
    import numpy as np
    import random
    pilot.seed_everything(111)
    state=pilot.capture_rng()
    first=(random.random(),float(np.random.rand()),float(torch.rand(())))
    pilot.restore_rng(state)
    second=(random.random(),float(np.random.rand()),float(torch.rand(())))
    assert first==second


def make_report(path,seed=5,variant='pgsma',stream='same'):
    path.mkdir(parents=True,exist_ok=True)
    m={'status':'CHECKS_PASSED','variant':variant,'seed':seed,
       'eval_indices':[4,10,20],'input_size':[576,1024],
       'clean_val_json_sha256':'gt_source','steps_requested':20,
       'checkpoint':{'sha256':'warm_hash'},
       'first_training_batch_sha256':'first','training_batch_stream_sha256':stream,
       'native_state_sha256':'native_match','successful_native_updates':20,
       'successful_semantic_updates':0,'last_telemetry':{}}
    stats={'AP50_95':0.2,'AP50':0.3,'AP75':0.1,'AR100':0.25}
    per={k:{'recall_iou50':0.4,'total_gt':10,'matched':4} for k in ('lt4','4to8','8to16','16to32','ge32')}
    records={'native':{'coco_metrics':stats,
                       'post_nms_recall_iou50_by_effective_min_side':per}}
    if variant!='native_scpa':
        records['pgsma' if variant=='pgsma' else 'memory']={
          'coco_metrics':dict(stats),'post_nms_recall_iou50_by_effective_min_side':per}
    atomic_json(path/'result.json',m)
    atomic_json(path/'clean_val_step_00000.json',{'metrics':records})
    atomic_json(path/'clean_val_step_00020.json',{'metrics':records})


def test_report_matching(tmp_path):
    folders=[tmp_path/v for v in ('native_scpa','memory','pgsma')]
    for folder in folders:
        make_report(folder,variant=folder.name)
    report=comp.compare(folders)
    assert report['status']=='CHECKS_PASSED'
    assert report['full_80_epoch_permission'] is False
    assert len(report['runs'])==3


def test_report_rejects_different_validation_seed(tmp_path):
    a,b=tmp_path/'x',tmp_path/'y'
    make_report(a,seed=1);make_report(b,seed=2)
    with pytest.raises(RuntimeError):comp.compare([a,b])


def test_report_warns_different_batch_stream(tmp_path):
    a,b=tmp_path/'x',tmp_path/'y'
    make_report(a,stream='x');make_report(b,stream='y')
    report=comp.compare([a,b])
    assert report['status']=='QUALIFICATION_WARNINGS'
    assert any('Data streams' in w for w in report['warnings'])
