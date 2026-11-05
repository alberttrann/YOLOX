"""Standalone CPU tests for source-only finite-step report math. No YOLOX needed.

To run inside YOLOX repo, copy this test to tests/ and run:
    python tests/test_safe_ttt_step_sweep.py
"""
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / 'tools'
# CLI imports YOLOX-dependent helpers; mock only those names for isolated tests.
for name in ('audit_safe_ttt_yolox','fixed_detection_targets','native_dark2_router',
             'safe_ttt_model','pilot_safe_ttt_meta'):
    mod = types.ModuleType(name)
    if name=='audit_safe_ttt_yolox':
        for v in ('setup_model','get_size_from_native','head_loss_pass','record_model_hash'):
            setattr(mod,v,lambda *args,**kw:None)
    if name=='fixed_detection_targets':
        for v in ('NativeMatchReplay','reconstruct_tde_loss','assert_head_loss_parity'):
            setattr(mod,v,lambda *args,**kw:None)
    if name=='native_dark2_router':
        setattr(mod,'Dark2AuditRouter',type('MockRouter',(),{}))
    if name=='safe_ttt_model':
        setattr(mod,'SafeSpatialTTT',type('MockSafe',(),{}))
    if name=='pilot_safe_ttt_meta':
        setattr(mod,'load_source_example',lambda *args,**kw:None)
    sys.modules[name]=mod
spec=importlib.util.spec_from_file_location('step_sweep',TOOLS/'audit_safe_ttt_step_sweep.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def example(total=1.,small=0.1,medium=0.2,large=0.3,background=0.4):
    return {'total':total,'components':{'box':small,'classification':medium,
        'positive_objectness':large,'background_objectness':background},
        'scales':{'small':{'box':small,'classification':0.,'positive_objectness':0.},
                  'medium':{'box':medium,'classification':0.,'positive_objectness':0.},
                  'large':{'box':large,'classification':0.,'positive_objectness':0.}},
        'levels':{},'counts':{'small':1,'medium':1,'large':1,'foreground':3,
                             'background':100}}


def test_improvement_cannot_hide_scale_regression():
    a=example()
    b=example(total=.95,small=.12,medium=.18,large=.25,background=.4)
    g=audit.evaluate_acceptance(a,b,1e-6,1e-6)
    assert g['meaningfully_beneficial']
    assert not g['scale_nonregressing']
    assert not g['strict_surrogate_pass']


def test_true_improvement_and_neutral_tolerance():
    a=example()
    b=example(total=.90,small=.09,medium=.18,large=.25,background=.38)
    g=audit.evaluate_acceptance(a,b,1e-6,1e-6)
    assert g['strict_surrogate_pass']
    c=example(total=.999999,small=.1,medium=.2,large=.3,background=.4)
    g=audit.evaluate_acceptance(a,c,1e-5,1e-5)
    assert not g['meaningfully_beneficial']


def test_empty_scale_excluded():
    a=example(); a['counts']['small']=0
    b=example(total=.90,small=.9,medium=.18,large=.25,background=.38)
    g=audit.evaluate_acceptance(a,b,1e-6,1e-6)
    assert 'small' not in g['present_scale_groups']


def test_summary_oracle_only_and_not_deployed():
    a=example()
    xs=[]
    for i in (10,37):
        row={'index':i,'feature_effects':{'0.0':0.,'1.0':.01},
             'candidates':{'0.0':audit.evaluate_acceptance(a,a,1e-6,1e-6),
                           '1.0':audit.evaluate_acceptance(a,example(total=.9,small=.09,medium=.18,large=.25,background=.38) if i==10 else example(total=1.1,small=.11,medium=.21,large=.31,background=.47),1e-6,1e-6)}}
        xs.append(row)
    summary=audit.summarize(xs,[0.,1.])
    assert summary['1.0']['n_beneficial']==1
    assert summary['1.0']['n_harmful']==1
    assert summary['oracle_diagnostic_only']['n_images_with_any_beneficial_nonzero_step']==1
    assert summary['oracle_diagnostic_only']['n_images_with_any_strict_surrogate_step']==1

if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn();print('PASS',name)
