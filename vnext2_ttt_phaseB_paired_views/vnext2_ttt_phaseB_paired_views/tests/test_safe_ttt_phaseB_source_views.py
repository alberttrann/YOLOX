"""Portable Phase B tests. Requires installed previous safe_ttt_* tools modules."""
from __future__ import annotations
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from ttt_source_views import sample_appearance,apply_appearance,image_fingerprint
from pilot_safe_ttt_paired_views import (meta_objective,reference_consistency,
    summarize_views,_atomic_save_sidecar)
from summarize_safe_ttt_phaseB import summarize_report

torch.set_num_threads(1)


def test_pair_determinism_geometry_and_seed_isolation():
    torch.manual_seed(7)
    x=(torch.rand(1,3,24,32)*255).float()
    rng_state=torch.get_rng_state().clone()
    spec=sample_appearance(20261104,300,strength=.85)
    assert spec==sample_appearance(20261104,300,strength=.85)
    a=apply_appearance(x,spec)
    b=apply_appearance(x,spec)
    assert torch.equal(a,b)
    assert a.shape==x.shape and a.dtype==x.dtype
    assert torch.equal(torch.get_rng_state(),rng_state)
    assert image_fingerprint(a)==image_fingerprint(b)
    assert not torch.equal(x,a)
    spec2=sample_appearance(20261104,301,strength=.85)
    assert spec2!=spec
    assert not torch.equal(apply_appearance(x,spec2),a)
    assert bool(((a>=0)&(a<=255)).all())
    padded=x.clone()
    padded[:,:,-4:,:]=114.0
    padded[:,:,:,-5:]=114.0
    shifted_pad=apply_appearance(padded,spec)
    assert torch.equal(shifted_pad[:,:,-4:,:],padded[:,:,-4:,:])
    assert torch.equal(shifted_pad[:,:,:,-5:],padded[:,:,:,-5:])
    assert not torch.equal(shifted_pad[:,:,:10,:10],padded[:,:,:10,:10])


def test_invalid_or_ambiguous_inputs_fail_closed():
    x=torch.rand(1,3,10,16)
    try:apply_appearance(x,sample_appearance(1,1))
    except ValueError as ex: assert 'normalized' in str(ex)
    else:raise AssertionError('Accepted normalized pixels without protocol')
    for bad in [-.01,1.01,float('nan')]:
        try:sample_appearance(1,1,strength=bad)
        except ValueError:pass
        else:raise AssertionError('Accepted invalid magnitude')
    x=x*255;x[0,0,1,1]=float('nan')
    try:apply_appearance(x,sample_appearance(1,1))
    except FloatingPointError:pass
    else:raise AssertionError('Accepted nan input')


def fixture_recon(*,task_delta=.0,small_delta=0.,bg_delta=0.):
    t=torch.tensor(1.0,requires_grad=True)
    zero=t*0.
    return {
       'total':t+task_delta,
       'scales':{
           'small':{'box':t*.4+small_delta,'classification':t*.1,'positive_objectness':t*.1},
           'medium':{'box':zero,'classification':zero,'positive_objectness':zero},
           'large':{'box':zero,'classification':zero,'positive_objectness':zero}},
       'components':{'background_objectness':t*.2+bg_delta},
       'counts':{'small':3,'medium':0,'large':0,'background':50},
       'num_foreground_normalizer':6,
    }


def test_fixed_target_scale_and_background_nonreg_sensitivity():
    base=fixture_recon()
    adapted=fixture_recon(task_delta=-.2,small_delta=.1,bg_delta=.1)
    loss,diag=meta_objective(base,adapted,objective_mode='detector_total',nonregression_weight=1)
    # 0.1 scale increase, reweighted from 6 FG normalizer to 3 small positives.
    assert abs(diag['per_scale']['small']-.2)<1e-5
    assert diag['background_violation']>0
    assert 'medium' not in diag['per_scale']
    assert torch.isfinite(loss)
    loss.backward()
    assert adapted['total'].grad_fn is not None
    loss2,_=meta_objective(base,adapted,objective_mode='scale_balanced',nonregression_weight=.5)
    assert torch.isfinite(loss2)
    assert float(loss2.detach())!=float(loss.detach())


def test_optional_teacher_has_no_targets_on_unmatched_anchors():
    torch.manual_seed(5)
    pred=torch.randn(1,12,14,requires_grad=True)
    teacher=torch.randn(1,12,14)
    mask=torch.zeros(12,dtype=torch.bool);mask[1]=True;mask[7]=True
    rec=SimpleNamespace(stored=[SimpleNamespace(match=[None,mask])])
    loss,count=reference_consistency(teacher,pred,rec,(1,3,64,64))
    assert count==2 and loss.requires_grad
    g,=torch.autograd.grad(loss,pred)
    assert g[0,mask].abs().sum()>0
    assert torch.equal(g[0,~mask],torch.zeros_like(g[0,~mask]))
    # no matched objects => differentiable zero loss without label leakage.
    rec.stored[0].match[1].zero_()
    loss0,count0=reference_consistency(teacher,pred,rec,(1,3,64,64))
    assert count0==0 and float(loss0.detach())==0


def report_image(index,native,clean0,clean1,shift0,shift1,small=False):
    def obj(delta):
        return {'native':native,'delta':delta,
          'scales':{'small':{'native_per_matched_positive':1.,
                  'adapted_minus_native_per_positive':delta*2}} if small else {},
          'background':{'delta_per_background_anchor':delta/100.}}
    return {'index':index,'clean_sha256':'samec','shifted_sha256':'sames',
            'clean':obj(clean0),'shifted':obj(shift0)},\
           {'index':index,'clean_sha256':'samec','shifted_sha256':'sames',
            'clean':obj(clean1),'shifted':obj(shift1)}


def test_holdout_summary_separate_clean_and_shifted_and_scale_harm():
    a,b=report_image(500,2.,-.0002,-.0003,.0003,-.0003,small=True)
    c,d=report_image(501,2.,-.0001,.0002,-.0001,.0002)
    # Simulate a scale regression despite an aggregate improvement.
    b['clean']['scales']['small']['adapted_minus_native_per_positive']=+.1
    x=summarize_views([a,c],[b,d])
    assert x['shifted']['better_than_initial']==1
    assert x['shifted']['worse_than_initial']==1
    assert x['clean']['scale_coverage']['small']==1
    assert not x['clean']['per_image'][0]['strict_all_present_scales_and_bg']  # scale worse than native
    d['shifted_sha256']='tampered'
    try:summarize_views([a,c],[b,d])
    except RuntimeError:pass
    else:raise AssertionError('Allowed changed holdout augmented pixels')


def test_summarizer_preserves_component_and_scale_denominators():
    a,b=report_image(2,1.,-.0001,-.0002,.0002,-.0001,small=True)
    def extend(r):
        for view in ('clean','shifted'):
            r[view]['components']={k:0. for k in ('box','classification','positive_objectness','background_objectness')}
            r[view]['counts']={'foreground':2}
            r[view]['scales']['small']['count']=2
    extend(a);extend(b)
    out=summarize_report({'initial_holdout':[a], 'trained_holdout':[b]})
    assert out['shifted']['count']==1
    assert out['shifted']['scale_matched_positive_count']['small']==2
    assert out['clean']['scale_mean_per_positive_delta']['medium'] is None
    assert out['shifted']['paired_training_effect_mean']<0

def test_sidecar_is_saved_atomically_without_overwrite():
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'a.pth'
        m=torch.nn.Linear(4,3)
        _atomic_save_sidecar(path,m,{'x':1},[{'step':1}],[{'index':1}],0.01)
        a=torch.load(path,map_location='cpu',weights_only=False)
        assert a['configuration']=={'x':1}
        try:_atomic_save_sidecar(path,m,{},[],[],0.)
        except FileExistsError:pass
        else:raise AssertionError('Clobbered sidecar')


if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):fn();print('PASS',name)
