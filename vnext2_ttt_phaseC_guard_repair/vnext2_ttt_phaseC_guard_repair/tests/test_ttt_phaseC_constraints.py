"""Portable regression checks. Run with: python tests/test_ttt_phaseC_constraints.py"""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from ttt_phaseC_constraints import ConstraintConfig,detector_unit_objective


def fixture(param=None, *, count=90, bg=12000, scale='small'):
    p=torch.tensor(0.0,requires_grad=True) if param is None else param
    base=p*0.0
    nat={'total':base+1.,'components':{'background_objectness':base+0.4},
         'counts':{'small':20 if scale=='small' else 0,
                   'medium':20 if scale=='medium' else 0,
                   'large':20 if scale=='large' else 0,
                   'background':bg,'foreground':count,'empty_images':0},
         'num_foreground_normalizer':count,
         'scales':{s:{k:base+0. for k in ('box','classification','positive_objectness')}
                   for s in ('small','medium','large')}}
    ada={'total':base+1. - .01, 'components':{'background_objectness':base+0.4+p},
         'counts':nat['counts'].copy(),'num_foreground_normalizer':count,
         'scales':{s:{k:base+0. for k in ('box','classification','positive_objectness')}
                   for s in ('small','medium','large')}}
    return p,nat,ada


def test_background_gradient_in_same_units():
    p=torch.tensor(.01,requires_grad=True)
    _,n,a=fixture(p)
    loss,info=detector_unit_objective(n,a,clean_view=False,
                 config=ConstraintConfig(background_weight=.5,scale_weight=0.))
    # New BG extra penalty derivative is 0.5, NOT 0.5*90/12000.
    # In the old Phase B guard it was only .5*.1*90/12000=0.000375.
    g,=torch.autograd.grad(loss,p)
    assert abs(g.item()-.5)<1e-5,g
    assert info['background_extra_gradient_multiplier_when_active']==.5
    assert info['background_guard_active'] is True
    p2=torch.tensor(.01,requires_grad=True)
    _,n2,a2=fixture(p2)
    loss2,diag=detector_unit_objective(n2,a2,clean_view=False,
                     config=ConstraintConfig(background_weight=.5,scale_weight=0.))
    g2,=torch.autograd.grad(loss2,p2)
    assert abs(g2.item()-.5)<1e-5
    assert diag['background_guard_active']
    assert abs(diag['background_delta_detector_units']-.01)<1e-6
    print('PASS detector-unit background guard and gradient coefficient')


def test_explicit_background_margin_in_detector_units():
    p=torch.tensor(.000054,requires_grad=True)
    _,n,a=fixture(p)
    _,stats=detector_unit_objective(n,a,clean_view=False,
             config=ConstraintConfig(background_margin=2e-5))
    assert stats['background_guard_active']
    assert stats['background_violation']>3e-5
    _,stats2=detector_unit_objective(n,a,clean_view=False,
             config=ConstraintConfig(background_margin=1e-4))
    assert not stats2['background_guard_active']
    print('PASS calibrated background margin semantics')


def test_clean_hinge_only_for_clean_episode():
    p=torch.tensor(.03,requires_grad=True)
    _,n,a=fixture(p)
    a['total']=n['total']+.01*p
    config=ConstraintConfig(background_weight=0,scale_weight=0,clean_total_weight=2.)
    clean,cs=detector_unit_objective(n,a,clean_view=True,config=config)
    shift,ss=detector_unit_objective(n,a,clean_view=False,config=config)
    assert cs['clean_total_guard_active'] and not ss['clean_total_guard_active']
    assert abs(torch.autograd.grad(clean,p,retain_graph=True)[0].item()-.03)<1e-5
    assert abs(torch.autograd.grad(shift,p)[0].item()-.01)<1e-5
    print('PASS clean total nonregression separate from shifted view')


def test_scale_only_present_and_amplification_explicit():
    p=torch.tensor(.03,requires_grad=True)
    _,n,a=fixture(p,scale='large')
    a['scales']['large']['box']=p*.1
    loss,stats=detector_unit_objective(n,a,clean_view=False,
                 config=ConstraintConfig(background_weight=0,scale_weight=1.))
    assert list(stats['scale_delta_per_matched_positive'])==['large']
    assert abs(stats['scale_delta_per_matched_positive']['large']-.0135)<1e-6
    g,=torch.autograd.grad(loss,p)
    assert abs(g.item()-.45)<1e-5
    print('PASS scale-specific matched-positive normalization and absent-scale handling')


def test_identity_nonreg_penalties_zero_and_reject_bad_config():
    p,n,a=fixture()
    a['total']=n['total'];a['components']['background_objectness']=n['components']['background_objectness']
    _,stats=detector_unit_objective(n,a,clean_view=True)
    assert stats['background_violation']==0
    assert stats['clean_total_penalty']==0
    assert stats['scale_penalty_mean']==0
    try:ConstraintConfig(background_weight=-1).validate()
    except ValueError:pass
    else:raise AssertionError('negative weight allowed')
    a['counts']['background']+=1
    try:detector_unit_objective(n,a,clean_view=False)
    except RuntimeError:pass
    else:raise AssertionError('changed assignments allowed')
    print('PASS zero-delta constraints and strict fixed-target accounting')

if __name__=='__main__':
    torch.set_num_threads(1)
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):fn()
