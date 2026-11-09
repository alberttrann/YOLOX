"""Check alternating-path repair, legal-geometry, and impossible-coverage cases."""
import torch
from test_a41_cpu import A41Head


def test_augmenting_path():
    head=A41Head(3,width=.25)
    cost=torch.tensor([[0.10, 1.10, float('inf')],
                       [float('inf'), 0.10, 1.00],
                       [0.15,float('inf'),float('inf')]])
    ious=torch.tensor([[.90,.20,0.],
                       [0.,.90,.30],
                       [.80,0.,0.]])
    gt=torch.tensor([0.,1.,2.]);candidates=torch.ones(3,dtype=torch.bool)
    n,classes,ious_matched,ids=head.simota_matching(cost,ious,gt,3,candidates)
    assert n==3,(n,ids,head._a41_last_repair)
    assert set(ids.tolist())=={0,1,2},ids
    assert head._a41_last_repair['augmenting_paths_successful']==1
    assert head._a41_last_repair['missing_after']==0
    # Returned class order corresponds to GT assignments after reallocation.
    assert set(classes.int().tolist())=={0,1,2}
    print('PASS GT coverage recovered via alternating assignment chain')


def test_impossible_geometry():
    head=A41Head(2,width=.25)
    cost=torch.tensor([[.1,float('inf')],[.2,float('inf')]])
    ious=torch.tensor([[.9,0.],[.8,0.]])
    gt=torch.tensor([0.,1.]);mask=torch.ones(2,dtype=torch.bool)
    n,classes,iou,ids=head.simota_matching(cost,ious,gt,2,mask)
    assert n==1
    assert head._a41_last_repair['missing_after']==1
    assert not bool(mask[1]), 'Must never assign a geometrically invalid anchor'
    print('PASS impossible assignment remains explicitly uncovered')


if __name__=='__main__':
    torch.set_num_threads(2)
    test_augmenting_path();test_impossible_geometry()
