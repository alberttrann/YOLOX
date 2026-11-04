import torch

def test_matching_exact_cardinality(src):
    m=src['scpa']
    assert m.maximum_matching([[0,1],[0]],2)==[1,0]
    result=m.maximum_matching([[0],[0]],1)
    assert sum(a>=0 for a in result)==1
    assert len([a for a in result if a>=0])==len(set(a for a in result if a>=0))
    assert m.maximum_matching([],5)==[]


def test_assign_coverage_conflict_and_dynamic(src):
    m=src['scpa']
    boxes=torch.tensor([[8.,8.,8.,8.],[8.1,8.1,8.,8.]])
    gtclasses=torch.tensor([0,1])
    centers=torch.tensor([[8.,8.],[16.,8.],[16.,16.],[8.,16.]])
    pred=torch.cat([centers,torch.full((4,2),8.)],dim=1)
    cl=torch.zeros(4,2);ob=torch.zeros(4,1)
    r=m.scpa_assign(boxes,gtclasses,pred,cl,ob,centers,
           torch.full((4,),8.),initial_candidates=1,max_candidates=4,extra_per_gt=1)
    assert len(r.uncovered_gt)==0
    assert len(r.anchor_indices)==len(torch.unique(r.anchor_indices))
    assert int(r.core.sum())==2
    assert len(r.matched_ious)==len(r.anchor_indices)
    assert m.scpa_assign(boxes[:0],gtclasses[:0],pred,cl,ob,centers,
                         torch.full((4,),8.)).anchor_indices.numel()==0
    r2=m.scpa_assign(boxes,gtclasses,pred,cl,ob,centers,
           torch.full((4,),8.),initial_candidates=1,max_candidates=1,extra_per_gt=0)
    assert r2.uncovered_gt.numel()==1 and r2.budget_exhausted


def test_iou_metric(src):
    m=src['scpa']
    q=torch.tensor([[4.,4.,2.,2.],[3.,3.,1.,1.]])
    i=m.aligned_iou_xywh(q,q)
    assert torch.allclose(i,torch.ones_like(i))


def test_scpa_refuses_unrepresentable_far_gt(src):
    m=src['scpa']
    gt=torch.tensor([[1000.,1000.,2.,2.]])
    anchors=torch.tensor([[4.,4.],[12.,12.]])
    boxes=torch.cat([anchors,torch.ones(2,2)],dim=1)
    r=m.scpa_assign(gt,torch.tensor([0]),boxes,torch.zeros(2,1),torch.zeros(2,1),
                    anchors,torch.full((2,),8.),initial_candidates=1,max_candidates=2)
    assert r.anchor_indices.numel()==0
    assert r.uncovered_reasons == ('no_eligible_geometry',)
    assert not r.budget_exhausted
