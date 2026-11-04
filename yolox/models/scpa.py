"""SCPA: geometry-first sparse, coverage-preserving training assignment.

This module is independent of YOLOX and never runs during inference.
Coverage is exact *on the enumerated bipartite graph*. Search-budget exhaustion is
reported and is not a statement that an object is physically unrepresentable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Assignment:
    anchor_indices: torch.Tensor  # long, unique, ordered by anchor
    gt_indices: torch.Tensor      # long, local per-image GT index
    matched_ious: torch.Tensor    # float, detached
    core: torch.Tensor            # bool: protected coverage match
    uncovered_gt: torch.Tensor    # long indices, if any
    budget_exhausted: bool
    candidate_edges: int
    uncovered_reasons: tuple = ()


def maximum_matching(adjacency: List[List[int]], num_anchors: int):
    """Exact Hopcroft-Karp maximum cardinality. Returns GT->anchor (-1 if lost)."""
    n = len(adjacency)
    left = [-1] * n
    right = [-1] * num_anchors
    dist = [0] * n
    infinity = n + 1

    def bfs():
        q = deque()
        found = False
        for g in range(n):
            if left[g] == -1:
                dist[g] = 0
                q.append(g)
            else:
                dist[g] = infinity
        while q:
            g = q.popleft()
            for a in adjacency[g]:
                partner = right[a]
                if partner < 0:
                    found = True
                elif dist[partner] == infinity:
                    dist[partner] = dist[g] + 1
                    q.append(partner)
        return found

    def dfs(g):
        for a in adjacency[g]:
            partner = right[a]
            if partner == -1 or (dist[partner] == dist[g] + 1 and dfs(partner)):
                right[a] = g
                left[g] = a
                return True
        dist[g] = infinity
        return False

    while bfs():
        for g in range(n):
            if left[g] == -1:
                dfs(g)
    return left


def aligned_iou_xywh(pred: torch.Tensor, gt: torch.Tensor):
    """Paired IoU for boxes in cx,cy,w,h. Finite for zero-area predictions."""
    pmin = pred[:, :2] - pred[:, 2:].clamp(min=0) / 2
    pmax = pred[:, :2] + pred[:, 2:].clamp(min=0) / 2
    gmin = gt[:, :2] - gt[:, 2:].clamp(min=0) / 2
    gmax = gt[:, :2] + gt[:, 2:].clamp(min=0) / 2
    inter = (torch.minimum(pmax, gmax) - torch.maximum(pmin, gmin)).clamp(min=0).prod(-1)
    pa = pred[:, 2:].clamp(min=0).prod(-1)
    ga = gt[:, 2:].clamp(min=0).prod(-1)
    return inter / (pa + ga - inter).clamp(min=1e-8)


@torch.no_grad()
def scpa_assign(gt_boxes, gt_classes, pred_boxes, cls_logits, obj_logits,
                centers, strides, valid_mask=None, initial_candidates=12,
                max_candidates=128, extra_per_gt=3, cost_iou_weight=3.0):
    """Exact max-cardinality cover on a geometry-eligible *sparse* graph.

    Eligible if an anchor is within the GT's own half-extent plus ONE grid
    stride in both axes. This is a dimensionless geometric representability
    envelope, not a weather- or category-specific match score. Out-of-envelope
    matches are never fabricated to increase nominal coverage.
    """
    device = pred_boxes.device
    n_gt = int(gt_boxes.shape[0]); n_all = int(pred_boxes.shape[0])
    empty = torch.empty(0, device=device, dtype=torch.long)
    if valid_mask is None:
        valid_mask = torch.ones(n_all, device=device, dtype=torch.bool)
    valid = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if n_gt == 0 or valid.numel() == 0:
        reason = tuple('no_valid_anchor' for _ in range(n_gt))
        return Assignment(empty, empty, pred_boxes.new_empty(0),
                          torch.empty(0, device=device, dtype=torch.bool),
                          torch.arange(n_gt, device=device), False, 0, reason)
    if (gt_boxes[:, 2:] <= 0).any() or not torch.isfinite(gt_boxes).all():
        raise ValueError('All input GT boxes must have finite positive width/height')
    if not (1 <= int(initial_candidates) <= int(max_candidates)):
        raise ValueError('Require 1 <= initial_candidates <= max_candidates')
    gt_boxes = gt_boxes.float()
    boxes = pred_boxes.detach().float()
    ctr = centers[valid].float()
    st = strides[valid].float().flatten().clamp(min=1)
    # Memory-bounded candidate generation: stream GTs rather than allocate
    # a [num_GT, num_anchors, 2] matrix on high-density 4K UAV scenes.
    n_valid = valid.numel()
    max_budget = min(int(max_candidates), n_valid)
    width = min(int(initial_candidates), max_budget)
    candidate_rows = []
    count_eligible = []
    for g in range(n_gt):
        delta = (ctr - gt_boxes[g, :2]).abs()
        envelope = gt_boxes[g, 2:4] * 0.5 + st[:, None]
        eligible = (delta <= envelope).all(dim=-1)
        local_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        n_eligible = int(local_ids.numel())
        count_eligible.append(n_eligible)
        if n_eligible:
            scale = torch.maximum(gt_boxes[g, 2:4], st[local_ids,None])
            distances = ((delta[local_ids] / scale)**2).sum(-1)
            k = min(max_budget, n_eligible)
            ordering = torch.topk(distances,k,largest=False,sorted=True).indices
            candidate_rows.append(local_ids[ordering].cpu().tolist())
        else:
            candidate_rows.append([])

    def adjacency_for(w):
        return [row[:w] for row in candidate_rows]

    while True:
        adjacency = adjacency_for(width)
        core = maximum_matching(adjacency, n_valid)
        if all(a >= 0 for a in core) or width == max_budget:
            break
        # Expansion is only useful while there are additional eligible edges.
        if not any(count_eligible[g] > width for g,a in enumerate(core) if a < 0):
            break
        width = min(max_budget,max(width+1,width*2))

    # Preserve sparse core matches, then add nonstealing extra positives ordered
    # by detached native matching cost. No NxGT dense pairwise IoU matrix.
    chosen = {a:(g,True) for g,a in enumerate(core) if a>=0}
    per_gt_count = [int(a>=0) for a in core]
    edge_list = [(g,int(a)) for g,row in enumerate(adjacency) for a in row]
    if edge_list:
        pair_g = torch.tensor([g for g,a in edge_list],device=device,dtype=torch.long)
        pair_local = torch.tensor([a for g,a in edge_list],device=device,dtype=torch.long)
        pair_a = valid[pair_local]
        ious = aligned_iou_xywh(boxes[pair_a],gt_boxes[pair_g])
        cp = torch.sigmoid(cls_logits.detach().float()[pair_a,gt_classes.long()[pair_g]])
        op = torch.sigmoid(obj_logits.detach().float().reshape(-1)[pair_a])
        cost = -torch.log((cp*op).clamp(min=1e-8))+float(cost_iou_weight)*(1-ious)
        if not torch.isfinite(cost).all():
            raise FloatingPointError('Nonfinite SCPA candidate costs')
        for e in torch.argsort(cost,stable=True).cpu().tolist():
            g,loc=edge_list[e]
            if loc in chosen or per_gt_count[g]>=1+max(0,int(extra_per_gt)):
                continue
            chosen[loc]=(g,False)
            per_gt_count[g]+=1
    ordered_chosen = sorted(chosen)
    ids = valid[torch.tensor(ordered_chosen,device=device,dtype=torch.long)]
    gt_ids = torch.tensor([chosen[a][0] for a in ordered_chosen],device=device,dtype=torch.long)
    core_flags = torch.tensor([chosen[a][1] for a in ordered_chosen],device=device,dtype=torch.bool)
    quality = aligned_iou_xywh(boxes[ids],gt_boxes[gt_ids]) if ids.numel() else boxes.new_empty(0)
    uncovered = [g for g,a in enumerate(core) if a<0]
    reasons = tuple('no_eligible_geometry' if count_eligible[g]==0
                    else ('candidate_budget_exhausted' if count_eligible[g]>width
                          else 'infeasible_on_enumerated_graph') for g in uncovered)
    return Assignment(ids,gt_ids,quality,core_flags,
                      torch.tensor(uncovered,device=device,dtype=torch.long),
                      any(r=='candidate_budget_exhausted' for r in reasons),
                      len(edge_list),reasons)
