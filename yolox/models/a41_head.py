"""A4.1 class-grounded semantic memory with non-starved query gradients.

The direct prototype CE trains query projectors and prototypes using detached
classification input features. It cannot directly backprop to spectral/stem/
neck/backbone through that auxiliary path. Ordinary detection gradients still
reach all shared layers; gradient conflicts MUST be measured empirically.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from .a4_head import A4Head
from .a41_spectral import A41SpectralResidual


class A41Head(A4Head):
    def __init__(self, num_classes, width=0.5, in_channels=(256,512,1024),
                 act='silu', loss_type='oiou', basis='dct', proto_weight=0.05):
        super().__init__(num_classes, width=width, in_channels=in_channels,
                         act=act, loss_type=loss_type, basis=basis)
        if not 0.0 <= float(proto_weight) <= 1.0:
            raise ValueError('proto_weight must be in [0,1]')
        c = int(256 * width)
        self.spectral_adapters = nn.ModuleList([
            A41SpectralResidual(c, basis=basis) for _ in self.cls_convs])
        self.fourier_variant = 'a41' if basis == 'dct' else 'a41_control'
        self.proto_weight = float(proto_weight)
        self.aux_memory_loss_raw = None
        self.last_proto_diagnostic = {}
        self._a41_detached_features = {}
        self.assignment_repair_enabled = True
        self.repair_candidates_per_gt = 128
        self.repair_max_depth = 32
        self.assignment_repair_log = []
        # Give task CE a small real memory contribution at step zero, bounded
        # downstream by tanh(mem) * channel RMS, rather than hard zero gate.
        with torch.no_grad():
            self.mem_gate.fill_(math.atanh(0.06))

    def forward(self, xin, labels=None, imgs=None):
        self._a41_detached_features = {}
        self.aux_memory_loss_raw = None
        self.last_proto_diagnostic = {}
        self.assignment_repair_log = []
        return super().forward(xin, labels, imgs)

    def _classification_memory(self, k, cls_feat):
        if self.training and self.memory_alignment_enabled:
            # Store the input to the query projector. Detach BEFORE projection.
            self._a41_detached_features[k] = cls_feat.detach()
        return super()._classification_memory(k, cls_feat)

    @torch.no_grad()
    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """Native dynamic-K + eligible-only limited augmenting-path repair.

        A4 greedy coverage repair cannot swap ownership along alternating
        anchor paths. We only invoke the bounded DFS on still-uncovered GT,
        and we never assign an anchor with invalid geometry.
        """
        if not self.assignment_repair_enabled:
            return super().simota_matching(cost,pair_wise_ious,gt_classes,num_gt,fg_mask)
        if cost.shape[1] == 0:
            raise RuntimeError('SimOTA has zero anchor candidates')
        finite = torch.isfinite(cost)
        topk = min(10, int(cost.shape[1]))
        valid_iou = pair_wise_ious.masked_fill(~finite, 0)
        best_ious = torch.topk(valid_iou, topk, dim=1).values
        dynamic_k = torch.clamp(best_ious.sum(1).int(), min=1, max=topk)
        matching = torch.zeros_like(finite)
        for gt in range(num_gt):
            valid_count = int(finite[gt].sum())
            if valid_count == 0:continue
            count = min(int(dynamic_k[gt]), valid_count)
            selected = cost[gt].topk(count, largest=False).indices
            matching[gt, selected] = True
        conflicts = matching.sum(0)>1
        if bool(conflicts.any()):
            shared = conflicts.nonzero(as_tuple=True)[0]
            owners = cost[:,shared].argmin(0)
            matching[:,shared] = False
            matching[owners,shared] = True
        missing_before = int((matching.sum(1)==0).sum())
        # Native A4 first recovers all uncovered GT that have an UNUSED
        # eligible anchor. This fast path handles most ordinary crowds.
        unused = matching.sum(0)==0
        for gt_tensor in (matching.sum(1)==0).nonzero(as_tuple=True)[0]:
            gt=int(gt_tensor)
            allowed = finite[gt] & unused
            if not bool(allowed.any()):continue
            anchor = int(cost[gt].masked_fill(~allowed,float('inf')).argmin())
            matching[gt,anchor] = True
            unused[anchor] = False
        attempts = 0
        repaired = 0
        capped = 0
        if bool((matching.sum(1)==0).any()):
            # Resolve genuine coverage conflicts by alternating-path search.
            # The search is bounded to avoid quadratic UAV crowd-time spikes.
            def ranked_choices(gt):
                options = torch.where(finite[gt])[0]
                if options.numel()==0:return []
                nonlocal capped
                if options.numel()>self.repair_candidates_per_gt:
                    capped += 1
                    best = torch.topk(cost[gt,options],self.repair_candidates_per_gt,
                                      largest=False).indices
                    options=options[best]
                else:
                    options=options[torch.argsort(cost[gt,options])]
                return [int(v) for v in options.tolist()]
            def augment(gt, seen_gt, seen_anchor, depth):
                if depth > self.repair_max_depth or gt in seen_gt:return False
                seen_gt.add(gt)
                for anchor in ranked_choices(gt):
                    if anchor in seen_anchor:continue
                    seen_anchor.add(anchor)
                    owners=matching[:,anchor].nonzero(as_tuple=True)[0]
                    if owners.numel()==0:
                        matching[gt,anchor]=True
                        return True
                    owner=int(owners[0])
                    if owner==gt:continue
                    if int(matching[owner].sum())>1 or augment(owner,seen_gt,seen_anchor,depth+1):
                        matching[owner,anchor]=False
                        matching[gt,anchor]=True
                        return True
                return False
            for gt_tensor in (matching.sum(1)==0).nonzero(as_tuple=True)[0]:
                gt=int(gt_tensor)
                attempts+=1
                if augment(gt,set(),set(),0):repaired+=1
        used = matching.any(0)
        fg_mask[fg_mask.clone()] = used
        ids = matching[:,used].float().argmax(0).long()
        cls = gt_classes[ids]
        selected_iou = (matching.float()*pair_wise_ious).sum(0)[used]
        self._a41_last_repair={
            'missing_before':missing_before,
            'missing_after':int((matching.sum(1)==0).sum()),
            'augmenting_paths_attempted':attempts,
            'augmenting_paths_successful':repaired,
            'searches_candidate_capped':capped,
        }
        return int(used.sum()),cls,selected_iou,ids

    @torch.no_grad()
    def get_assignments(self, *args, **kwargs):
        self._a41_last_repair = None
        result=super().get_assignments(*args,**kwargs)
        if self._a41_last_repair is not None:
            self.assignment_repair_log.append(dict(self._a41_last_repair))
        return result

    def _get_losses_portable(self, *args, **kwargs):
        result=super()._get_losses_portable(*args,**kwargs)
        if self.assignment_repair_log:
            for key in ('missing_before','missing_after','augmenting_paths_attempted',
                        'augmenting_paths_successful','searches_candidate_capped'):
                self.last_assignment_stats['a41_'+key]=sum(
                    int(m.get(key,0)) for m in self.assignment_repair_log)
        return result

    def _foreground_proto_loss(self):
        if not self.memory_alignment_enabled or not self._matches or not self.proto_weight:
            return None
        if len(self._a41_detached_features) != len(self.cls_convs):
            raise RuntimeError('Missing detached class features for prototype grounding')
        queries = {}
        starts = [0]
        for k in range(len(self.cls_convs)):
            source = self._a41_detached_features[k]
            z = self.latent_projectors[k](source)
            z = z + self.spatial_smoothers[k](z)
            queries[k] = F.normalize(z.float().flatten(2).transpose(1, 2), dim=-1)
            starts.append(starts[-1] + queries[k].shape[1])
        # Per-batch class balanced CE across ALL matched class positives.
        # No hard-coded BDD class priors or confidence cutoffs.
        collected = []
        for batch_index, fg, classes in self._matches:
            ids = fg.nonzero(as_tuple=True)[0]
            if ids.numel() != classes.numel():
                raise RuntimeError('Foreground indices and class assignments differ')
            for level, q in queries.items():
                selection = (ids >= starts[level]) & (ids < starts[level + 1])
                if not bool(selection.any()):
                    continue
                q_foreground = q[batch_index, ids[selection] - starts[level]]
                labels = classes[selection].long()
                bank = self.memory_banks[level]
                prototypes = F.normalize(bank.prototypes.float(), dim=-1)
                sim = torch.einsum('pd,kmd->pkm', q_foreground, prototypes)
                logits = torch.logsumexp(sim * bank.temperature, dim=-1) - math.log(prototypes.shape[1])
                losses = F.cross_entropy(logits, labels, reduction='none')
                collected.append((labels, losses))
        if not collected:
            self.last_proto_diagnostic = {'positive_assignments': 0, 'classes': 0}
            return None
        labels = torch.cat([item[0] for item in collected])
        losses = torch.cat([item[1] for item in collected])
        unique_classes = labels.unique()
        raw = torch.stack([losses[labels == cls].mean() for cls in unique_classes]).mean()
        self.aux_memory_loss_raw = raw.detach()
        self.last_proto_diagnostic = {
            'positive_assignments': int(labels.numel()),
            'classes': int(unique_classes.numel()),
            'proto_weight': self.proto_weight,
            # No float(raw) GPU synchronization on every optimizer step.
            # Raw CE is available as self.aux_memory_loss_raw.detach().
        }
        return raw * self.proto_weight
