"""vNext.3 native-preserving head. Requires upstream YOLOX + archived TDE head.

Unlike Engram/FiLM, PG-SMA never enters box/objectness/classifier towers.
The former Engram parameters are *removed*, not left registered and unused.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .engram_head import TDE_Head
from .losses import IOUloss
from .scpa import scpa_assign


class PGSMAHead(TDE_Head):
    def __init__(self, num_classes, width=0.5, in_channels=(256, 512, 1024),
                 act='silu', loss_type='iou', assignment='scpa',
                 per_gt_normalization=False):
        super().__init__(num_classes=num_classes, width=width,
                         in_channels=list(in_channels), act=act, loss_type=loss_type)
        if assignment not in ('scpa', 'baseline'):
            raise ValueError(assignment)
        self.assignment = assignment
        self.use_baseline_assignments = (assignment == 'baseline')
        self.per_gt_normalization = bool(per_gt_normalization)
        self.iou_loss = IOUloss(reduction='none', loss_type=loss_type)
        self.enable_memory_restoration = False
        for old in ('memory_banks','latent_projectors','spatial_smoothers',
                    'memory_proj_back','memory_norms','mem_gate','gamma_reg'):
            if hasattr(self, old): delattr(self, old)
        self.pgsma_cache = None
        self.last_assignment_stats = {}
        self.scpa_initial_candidates = 12
        self.scpa_max_candidates = 128
        self.scpa_extra_positives = 3

    def forward(self, xin, labels=None, imgs=None, valid_anchor_mask=None):
        outputs = []
        origin_preds = []
        x_shifts = []; y_shifts = []; expanded_strides = []
        features = []; native_cls = []; native_obj = []
        for k, (cls_conv, reg_conv, stride, x) in enumerate(zip(
                self.cls_convs, self.reg_convs, self.strides, xin)):
            stem = self.stems[k](x)
            reg_feat = reg_conv(stem)
            cls_feat = cls_conv(stem)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            cls_output = self.cls_preds[k](cls_feat)
            features.append(cls_feat.detach())
            native_cls.append(cls_output.detach())
            native_obj.append(obj_output.detach())
            if self.training:
                raw = torch.cat([reg_output, obj_output, cls_output], dim=1)
                encoded, grid = self.get_output_and_grid(raw, k, stride, xin[0].type())
                outputs.append(encoded)
                x_shifts.append(grid[:, :, 0]); y_shifts.append(grid[:, :, 1])
                expanded_strides.append(grid.new_full((1, grid.shape[1]), stride))
                if self.use_l1:
                    b, _, h, w = reg_output.shape
                    origin_preds.append(reg_output.view(b, 1, 4, h, w).permute(
                        0, 1, 3, 4, 2).reshape(b, -1, 4))
            else:
                outputs.append(reg_output)
        if self.training:
            # Training-only cache; do not mutate module state with feature
            # tensors during inference or ONNX export.
            self.pgsma_cache = {'features': features, 'native_cls': native_cls,
                                'native_obj': native_obj, 'match': None}
            if labels is None:
                raise ValueError('Native training requires labels')
            return self._native_losses(torch.cat(outputs, dim=1), labels,
                                       x_shifts, y_shifts, expanded_strides,
                                       origin_preds, valid_anchor_mask)
        mode = getattr(self, 'inference_semantic_mode', 'pgsma')
        if hasattr(self, 'pgsma_engine'):
            corrected = self.pgsma_engine.forward_levels(
                features, native_cls, native_obj, mode=mode)
        else:
            corrected = native_cls
        # Shape-safe per-level decoding; no self.hw / stale Python grid cache.
        flat_levels=[]
        for reg, obj, cls, stride in zip(outputs, native_obj, corrected, self.strides):
            b, _, h, w = reg.shape
            raw = torch.cat([reg, obj.sigmoid(), cls.sigmoid()],dim=1)
            pred = raw.flatten(start_dim=2).transpose(1,2)
            if self.decode_in_inference:
                yy, xx = torch.meshgrid(torch.arange(h,device=reg.device),
                                        torch.arange(w,device=reg.device),indexing='ij')
                grid = torch.stack((xx,yy),dim=-1).reshape(1,-1,2).to(pred.dtype)
                xy = (pred[...,:2]+grid)*stride
                wh = torch.exp(pred[...,2:4].clamp(min=-5,max=7))*stride
                pred = torch.cat([xy,wh,pred[...,4:]],dim=-1)
            flat_levels.append(pred)
        return torch.cat(flat_levels,dim=1)

    def _native_losses(self, outputs, labels, x_shift, y_shift, strides,
                       origin_preds, valid_anchor_mask):
        # SCPA requires detached predictions; native gradients only through losses.
        bsz, total, _ = outputs.shape
        bboxes = outputs[..., :4].float()
        obj = outputs[..., 4:5].float()
        cls = outputs[..., 5:].float()
        xshift = torch.cat(x_shift, dim=1).float().to(bboxes.device)
        yshift = torch.cat(y_shift, dim=1).float().to(bboxes.device)
        stride = torch.cat(strides, dim=1).float().to(bboxes.device)
        centers = torch.stack(((xshift[0]+.5)*stride[0], (yshift[0]+.5)*stride[0]), dim=1)
        if valid_anchor_mask is not None:
            if valid_anchor_mask.shape != (bsz, total):
                raise ValueError(f'valid_anchor_mask {tuple(valid_anchor_mask.shape)} '
                                 f'!= expected {(bsz,total)}')
            valid_anchor_mask = valid_anchor_mask.to(device=bboxes.device,dtype=torch.bool)
        targets_bbox = []; predictions_bbox = []; cls_target = []; cls_pred = []
        obj_target = torch.zeros_like(obj)
        flat_pos = []; flat_lab = []; flat_quality = []; flat_uid = []
        fg_mask = torch.zeros((bsz, total), device=bboxes.device, dtype=torch.bool)
        gt_by_image = []; flat_sizes = []
        total_gt = 0; uncovered = 0; edge_total = 0; core_count = 0
        for b in range(bsz):
            image_labels = labels[b] if torch.is_tensor(labels) else labels[b]
            image_labels = image_labels.float().to(bboxes.device)
            valid_gt = ((image_labels[:, 3] > 0) & (image_labels[:, 4] > 0) &
                        torch.isfinite(image_labels).all(dim=1))
            boxes = image_labels[valid_gt, 1:5]
            gt_by_image.append(boxes.detach())
            classes = image_labels[valid_gt, 0].long()
            n_gt = len(boxes)
            if classes.numel() and ((classes < 0).any() or (classes >= self.num_classes).any()):
                raise ValueError('Class ID out of valid configured taxonomy')
            if valid_anchor_mask is None:
                valid_anchors = torch.ones(total, device=bboxes.device, dtype=torch.bool)
            else:
                valid_anchors = valid_anchor_mask[b].to(device=bboxes.device, dtype=torch.bool)
            if self.assignment == 'scpa':
                match = scpa_assign(boxes, classes, bboxes[b], cls[b], obj[b],
                                    centers, stride[0], valid_anchors,
                                    self.scpa_initial_candidates,
                                    self.scpa_max_candidates,
                                    self.scpa_extra_positives)
                ai = match.anchor_indices; gi = match.gt_indices
                quality = match.matched_ious
                uncovered += match.uncovered_gt.numel(); edge_total += match.candidate_edges
                core_count += int(match.core.sum().item())
            else:
                # Explicit scientific control. Native YOLOX matching API.
                if n_gt:
                    (matched_cls, mask, quality, gi, _nfg) = super().get_assignments(
                        b, n_gt, boxes, classes, bboxes[b], stride, xshift, yshift,
                        cls, obj)
                    ai = torch.nonzero(mask, as_tuple=False).flatten()
                else:
                    ai = torch.empty(0, device=bboxes.device, dtype=torch.long)
                    gi = ai; quality = bboxes.new_empty(0)
            ai = ai.long(); gi = gi.long(); quality = quality.float().detach()
            if ai.numel():
                if not valid_anchors[ai].all():
                    raise RuntimeError('Positive assigned on invalid/padded region')
                fg_mask[b, ai] = True
                obj_target[b, ai] = 1.0  # consistent binary objectness protocol
                targets_bbox.append(boxes[gi]); predictions_bbox.append(bboxes[b, ai])
                target = F.one_hot(classes[gi], self.num_classes).float() * quality[:, None]
                cls_target.append(target)
                cls_pred.append(cls[b, ai])
                flat_pos.append(ai + b * total)
                flat_lab.append(classes[gi]); flat_quality.append(quality)
                flat_uid.append(gi + total_gt)
                flat_sizes.append(boxes[gi,2:4].amin(dim=1).detach())
            total_gt += n_gt
        n_fg = sum(len(x) for x in flat_pos)
        empty = torch.empty(0, dtype=torch.long, device=bboxes.device)
        fg_ids = torch.cat(flat_pos) if flat_pos else empty
        self.pgsma_cache['match'] = {
            'fg_ids': fg_ids.detach(),
            'fg_labels': torch.cat(flat_lab).detach() if flat_lab else empty,
            'fg_quality': torch.cat(flat_quality).detach() if flat_quality else bboxes.new_empty(0),
            'gt_uid': torch.cat(flat_uid).detach() if flat_uid else empty,
            'fg_sizes': torch.cat(flat_sizes).detach() if flat_sizes else bboxes.new_empty(0),
            # Semantic negatives must not be sampled inside/near annotated GT.
            # This does NOT change native assignment, loss or objectness targets.
            'bg_ids': self._hard_background(obj.detach(), fg_mask, valid_anchor_mask,
                                            centers=centers, gt_by_image=gt_by_image),
        }
        self.last_assignment_stats = dict(num_fg_total=n_fg, num_gt_total=total_gt,
                                          uncovered_gt=uncovered, core_positives=core_count,
                                          candidate_edges=edge_total)
        zero = obj.sum() * 0.0
        if n_fg:
            per_iou = self.iou_loss(torch.cat(predictions_bbox, 0), torch.cat(targets_bbox, 0))
            per_cls = F.binary_cross_entropy_with_logits(torch.cat(cls_pred, 0),
                                                           torch.cat(cls_target, 0), reduction='none').sum(-1)
            if self.per_gt_normalization:
                uid = self.pgsma_cache['match']['gt_uid']
                _, inv, counts = torch.unique(uid, sorted=True, return_inverse=True, return_counts=True)
                denom = max(int(counts.numel()), 1)
                iou_loss = (per_iou / counts[inv]).sum() / denom
                cls_loss = (per_cls / counts[inv]).sum() / denom
            else:
                iou_loss = per_iou.sum() / n_fg
                cls_loss = per_cls.sum() / n_fg
        else:
            iou_loss = zero; cls_loss = zero
        # Keep ordinary YOLOX objectness scaling as a controlled baseline.
        per_obj = F.binary_cross_entropy_with_logits(obj, obj_target, reduction='none')
        if valid_anchor_mask is not None:
            per_obj = per_obj * valid_anchor_mask.unsqueeze(-1).to(per_obj.dtype)
        obj_loss = per_obj.sum() / max(n_fg, 1)
        l1_loss = zero
        if self.use_l1 and n_fg:
            origin = torch.cat(origin_preds, 1).float()
            # Build raw regression targets using standard anchor-grid transform.
            reg_t = torch.cat(targets_bbox, 0)
            pos_mask = fg_mask.reshape(-1)
            target_l1 = self.get_l1_target(reg_t.new_zeros((n_fg,4)), reg_t,
                    stride.expand(bsz,-1).reshape(-1)[pos_mask],
                    xshift.expand(bsz,-1).reshape(-1)[pos_mask],
                    yshift.expand(bsz,-1).reshape(-1)[pos_mask])
            l1_loss = self.l1_loss(origin.reshape(-1,4)[pos_mask], target_l1).sum() / n_fg
        loss = 5.0 * iou_loss + obj_loss + cls_loss + l1_loss
        return {'total_loss': loss, 'det_loss': loss, 'iou_loss': 5.0 * iou_loss,
                'conf_loss': obj_loss, 'cls_loss': cls_loss, 'l1_loss': l1_loss,
                'num_fg': n_fg / max(total_gt, 1), 'num_fg_total': n_fg,
                'num_gt_total': total_gt, 'uncovered_gt': uncovered,
                'core_positives': core_count}

    @staticmethod
    @torch.no_grad()
    def _hard_background(obj, positives, valid_mask, max_neg=256,
                         centers=None, gt_by_image=None, halo=1.25):
        """Native-objectness top-k for semantics, GT-halo filtered.

        GT coordinates are cxcywh in effective training input pixels. Only
        semantic negative mining is changed. Chunked geometry prevents a huge
        anchors-by-GTs allocation on high-resolution UAV imagery.
        """
        b, n, _ = obj.shape
        mask = ~positives
        if valid_mask is not None:
            mask &= valid_mask.to(mask.device).bool()
        if not (centers is None and gt_by_image is None):
            if centers is None or gt_by_image is None or len(gt_by_image) != b:
                raise ValueError('centers and one GT tensor per image required together')
            if centers.shape != (n,2):
                raise ValueError('centers must have shape [num_anchors,2]')
            if not (1.0 <= float(halo) <= 3.0):
                raise ValueError('GT halo outside [1,3]')
        out=[]
        for i in range(b):
            ids = torch.nonzero(mask[i], as_tuple=False).flatten()
            if ids.numel() and centers is not None:
                gt = gt_by_image[i].detach().float()
                if gt.numel():
                    survivors = []
                    for id_chunk in ids.split(2048):
                        xy = centers[id_chunk].float()
                        inside = torch.zeros(len(id_chunk),device=xy.device,dtype=torch.bool)
                        for g in gt.split(64):
                            inside |= ((xy[:,None,:]-g[None,:,:2]).abs() <=
                                       (g[None,:,2:4]*float(halo)/2)).all(-1).any(-1)
                        survivors.append(id_chunk[~inside])
                    ids = torch.cat(survivors)
            if ids.numel():
                k = min(int(max_neg), ids.numel())
                chosen = torch.topk(obj[i,ids,0],k=k,sorted=False).indices
                out.append(ids[chosen]+i*n)
        return (torch.cat(out).detach() if out else
                torch.empty(0,device=obj.device,dtype=torch.long))
