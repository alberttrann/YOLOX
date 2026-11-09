"""Complete, source-independent classification integration for tdevNext.3.

This file does NOT rewrite or monkeypatch the user's engram_head.py. It keeps
TDE_Head's pretrained-compatible modules and detection losses/assignment
interfaces while defining an explicit, auditable inference/training forward.

Modes: off (A0), spatial (A1), spectral (A2), spectral_memory (A3).
No runtime FFT, test-time optimizer, Fourier reconstruction, or source-class
margins. Regression/objectness use the same native branch for every variant.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .engram_head import TDE_Head
from yolox.utils import bboxes_iou
from .fourier_invariance import LocalSpectralResidual


class FourierTDEHead(TDE_Head):
    SUPPORTED = ("off", "spatial", "spectral", "spectral_memory")

    def __init__(self, num_classes, width=0.5, in_channels=(256, 512, 1024),
                 act="silu", variant="spectral", loss_type="oiou"):
        if num_classes < 1:
            raise ValueError("Must have at least one object class")
        if variant not in self.SUPPORTED:
            raise ValueError(f"variant must be one of {self.SUPPORTED}, not {variant}")
        super().__init__(num_classes, width=width, in_channels=list(in_channels),
                         act=act, loss_type=loss_type)
        self.fourier_variant = variant
        self.enable_engram = variant == "spectral_memory"
        c = int(256 * width)
        if variant == "spatial":
            self.spectral_adapters = nn.ModuleList(
                [LocalSpectralResidual(c, mode="spatial") for _ in self.cls_convs])
        elif variant in ("spectral", "spectral_memory"):
            self.spectral_adapters = nn.ModuleList(
                [LocalSpectralResidual(c, mode="spectral") for _ in self.cls_convs])
        else:
            self.spectral_adapters = nn.ModuleList([nn.Identity() for _ in self.cls_convs])

        if not self.enable_engram:
            # Disabled branches are not registered: truly no memory compute,
            # params or auxiliary gradients in A0/A1/A2.
            for attr in ("memory_banks", "latent_projectors", "spatial_smoothers",
                         "memory_proj_back", "memory_norms"):
                delattr(self, attr)
            delattr(self, "mem_gate")
        self.last_fourier_stats = {}
        self.last_assignment_stats = {}
        self.initialize_biases(1e-2)

    def set_epoch_state(self, epoch, max_epochs):
        # Explicitly independent of weather/epoch-specific confidence targets.
        self.current_epoch = int(epoch)
        self.max_epochs = int(max_epochs)

    def _classification_memory(self, k, cls_feat):
        """Optional class-symmetric multi-centroid residual, not an attractor.

        Never requires objectness to activate, no absolute cosine cutoff or
        BDD label-specific geometry, no forced prototype diversity/repulsion.
        A zero-initialized residual gate protects the native classifier.
        """
        b, c, h, w = cls_feat.shape
        z = self.latent_projectors[k](cls_feat)
        z = z + self.spatial_smoothers[k](z)
        q = F.normalize(z.float().flatten(2).transpose(1, 2), dim=-1)
        bank = self.memory_banks[k]
        # normalized on every call, so no in-place optimizer-state mutation
        proto = F.normalize(bank.prototypes.float().flatten(0, 1), dim=-1)
        # Similarity scale is a declared temperature/model-capacity parameter,
        # not dataset-specific weather or absolute OOD threshold.
        # Stream prototype lookup in bounded query chunks. Chunk size follows
        # the declared SIMILARITY ELEMENT budget, not scene kind or class IDs.
        # This bounds inference temporaries; training autograd still retains
        # individual chunk graphs and requires separate memory profiling.
        budget = 1 << 20
        chunk = max(1, budget // max(1, b * proto.shape[0]))
        retrieved_chunks = []
        for first in range(0, q.shape[1], chunk):
            scores = torch.matmul(q[:,first:first+chunk], proto.t())
            attention = torch.softmax(scores * bank.temperature, dim=-1)
            retrieved_chunks.append(attention @ proto)
        retrieved = torch.cat(retrieved_chunks, dim=1)
        mem = retrieved.transpose(1, 2).reshape(b, self.latent_dim, h, w)
        mem = self.memory_proj_back[k](mem.to(cls_feat.dtype))
        mem = self.memory_norms[k](mem.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Normalized feature energy, not a fixed pixel-amplitude clip.
        scale = (cls_feat.float().square().mean(1, keepdim=True) + 1e-12).sqrt().detach()
        # No confidence threshold; zero gate protects from random prototypes.
        delta = torch.tanh(mem.float()) * scale
        result = cls_feat.float() + torch.tanh(self.mem_gate) * delta
        return result.to(cls_feat.dtype)

    def _get_losses_portable(self, imgs, x_shifts, y_shifts, expanded_strides,
                             labels, outputs, origin_preds, dtype):
        """Detection-only objective, with constant binary objectness meaning.

        This is a complete local implementation rather than hoping a legacy
        TDE_Head's epoch-10..50 curriculum happens to be patched correctly.
        Retains its OIoU, classification, and GT assignment APIs.
        """
        with torch.amp.autocast("cuda", enabled=False):
            bbox_preds, obj_preds, cls_preds = (outputs[:, :, :4].float(),
                                                 outputs[:, :, 4:5].float(),
                                                 outputs[:, :, 5:].float())
            # YOLOX padded labels: valid rows have positive width and height.
            # Class 0 is valid; never infer validity from class > 0.
            nlabel = ((labels[..., 3] > 0) & (labels[..., 4] > 0)).sum(1)
            n_anchors = outputs.shape[1]
            x_shifts = torch.cat(x_shifts, 1).float()
            y_shifts = torch.cat(y_shifts, 1).float()
            expanded_strides = torch.cat(expanded_strides, 1).float()
            if self.use_l1:
                origin_preds = torch.cat(origin_preds, 1).float()
            reg_all, cls_all, obj_all, masks, l1_all = [], [], [], [], []
            n_fg = 0.0
            n_gt_total = 0
            uncovered = 0
            for b in range(outputs.shape[0]):
                num_gt = int(nlabel[b].item())
                n_gt_total += num_gt
                obj_target = outputs.new_zeros((n_anchors, 1), dtype=torch.float32)
                fg = torch.zeros(n_anchors, dtype=torch.bool, device=outputs.device)
                reg_target = outputs.new_zeros((0,4), dtype=torch.float32)
                cls_target = outputs.new_zeros((0,self.num_classes), dtype=torch.float32)
                l1_target = outputs.new_zeros((0,4), dtype=torch.float32)
                if num_gt:
                    # Do not assume nonempty GT rows are contiguous. Some
                    # dataset adapters interleave ignored / padding rows.
                    gt_rows = labels[b][(labels[b,:,3] > 0) & (labels[b,:,4] > 0)]
                    gt = gt_rows[:,1:5].float()
                    classes = gt_rows[:,0].long()
                    if (classes < 0).any() or (classes >= self.num_classes).any():
                        raise ValueError("Class ID outside configured num_classes")
                    (matched_classes, fg, matched_iou, matched_ids,
                     n_fg_img) = self.get_assignments(
                        b, num_gt, gt, classes.float(), bbox_preds[b], expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds)
                    n_fg += n_fg_img
                    if n_fg_img:
                        obj_target[fg] = 1.0
                        reg_target = gt[matched_ids]
                        cls_target = F.one_hot(matched_classes.long(), self.num_classes).float()
                        cls_target = cls_target * matched_iou.float().unsqueeze(-1)
                        uncovered += num_gt - int(matched_ids.unique().numel())
                        if self.use_l1:
                            l1_target = self.get_l1_target(
                                outputs.new_zeros((n_fg_img,4), dtype=torch.float32),
                                gt[matched_ids], expanded_strides[0][fg],
                                x_shifts[0][fg], y_shifts[0][fg])
                    else:
                        uncovered += num_gt
                masks.append(fg)
                reg_all.append(reg_target)
                cls_all.append(cls_target)
                obj_all.append(obj_target)
                if self.use_l1:
                    l1_all.append(l1_target)
            fg = torch.cat(masks,0)
            reg_target = torch.cat(reg_all,0)
            cls_target = torch.cat(cls_all,0)
            obj_target = torch.cat(obj_all,0)
            denom = max(float(n_fg),1.0)
            iou_loss = self.iou_loss(bbox_preds.reshape(-1,4)[fg],reg_target).sum()/denom
            obj_loss = self.bcewithlog_loss(obj_preds.reshape(-1,1),obj_target).sum()/denom
            cls_loss = self.bcewithlog_loss(cls_preds.reshape(-1,self.num_classes)[fg],cls_target).sum()/denom
            if self.use_l1:
                l1_target = torch.cat(l1_all,0)
                l1_loss = self.l1_loss(origin_preds.reshape(-1,4)[fg],l1_target).sum()/denom
            else:
                l1_loss = iou_loss.new_zeros(())
            # Preserve vNext.2's established relative detector weights in the
            # baseline; changing it is a distinct ablation, not a hidden fix.
            weighted_iou = 2.5 * iou_loss
            total = weighted_iou + obj_loss + cls_loss + l1_loss
            self.last_assignment_stats = {
                "num_fg_total": int(n_fg), "num_gt_total": n_gt_total,
                "uncovered_gt": int(uncovered), "assignment_mode": "vnext3_portable"}
            return total, weighted_iou, obj_loss, cls_loss, l1_loss, n_fg/max(n_gt_total,1)

    @torch.no_grad()
    def get_assignments(self,batch_idx,num_gt,gt_bboxes_per_image,gt_classes,
                        bboxes_preds_per_image,expanded_strides,x_shifts,y_shifts,
                        cls_preds,obj_preds,mode="gpu"):
        """Memory-efficient SimOTA classification cost for dense UAV scenes.

        Derives exact one-hot BCE cost algebraically without materializing the
        [GT, candidate_anchor, num_classes] tensor from legacy TDE_Head.
        Keeps the user's existing geometry-interface and IoU cost balance.
        """
        original_device=gt_bboxes_per_image.device
        if mode=='cpu':
            gt_bboxes_per_image=gt_bboxes_per_image.float().cpu()
            gt_classes=gt_classes.cpu()
            bboxes_preds_per_image=bboxes_preds_per_image.float().cpu()
            expanded_strides=expanded_strides.float().cpu()
            x_shifts=x_shifts.cpu();y_shifts=y_shifts.cpu()
            cls_preds=cls_preds.cpu();obj_preds=obj_preds.cpu()
        fg_mask,geometry=self.get_geometry_constraint(gt_bboxes_per_image,
                expanded_strides,x_shifts,y_shifts)
        count=int(fg_mask.sum().item())
        if not count:
            return (gt_classes.new_zeros((0,),dtype=torch.long).to(original_device),
                    fg_mask.to(original_device),
                    gt_bboxes_per_image.new_zeros((0,)).to(original_device),
                    gt_classes.new_zeros((0,),dtype=torch.long).to(original_device),0)
        boxes=bboxes_preds_per_image[fg_mask]
        ious=bboxes_iou(gt_bboxes_per_image,boxes,False).float()
        with torch.amp.autocast('cuda',enabled=False):
            cls_prob=torch.sigmoid(cls_preds[batch_idx][fg_mask].float())
            obj_prob=torch.sigmoid(obj_preds[batch_idx][fg_mask].float())
            q=(cls_prob*obj_prob).clamp_min(0).sqrt()
            eps=torch.finfo(q.dtype).eps
            q=q.clamp(min=eps,max=1-eps)
            # BCE(one_hot(class)) = sum[-log(1-q_j)]
            #                        +log(1-q_class)-log(q_class)
            neg_base=-torch.log1p(-q).sum(-1)
            target_class=gt_classes.long()
            q_true=q.transpose(0,1).index_select(0,target_class)
            class_cost=(neg_base.unsqueeze(0)+torch.log1p(-q_true)-torch.log(q_true))
            iou_cost=-torch.log(ious.clamp_min(torch.finfo(ious.dtype).tiny))
            cost=(class_cost+3.0*iou_cost).masked_fill(~geometry,float('inf'))
        n,classes,matched_iou,ids=self.simota_matching(
            cost,ious,gt_classes,num_gt,fg_mask)
        return classes.to(original_device),fg_mask.to(original_device),            matched_iou.to(original_device),ids.to(original_device),n

    def simota_matching(self,cost,pair_wise_ious,gt_classes,num_gt,fg_mask):
        """Dynamic-K with eligible-only geometry and crowd conflict repair.

        Invalid geometry has infinite cost rather than a magic 1e6 penalty.
        A GT without any candidate remains uncovered and is recorded, rather
        than being assigned a geometrically invalid positive.
        """
        if cost.shape[1]==0:
            raise RuntimeError('No anchor candidates in SimOTA matching')
        candidate_budget=min(10,cost.shape[1])
        finite=torch.isfinite(cost)
        eligible_ious=pair_wise_ious.masked_fill(~finite,0)
        topk,_=torch.topk(eligible_ious,candidate_budget,dim=1)
        count=torch.clamp(topk.sum(1).int(),min=1,max=candidate_budget)
        matching=torch.zeros_like(cost,dtype=torch.bool)
        for gt in range(num_gt):
            valid=int(finite[gt].sum().item())
            if not valid:continue
            k=min(int(count[gt].item()),valid)
            ix=cost[gt].topk(k,largest=False).indices
            matching[gt,ix]=True
        conflicts=matching.sum(0)>1
        if conflicts.any():
            shared=conflicts.nonzero(as_tuple=True)[0]
            owner=cost[:,shared].argmin(0)
            matching[:,shared]=False
            matching[owner,shared]=True
        unused=matching.sum(0)==0
        for gt in (matching.sum(1)==0).nonzero(as_tuple=True)[0]:
            allowed=finite[gt] & unused
            if not allowed.any():continue
            ix=cost[gt].masked_fill(~allowed,float('inf')).argmin()
            matching[gt,ix]=True
            unused[ix]=False
        used=matching.any(0)
        fg_mask[fg_mask.clone()]=used
        matched=matching[:,used].to(torch.float32).argmax(0).long()
        classes=gt_classes[matched]
        selected_iou=(matching.float()*pair_wise_ious).sum(0)[used]
        return int(used.sum().item()),classes,selected_iou,matched

    def forward(self, xin, labels=None, imgs=None):
        """Explicit full-fidelity multi-scale head forward; no source anchors."""
        outputs=[]
        origin_preds=[]
        x_shifts=[]; y_shifts=[]; expanded_strides=[]
        for k, (cls_conv,reg_conv,stride,x) in enumerate(
                zip(self.cls_convs,self.reg_convs,self.strides,xin)):
            stem=self.stems[k](x)
            reg_feat=reg_conv(stem) + 0.20 * torch.tanh(self.gamma_reg) * stem
            reg_pred=self.reg_preds[k](reg_feat)
            obj_pred=self.obj_preds[k](reg_feat)
            cls_feat=self.spectral_adapters[k](cls_conv(stem))
            if self.enable_engram:
                cls_feat=self._classification_memory(k,cls_feat)
            cls_pred=self.cls_preds[k](cls_feat)
            if self.training:
                output=torch.cat((reg_pred,obj_pred,cls_pred),1)
                output,grid=self.get_output_and_grid(output,k,stride,xin[0].type())
                x_shifts.append(grid[:,:,0]); y_shifts.append(grid[:,:,1])
                expanded_strides.append(torch.full((1,grid.shape[1]),stride,device=x.device,dtype=x.dtype))
                if self.use_l1:
                    B,_,H,W=reg_pred.shape
                    p=reg_pred.view(B,1,4,H,W).permute(0,1,3,4,2).reshape(B,-1,4)
                    origin_preds.append(p.clone())
            else:
                output=torch.cat((reg_pred,obj_pred.sigmoid(),cls_pred.sigmoid()),1)
            outputs.append(output)
        if self.training:
            return self._get_losses_portable(imgs,x_shifts,y_shifts,expanded_strides,
                                             labels,torch.cat(outputs,1),origin_preds,
                                             dtype=xin[0].dtype)
        self.hw=[x.shape[-2:] for x in outputs]
        outputs=torch.cat([o.flatten(2) for o in outputs],dim=2).permute(0,2,1)
        return self.decode_outputs(outputs,dtype=xin[0].type()) if self.decode_in_inference else outputs
