"""A4 task-grounded Fourier + Engram head.

Foreground class grounding uses detached query embeddings. Thus auxiliary
classification gradients train prototype parameters ONLY, never the shared
backbone, head stems, or Fourier adapter. The native detector loss remains
separately observable. Queries retain task-driven gradients via detection CE.
"""
import math
import torch
import torch.nn.functional as F
from .fourier_head import FourierTDEHead
from .a4_spectral import A4SpectralResidual


class A4Head(FourierTDEHead):
    def __init__(self, num_classes, width=0.5, in_channels=(256,512,1024),
                 act='silu', loss_type='oiou', basis='dct'):
        super().__init__(num_classes, width=width, in_channels=in_channels,
                         act=act, variant='spectral_memory', loss_type=loss_type)
        c = int(256 * width)
        self.spectral_adapters = torch.nn.ModuleList([
            A4SpectralResidual(c, basis=basis) for _ in self.cls_convs])
        self.fourier_variant = 'a4' if basis == 'dct' else 'a4_control'
        self.memory_enabled = True
        self.memory_alignment_enabled = True
        self.aux_memory_loss = None
        self.memory_probes = []
        self.last_coverage_by_size = {}
        self._queries = []
        self._matches = []
        self._input_hw = (0, 0)
        self.last_mem_residual = None
        self.capture_memory = False
        self.memory_probes = []

    def forward(self, xin, labels=None, imgs=None):
        self._queries = []
        self._matches = []
        self.aux_memory_loss = None
        self.memory_probes = []
        self.last_coverage_by_size = {}
        if imgs is not None:
            self._input_hw = tuple(int(v) for v in imgs.shape[-2:])
        return super().forward(xin, labels, imgs)

    def _classification_memory(self, k, cls_feat):
        if not self.memory_enabled and not self.training:
            return cls_feat
        b, _, h, w = cls_feat.shape
        z = self.latent_projectors[k](cls_feat)
        z = z + self.spatial_smoothers[k](z)
        q = F.normalize(z.float().flatten(2).transpose(1,2), dim=-1)
        if self.training:
            # Detach here: auxiliary branch cannot backprop to z/neck/backbone.
            self._queries.append(q.detach())
        if not self.memory_enabled:
            return cls_feat
        bank = self.memory_banks[k]
        proto = F.normalize(bank.prototypes.float().flatten(0,1), dim=-1)
        budget = 1 << 20
        chunk = max(1, budget // max(1, b*proto.shape[0]))
        retrieval = []
        for offset in range(0, q.shape[1], chunk):
            sim = q[:,offset:offset+chunk] @ proto.T
            attn = torch.softmax(sim * bank.temperature, dim=-1)
            retrieval.append(attn @ proto)
        mem = torch.cat(retrieval, dim=1).transpose(1,2).reshape(b,self.latent_dim,h,w)
        mem = self.memory_proj_back[k](mem.to(cls_feat.dtype))
        mem = self.memory_norms[k](mem.permute(0,2,3,1)).permute(0,3,1,2)
        scale = (cls_feat.float().square().mean(1,keepdim=True)+1e-12).sqrt().detach()
        delta = torch.tanh(mem.float())*scale
        delta = torch.tanh(self.mem_gate)*delta
        if self.capture_memory:
            self.memory_probes.append((cls_feat.detach(),delta.detach()))
        self.last_mem_residual = delta.detach() if self.capture_memory else None
        return (cls_feat.float()+delta).to(cls_feat.dtype)

    @torch.no_grad()
    def get_assignments(self, batch_idx,num_gt,gt_bboxes_per_image,gt_classes,
                        bboxes_preds_per_image,expanded_strides,x_shifts,y_shifts,
                        cls_preds,obj_preds,mode='gpu'):
        result = super().get_assignments(
            batch_idx,num_gt,gt_bboxes_per_image,gt_classes,bboxes_preds_per_image,
            expanded_strides,x_shifts,y_shifts,cls_preds,obj_preds,mode=mode)
        matched_classes, fg, _, gt_ids, nfg = result
        self._matches.append((int(batch_idx), fg.detach().clone(),
                              matched_classes.long().detach().clone()))
        h,w=self._input_hw
        if h>0 and w>0:
            a = (gt_bboxes_per_image[:,2].float()*gt_bboxes_per_image[:,3].float())/(h*w)
            box_area = a.clamp_min(0)
            covered = torch.zeros(num_gt, dtype=torch.bool, device=box_area.device)
            if gt_ids.numel(): covered[gt_ids.long()] = True
            # Native image-relative area categories; separate from COCO
            # original-image small/medium/large bins.
            groups = {'tiny_relative': box_area < (16/640)**2,
                      'middle_relative': (box_area >= (16/640)**2) & (box_area < (96/640)**2),
                      'large_relative': box_area >= (96/640)**2}
            for name, mask in groups.items():
                rec=self.last_coverage_by_size.setdefault(name,[0,0])
                rec[0]+=int(mask.sum()); rec[1]+=int((mask & covered).sum())
            self.last_coverage_by_size.setdefault('overall',[0,0])
            self.last_coverage_by_size['overall'][0]+=num_gt
            self.last_coverage_by_size['overall'][1]+=int(covered.sum())
        return result

    def _foreground_proto_loss(self):
        if not self.memory_alignment_enabled or not self._matches:
            return None
        losses=[]
        starts=[0]
        for q in self._queries: starts.append(starts[-1]+q.shape[1])
        for b,mask,classes in self._matches:
            ids=mask.nonzero(as_tuple=True)[0]
            if ids.numel()==0: continue
            for lvl,q in enumerate(self._queries):
                selection=(ids>=starts[lvl]) & (ids<starts[lvl+1])
                if not selection.any():continue
                queries=q[b,ids[selection]-starts[lvl]]
                labels=classes[selection]
                bank=self.memory_banks[lvl]
                proto=F.normalize(bank.prototypes.float(),dim=-1)
                # Class-marginalized multi-centroid logits, symmetric in class ID.
                sim=torch.einsum('pd,kmd->pkm',queries.float(),proto)*bank.temperature
                logits=torch.logsumexp(sim,dim=-1)-math.log(proto.shape[1])
                per_obj=F.cross_entropy(logits,labels,reduction='none')
                # Average classes PRESENT in this minibatch equally, not a
                # handwritten BDD/ExDark class prior.
                for class_id in labels.unique():
                    losses.append(per_obj[labels==class_id].mean())
        return torch.stack(losses).mean() if losses else None

    def _get_losses_portable(self,imgs,x_shifts,y_shifts,expanded_strides,
                             labels,outputs,origin_preds,dtype):
        det = super()._get_losses_portable(imgs,x_shifts,y_shifts,expanded_strides,
                                           labels,outputs,origin_preds,dtype)
        proto_loss=self._foreground_proto_loss()
        if proto_loss is None: proto_loss=det[0].new_zeros(())
        self.aux_memory_loss=proto_loss
        self.last_assignment_stats['grounded_memory_pos'] = sum(
            int(m[1].sum()) for m in self._matches)
        return (det[0]+proto_loss, *det[1:])
