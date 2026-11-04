"""Shared helpers for non-destructive vNext.3 PG-SMA checkout audits.

Keep this dependency-light so metadata/checkpoint audits work on Windows.
User-controlled checkpoints only: torch.load uses PyTorch's pickle semantics.
"""
from __future__ import annotations
import collections
import json
import math
import os
from pathlib import Path
import sys


def dump_json(payload, output=None):
    txt = json.dumps(payload, indent=2, default=str, allow_nan=False)
    print(txt)
    if output:
        dst = Path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(txt + "\n", encoding="utf-8")


def load_torch_checkpoint(path):
    import torch
    # Only load checkpoints from the user's own trusted files; this is not a
    # generic untrusted pickle reader. Prefer weights_only when supported.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Safe checkpoint load failed for {path!r}. Only use checkpoints you trust; "
            "do not silently retry with weights_only=False. " + str(exc)
        ) from exc
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(ckpt)!r}")
    return ckpt


def unwrap_state_dict(ckpt, prefer_ema=False):
    candidates = ("ema_model", "model") if prefer_ema else ("model", "ema_model")
    for key in candidates:
        state = ckpt.get(key)
        if isinstance(state, dict) and state:
            return state, key
    if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt, "root"
    raise ValueError("No model/ema_model state_dict found in checkpoint")


def normalize_state_keys(state):
    # Strip common wrappers only when this does not produce collisions.
    new = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue
        while k.startswith(("module.", "_orig_mod.")):
            if k.startswith("module."):
                k = k[len("module."):]
            else:
                k = k[len("_orig_mod."):]
        if k in new:
            raise ValueError(f"State key collision after removing common wrappers: {k}")
        new[k] = v
    return new


def checkpoint_coverage(model, state):
    expected = model.state_dict()
    matched = {k:v for k,v in state.items() if k in expected and tuple(getattr(v,"shape",())) == tuple(expected[k].shape)}
    unexpected = [k for k in state if k not in expected]
    incompatible = [k for k,v in state.items() if k in expected and tuple(getattr(v,"shape",())) != tuple(expected[k].shape)]
    missing = [k for k in expected if k not in matched]
    groups = {}
    for prefix in ("backbone.backbone.stem.","backbone.backbone.dark2.",
                   "backbone.backbone.dark3.","backbone.backbone.dark4.",
                   "backbone.backbone.dark5.","backbone.","head.stems.",
                   "head.cls_convs.","head.reg_convs.","head.cls_preds.",
                   "head.reg_preds.","head.obj_preds.","head.pgsma_engine."):
        subset = {k:v for k,v in expected.items() if k.startswith(prefix)}
        good = {k:v for k,v in subset.items() if k in matched}
        n_all = sum(v.numel() for v in subset.values())
        n_good = sum(v.numel() for v in good.values())
        groups[prefix.rstrip('.')] = {
            "tensors":len(good),"expected_tensors":len(subset),
            "elements":n_good,"expected_elements":n_all,
            "element_ratio":round(n_good/max(n_all,1),6) if subset else None
        }
    return {
        "matched":matched,"missing_keys":missing,"shape_mismatches":incompatible,
        "unexpected_keys":unexpected,"groups":groups,
        "matched_parameter_elements":sum(v.numel() for v in matched.values())
    }


def image_gt_sizes(labels):
    """Return per-image valid GT and size diagnostics, labels [B,L,5] cxcywh."""
    import torch
    result=[]
    for y in labels:
        valid=torch.isfinite(y).all(-1)&(y[:,3]>0)&(y[:,4]>0)
        boxes=y[valid,1:5]
        result.append({"count":int(boxes.shape[0]),
                       "min_width":float(boxes[:,2].min()) if len(boxes) else None,
                       "min_height":float(boxes[:,3].min()) if len(boxes) else None,
                       "under_4px":int(((boxes[:,2]<4)|(boxes[:,3]<4)).sum()) if len(boxes) else 0})
    return result


def one_real_train_batch(exp, batch_size=1, mosaic=False):
    """Get a single batch from the user's actual YOLOX data path; no synthetic data."""
    exp.data_num_workers=0  # deterministic, spawn-free Windows preflight
    loader=exp.get_data_loader(batch_size=int(batch_size), is_distributed=False,
                               no_aug=not mosaic, cache_img=False)
    batch=next(iter(loader))
    if not isinstance(batch,(list,tuple)) or len(batch)<2:
        raise TypeError("YOLOX training loader must return (images, labels, ...)")
    images,labels=batch[:2]
    if not hasattr(images,"ndim") or images.ndim!=4:
        raise ValueError(f"Expected [B,3,H,W] training images, got {getattr(images,'shape',None)}")
    if not hasattr(labels,"ndim") or labels.ndim!=3 or labels.shape[-1]!=5:
        raise ValueError(f"Expected padded [B,L,5] labels, got {getattr(labels,'shape',None)}")
    if images.shape[0]!=labels.shape[0]:
        raise ValueError("Batch/label count mismatch")
    return images,labels,batch,loader


def load_exp():
    from exps.default.tde_yolox_s_pgsma import Exp
    return Exp()


def class_index_stats(classes, nclass):
    import torch
    valid=classes[torch.isfinite(classes)]
    if len(valid)==0: return {"min":None,"max":None,"invalid":0}
    return {"min":float(valid.min()),"max":float(valid.max()),
            "invalid":int(((valid<0)|(valid>=nclass)|(valid!=valid.round())).sum())}


def summarize_grads(model, loss):
    import torch
    names=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    raw=torch.autograd.grad(loss,[p for _,p in names],allow_unused=True,retain_graph=True)
    return {"native_nonzero":sum(g is not None and bool((g.detach()!=0).any())
                                 for (n,_),g in zip(names,raw) if 'pgsma_engine.' not in n),
            "semantic_nonzero":sum(g is not None and bool((g.detach()!=0).any())
                                   for (n,_),g in zip(names,raw) if 'pgsma_engine.' in n)}


def fail_or_warn(errors, warnings, text, hard=True):
    (errors if hard else warnings).append(text)
