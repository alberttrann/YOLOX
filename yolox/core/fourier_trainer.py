"""tdevNext.3 trainer: preserves the stable vNext2 optimizer/AMP/EMA pipeline.

Removes the legacy weather/TTT-specific epoch phase choreography from
before_epoch(). Other original optimization machinery is deliberately retained
until a separate controlled optimizer ablation demonstrates improvement.
"""
import gc
import hashlib
import json
import torch
from pathlib import Path
from loguru import logger
from yolox.core.trainer import Trainer
from yolox.data import DataPrefetcher


class FourierTrainer(Trainer):

    def resume_train(self, model):
        """Correctly warm-start native YOLOX dark2 INCLUDING BatchNorm buffers.

        The legacy vNext.2 loader failed to remap .bn.* keys to
        dark2.backbone_stage.*; these are essential pretrained statistics.
        Same-variant true --resume still uses strict parent checkpoint + optimizer.
        """
        if self.args.resume:
            return super().resume_train(model)
        self.start_epoch=0
        if self.args.ckpt is None:
            logger.warning('Training from random initialization; no checkpoint provided')
            return model
        ckpt_path=Path(self.args.ckpt)
        saved=torch.load(str(ckpt_path),map_location='cpu',weights_only=False)
        source=saved.get('model',saved) if isinstance(saved,dict) else saved
        if not isinstance(source,dict):
            raise TypeError('Checkpoint does not contain a valid model state dictionary')
        source={k.removeprefix('module.'):v for k,v in source.items()}
        own=model.state_dict()
        selected={}
        for k,v in source.items():
            names=[k]
            prefix='backbone.backbone.dark2.'
            if k.startswith(prefix) and '.backbone_stage.' not in k:
                names.append(k.replace(prefix,prefix+'backbone_stage.',1))
            for new_key in names:
                if new_key in own and hasattr(v,'shape') and own[new_key].shape==v.shape:
                    selected[new_key]=v
        if not selected:
            raise RuntimeError('No compatible checkpoint keys. Refusing silent random warm-start')
        stem=sum(k.startswith('backbone.backbone.stem.') for k in selected)
        dark2=sum(k.startswith('backbone.backbone.dark2.backbone_stage.') for k in selected)
        if stem==0 or dark2==0:
            raise RuntimeError('Pretrained stem/dark2 weights absent. Do not silently '
                               'discard pretrained spatial competence')
        # No optimizer state is restored for a NEW Fourier architecture.
        model.load_state_dict({**own,**selected},strict=True)
        count=sum(int(own[k].numel()) for k in selected if own[k].is_floating_point())
        total=sum(int(v.numel()) for v in own.values() if v.is_floating_point())
        logger.info('FOURIER_PRETRAIN_LOADED matched={} float_elements={}/{} '
                    'ratio={:.4f} stem_keys={} dark2_keys={} (BN included)',
                    len(selected),count,total,count/max(total,1),stem,dark2)
        return model

    def before_train(self):
        super().before_train()
        if self.rank==0:
            root=Path(__file__).resolve().parents[2]
            selected=[
                'yolox/models/fourier_head.py','yolox/models/fourier_backbone.py',
                'yolox/models/fourier_detector.py','yolox/models/fourier_invariance.py',
                'yolox/models/engram_head.py','yolox/models/ttt_modules.py',
                'yolox/models/tribrid_neck.py','yolox/models/yolo_pafpn.py',
                'exps/default/tde_yolox_s.py',
            ]
            hashes={}
            for rel in selected:
                path=root/rel
                hashes[rel]=(hashlib.sha256(path.read_bytes()).hexdigest()
                             if path.is_file() else None)
            model=self.model.module if hasattr(self.model,'module') else self.model
            manifest={
                'architecture':'tdevnext3_fourier_full_fidelity_1.0',
                'variant':getattr(self.exp,'fourier_variant',None),
                'class_count':model.num_classes,
                'dataset_preflight':self.exp._preflight(),
                'native_inference_ttt':False,
                'pretrained_source':self.args.ckpt,
                'source_sha256':hashes,
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            filename=Path(self.file_name)/'fourier_architecture_manifest.json'
            filename.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            logger.info('Wrote immutable architecture / dataset provenance: {}',filename)

    def before_epoch(self):
        model_ref=self.model.module if hasattr(self.model,'module') else self.model
        if hasattr(model_ref,'set_meta_training_state'):
            model_ref.set_meta_training_state(self.epoch+1,self.max_epoch)
        logger.info('---> start Fourier-invariant epoch {} (no TTT / no objectness target switch)',self.epoch+1)
        # No hardcoded epoch-55..70 mosaic/L1/EMA changes. All experiments have
        # one consistent forward/loss policy from the first step to the last.
        if hasattr(self,'prefetcher'):
            del self.prefetcher
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        self.prefetcher=DataPrefetcher(self.train_loader)

    def evaluate_and_save_model(self):
        # The detector is strictly feed-forward at inference. Its epoch state is
        # bookkeeping only; no forced TTT and no different objective in eval.
        return super().evaluate_and_save_model()
