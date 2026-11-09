"""A4 trainer: retains stable native SGD/AMP/EMA, restores terminal no-aug/L1.

Provenance records actual source files; logs effective A4 policy. No legacy
TTT, objectness target transitions, or weather-conditioned hyperparameters.
"""
import gc
import hashlib
import json
from pathlib import Path
import torch
from loguru import logger
from yolox.core.fourier_trainer import FourierTrainer
from yolox.data import DataPrefetcher


class A4Trainer(FourierTrainer):
    def before_train(self):
        super().before_train()
        if self.rank != 0: return
        root=Path(__file__).resolve().parents[2]
        paths=['yolox/models/a4_head.py','yolox/models/a4_spectral.py',
               'yolox/models/a4_detector.py','yolox/models/a4_backbone.py',
               'yolox/core/a4_trainer.py','yolox/models/fourier_head.py',
               'yolox/models/engram_head.py','yolox/models/tribrid_neck.py',
               'yolox/models/ttt_modules.py','exps/default/tde_yolox_s_fourier_a4.py']
        hashes={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in paths}
        manifest=Path(self.file_name)/'a4_provenance.json'
        data={'architecture':'tdevnext3_A4','filter_basis':self.exp.a4_filter_basis,
              'source_sha256':hashes,'dataset_preflight':self.exp._preflight(),
              'seed':self.exp.seed,'batch_size':self.args.batch_size,
              'max_epoch_lr_horizon':self.max_epoch,
              'no_aug_epochs':self.exp.no_aug_epochs,'l1_policy':'on_when_no_aug',
              'memory_grounding':'foreground_matched_prototypes_query_detached',
              'weight_source':self.args.ckpt,'ema':self.use_model_ema,
              'disabled':['TTT','reconstruction','affordance','prototype_repulsion']}
        manifest.write_text(json.dumps(data,indent=2),encoding='utf-8')
        logger.info('A4 manifest: {}',manifest)

    def before_epoch(self):
        from yolox.core.a4_schedule import before_epoch_with_native_finish
        return before_epoch_with_native_finish(self)
