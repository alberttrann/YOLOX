"""Shared, testable no-augmentation/L1 terminal schedule for ALL long-run variants.

Unlike vNext2's weather/TTT curriculum, this only changes ordinary YOLOX
augmentation and regression refinement over the declared no_aug_epochs.
"""
import gc
import torch
from loguru import logger
from yolox.data import DataPrefetcher


def before_epoch_with_native_finish(trainer):
    current=trainer.epoch+1
    model=trainer.model.module if hasattr(trainer.model,'module') else trainer.model
    model.set_meta_training_state(current,trainer.max_epoch)
    close_start=trainer.max_epoch-int(trainer.exp.no_aug_epochs)+1
    if current>=close_start:
        if not trainer.no_aug:
            if not hasattr(trainer.train_loader,'close_mosaic'):
                raise RuntimeError('Train loader cannot close Mosaic; terminal schedule not certified')
            trainer.train_loader.close_mosaic()
            trainer.no_aug=True
            trainer.save_ckpt(ckpt_name='last_mosaic_epoch')
            logger.info('NATIVE_FINISH: mosaic disabled at epoch={}',current)
        model.head.use_l1=True
    else:
        model.head.use_l1=False
    logger.info('NATIVE_FINISH epoch={} max_epoch={} no_aug={} use_l1={}',
                current,trainer.max_epoch,trainer.no_aug,model.head.use_l1)
    if hasattr(trainer,'prefetcher'):del trainer.prefetcher
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    trainer.prefetcher=DataPrefetcher(trainer.train_loader)
