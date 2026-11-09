"""A4.1 training: actual Mosaic, verified terminal policy and transitive provenance."""
import hashlib
import json
from pathlib import Path
from loguru import logger
from yolox.core.fourier_trainer import FourierTrainer
from yolox.core.a4_schedule import before_epoch_with_native_finish

# All known training-critical files; fail closed if the local tree lacks one.
A41_SOURCE_FILES = (
    'tools/train_fourier_a41.py',
    'exps/default/tde_yolox_s.py',
    'exps/default/tde_yolox_s_fourier.py',
    'exps/default/tde_yolox_s_fourier_a4.py',
    'exps/default/tde_yolox_s_fourier_a41.py',
    'exps/default/tde_yolox_s_fourier_a41_control.py',
    'exps/default/tde_yolox_s_fourier_a41_mosaic_anchor.py',
    'yolox/core/trainer.py',
    'yolox/core/fourier_trainer.py',
    'yolox/core/a4_schedule.py',
    'yolox/core/a41_trainer.py',
    'yolox/models/a41_backbone.py',
    'yolox/models/a41_spectral.py',
    'yolox/models/a41_head.py',
    'yolox/models/a4_detector.py',
    'yolox/models/a4_head.py',
    'yolox/models/a4_spectral.py',
    'yolox/models/fourier_detector.py',
    'yolox/models/fourier_invariance.py',
    'yolox/models/fourier_backbone.py',
    'yolox/models/fourier_head.py',
    'yolox/models/darknet.py',
    'yolox/models/ttt_modules.py',
    'yolox/models/tribrid_neck.py',
    'yolox/models/yolo_pafpn.py',
    'yolox/models/engram_head.py',
    'yolox/models/network_blocks.py',
    'yolox/models/losses.py',
    'yolox/data/datasets/coco.py',
    'yolox/data/datasets/mosaicdetection.py',
    'yolox/data/datasets/datasets_wrapper.py',
    'yolox/data/data_augment.py',
    'yolox/data/dataloading.py',
    'yolox/data/samplers.py',
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def a41_source_hashes(repo):
    repo = Path(repo)
    missing = [p for p in A41_SOURCE_FILES if not (repo / p).is_file()]
    if missing:
        raise RuntimeError('Missing training source files (no silent manifest gaps): '+str(missing))
    return {rel: sha256(repo / rel) for rel in A41_SOURCE_FILES}


def build_a41_provenance(repo, exp, args):
    source_hashes = a41_source_hashes(repo)
    for ann in (exp.train_ann, exp.val_ann):
        if not Path(ann).is_file():
            raise RuntimeError('Missing training/validation annotation file '+str(ann))
    return {
        'schema': 'tdevnext3_a41_provenance_v1',
        'architecture': 'A4.1 actual_mosaic_live_gradients',
        'basis': exp.a4_filter_basis,
        'experiment': exp.exp_name,
        'seed': exp.seed,
        'batch_size': int(args.batch_size),
        'max_epoch': int(exp.max_epoch),
        'no_aug_epochs': int(exp.no_aug_epochs),
        'proto_weight': float(exp.a41_proto_weight),
        'mosaic_prob': float(exp.mosaic_prob),
        'mixup_prob': float(exp.mixup_prob),
        'source_sha256': source_hashes,
        'train_ann_sha256': sha256(exp.train_ann),
        'val_ann_sha256': sha256(exp.val_ann),
        'dataset_preflight': exp._preflight(),
        'train_from_checkpoint': str(args.ckpt),
        'architecture_controls': {
            'dynamic_input': 'stride32_rectangular_canvas',
            'native_p3_route': 'direct_plus_bounded_spi_residual',
            'context': 'bounded_nonzero_gate_per_fpn_stage',
            'fourier': 'local_depthwise_dct_or_operator_matched_control',
            'memory_aux': ('A4_1.0_CE_detached_query' if exp.fourier_variant=='a41_mosaic_a4'
                           else 'A4.1_0.05_CE_detach_before_query'),
            'TTT': False,
            'L1_terminal': True,
            'mosaic_wrapper': 'MosaicDetection',
            'mosaic_preprocess_passes': 1,
        },
    }


def verify_a41_provenance(previous, current):
    # Ignore initial ckpt path and live dataset sampling; source/data and
    # optimization policy must match exactly on resume.
    fields = ('schema', 'architecture', 'basis', 'experiment', 'seed',
              'batch_size', 'max_epoch', 'no_aug_epochs', 'proto_weight',
              'mosaic_prob', 'mixup_prob', 'source_sha256',
              'train_ann_sha256', 'val_ann_sha256', 'dataset_preflight',
              'architecture_controls')
    differences = [k for k in fields if previous.get(k) != current.get(k)]
    if differences:
        raise RuntimeError('Refusing A4.1 resume after provenance drift: '+str(differences))


class A41Trainer(FourierTrainer):
    def before_train(self):
        super().before_train()
        from yolox.data import MosaicDetection
        dataset = getattr(self.train_loader, 'dataset', None)
        if not isinstance(dataset, MosaicDetection):
            raise RuntimeError('A4.1 must train with a REAL MosaicDetection wrapper')
        inner = getattr(dataset, '_dataset', None)
        if inner is None or getattr(inner, 'preproc', 'missing') is not None:
            raise RuntimeError('Training pipeline double/preprocessing mismatch')
        if getattr(dataset, 'preproc', None) is None:
            raise RuntimeError('MosaicDetection must own TrainTransform')
        sampler = getattr(self.train_loader, 'batch_sampler', None)
        if sampler is None or not hasattr(sampler, 'mosaic'):
            raise RuntimeError('Mosaic sampling cannot be controlled by native trainer')
        if bool(sampler.mosaic) == bool(self.no_aug):
            raise RuntimeError('Mosaic sampler flag inconsistent with no_aug state')
        logger.info('A41_DATA_CERTIFIED wrapper=MosaicDetection raw_preproc=None '
                    'wrapper_preproc={} mosaic_prob={} sampler_mosaic={} max_labels={}',
                    type(dataset.preproc).__name__, self.exp.mosaic_prob,
                    sampler.mosaic, self.exp._capacity)
        if self.rank != 0:
            return
        repo = Path(__file__).resolve().parents[2]
        provenance = build_a41_provenance(repo, self.exp, self.args)
        dest = Path(self.file_name) / 'a41_provenance.json'
        if self.args.resume:
            if not dest.is_file():
                raise RuntimeError('Cannot resume A4.1 without original manifest')
            verify_a41_provenance(json.loads(dest.read_text(encoding='utf-8')),
                                  provenance)
        else:
            if dest.exists():
                raise RuntimeError('Cannot overwrite existing A4.1 provenance')
            dest.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        logger.info('A41_TRANSITIVE_PROVENANCE_PASS count={} path={}',
                    len(provenance['source_sha256']), dest)

    def before_epoch(self):
        before_epoch_with_native_finish(self)
        sampler = getattr(self.train_loader, 'batch_sampler', None)
        if sampler is None or bool(sampler.mosaic) == bool(self.no_aug):
            raise RuntimeError('Actual sampler Mosaic flag differs from terminal schedule')
        if self.no_aug and not bool(self.model.module.head.use_l1 if hasattr(self.model, 'module')
                                    else self.model.head.use_l1):
            raise RuntimeError('L1 must activate in the no-augmentation phase')
