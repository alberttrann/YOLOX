"""Reproducible A0/A1/A3 research suite; stop at *complete epoch boundaries*.

This wrapper does not edit the existing FourierTrainer, model, experiments,
optimizer, scheduler, evaluation, or checkpoints. The 80-epoch schedule remains
unchanged.  A2's checkpoint and original launcher are never touched.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.backends.cudnn as cudnn
from loguru import logger
import tools.train as native_train
from yolox.exp import get_exp
from yolox.core.launch import launch
from yolox.core.fourier_trainer import FourierTrainer

VARIANT_TO_EXP = {
    'off': 'tdevnext3_a0_off',
    'spatial': 'tdevnext3_a1_spatial',
    'spectral': 'tdevnext3_a2_spectral',
    'spectral_memory': 'tdevnext3_a3_spectral_memory',
}
REFERENCE_PATH = REPO / 'YOLOX_outputs' / 'tdevnext3_a2_spectral' / 'fourier_architecture_manifest.json'


def selected_epoch_indices(start_epoch: int, max_epoch: int, completed_epoch_limit: int):
    """Keep scheduler's original horizon; limit only completed epoch count."""
    if completed_epoch_limit < 0:
        raise ValueError('completed_epoch_limit must be nonnegative')
    stop = max_epoch if completed_epoch_limit == 0 else min(max_epoch, completed_epoch_limit)
    return range(start_epoch, stop)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_reference_a2(exp, args, *, ref_path=REFERENCE_PATH):
    """Refuse a silent code/geometry drift relative to existing A2 results."""
    if not ref_path.is_file():
        if not args.allow_missing_a2_reference:
            raise RuntimeError(
                'A2 architecture manifest not found at ' + str(ref_path) + '. '
                'Provide --allow-missing-a2-reference ONLY for a standalone suite.'
            )
        logger.warning('No A2 manifest: cross-run equivalence CANNOT be certified')
        return None
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    if ref.get('variant') != 'spectral':
        raise RuntimeError('A2 reference manifest has unexpected variant: ' + str(ref.get('variant')))
    for rel, digest in ref.get('source_sha256', {}).items():
        current = REPO / rel
        if digest is not None and (not current.is_file() or sha256_file(current) != digest):
            raise RuntimeError('Source changed since A2 training: ' + rel + '. '
                               'Do not compare drifted model definitions as the same experiment.')
    here = exp._preflight()
    if ref.get('dataset_preflight') != here:
        raise RuntimeError('Dataset preflight differs from A2: ' +
                           json.dumps({'a2': ref.get('dataset_preflight'), 'current': here}))
    if ref.get('class_count') != exp.num_classes:
        raise RuntimeError('Class-count mismatch with A2')
    if int(args.batch_size) != 8:
        raise RuntimeError('Existing A2 was trained at global batch 8; use -b 8 for matched suite')
    if not args.fp16:
        raise RuntimeError('Existing A2 used --fp16; match it for valid suite comparisons')
    if int(exp.max_epoch) != 80:
        raise RuntimeError('Existing A2 used 80-epoch LR horizon; do not change max_epoch')
    if int(exp.seed) != 0:
        raise RuntimeError('Existing A2 used seed=0')
    if getattr(exp, 'ema', None) is not True:
        raise RuntimeError('Existing A2 used EMA checkpoints')
    logger.info('A2_REFERENCE_CHECK PASSED code hashes, dataset geometry, classes, seed, batch, EMA')
    return ref


def check_output_safety(exp, args):
    exp_dir = REPO / exp.output_dir / args.experiment_name
    active_checkpoints = sorted(exp_dir.glob('*_ckpt.pth'))
    if args.resume:
        manifest = exp_dir / 'fourier_architecture_manifest.json'
        if not manifest.is_file():
            raise RuntimeError('Refusing resume without matching variant manifest: '+str(manifest))
        recorded = json.loads(manifest.read_text(encoding='utf-8'))
        if recorded.get('variant') != exp.fourier_variant:
            raise RuntimeError('Resume variant does not match output provenance')
        if args.ckpt is not None and not (Path(args.ckpt) if Path(args.ckpt).is_absolute() else REPO / args.ckpt).is_file():
            raise RuntimeError('Resume checkpoint not found: ' + str(args.ckpt))
    elif active_checkpoints:
        raise RuntimeError('Fresh training would overwrite existing checkpoints in '+str(exp_dir)
                           + '. Use --resume ONLY with a same-variant training checkpoint.')


class SuiteFourierTrainer(FourierTrainer):
    def train_in_epoch(self):
        cap = int(self.args.suite_stop_after_epoch)
        if cap == 0:
            return super().train_in_epoch()
        for self.epoch in selected_epoch_indices(self.start_epoch, self.max_epoch, cap):
            self.before_epoch()
            self.train_in_iter()
            # Native checkpoint save + complete clean evaluation + history save.
            self.after_epoch()
        logger.info('FOURIER_SUITE_STOP completed_epoch_limit={} '
                    'lr_horizon_max_epoch={} (optimizer/scheduler unchanged)',cap,self.max_epoch)

    def before_train(self):
        super().before_train()
        if self.rank == 0:
            p = Path(self.file_name) / 'suite_protocol.json'
            protocol = {
                'variant': self.exp.fourier_variant,
                'experiment': self.args.experiment_name,
                'comparison': 'A0/A1/A2/A3 using 80-epoch horizon; evaluation at epochs 1 and 2',
                'stop_after_completed_epoch': self.args.suite_stop_after_epoch,
                'max_epoch_lr_horizon': self.exp.max_epoch,
                'batch_size': self.args.batch_size,
                'fp16': self.args.fp16,
                'seed': self.exp.seed,
                'pretrained_checkpoint': self.args.ckpt,
                'pretrained_sha256': (sha256_file((Path(self.args.ckpt) if Path(self.args.ckpt).is_absolute() else REPO / self.args.ckpt)) if
                                      self.args.ckpt and (Path(self.args.ckpt) if Path(self.args.ckpt).is_absolute() else REPO / self.args.ckpt).is_file() else None),
                'source_manifest': 'fourier_architecture_manifest.json',
                'eval_model': 'EMA',
                'fourier_auxiliary_loss': False,
                'ttt_enabled': False,
                'dataset_preflight': self.exp._preflight(),
            }
            p.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
            logger.info('FOURIER_SUITE_PROTOCOL {}', p)


def worker(exp, args):
    if getattr(exp, 'seed', None) is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
    from yolox.utils import configure_nccl, configure_omp
    configure_nccl()
    configure_omp()
    # This matches the ORIGINAL A2 launcher; changing benchmark policy now
    # would introduce a different experiment rather than preserve A2 parity.
    cudnn.benchmark = True
    logger.info('Fourier suite trainer selected: SuiteFourierTrainer')
    SuiteFourierTrainer(exp, args).train()


def main(argv=None):
    parser = native_train.make_parser()
    parser.add_argument('--suite-stop-after-epoch', type=int, default=2,
                        help='Stop after evaluation/checkpoint for completed epoch N; '
                             '0 means full run. Does not alter max_epoch or LR.')
    parser.add_argument('--allow-missing-a2-reference', action='store_true',
                        help='Opt out only when running a standalone new suite')
    args = parser.parse_args(argv)
    if not args.exp_file:
        parser.error('Provide -f exps/default/tde_yolox_s_fourier_<variant>.py')
    exp = get_exp(args.exp_file, args.name)
    if args.opts:
        exp.merge(args.opts)
    if args.experiment_name is None:
        args.experiment_name = exp.exp_name
    if exp.fourier_variant not in VARIANT_TO_EXP:
        raise RuntimeError('Unknown Fourier suite variant')
    if args.experiment_name != VARIANT_TO_EXP[exp.fourier_variant]:
        raise RuntimeError('Refusing experiment/output-directory mismatch')
    if exp.fourier_variant == 'spectral':
        raise RuntimeError('A2 already has a reference checkpoint. Do not overwrite it '
                           'using the comparison runner; use the original launcher only for same-variant resume.')
    if args.suite_stop_after_epoch < 0 or args.suite_stop_after_epoch > exp.max_epoch:
        raise RuntimeError('Invalid suite epoch boundary')
    n_gpus = torch.cuda.device_count() if args.devices is None else int(args.devices)
    if n_gpus != 1 or int(args.num_machines) != 1:
        raise RuntimeError('The suite is validated for one GPU; pass -d 1')
    check_reference_a2(exp, args)
    check_output_safety(exp, args)
    logger.info('FOURIER_SUITE_VARIANT={} LR_HORIZON={} STOP_AFTER_EPOCH={}',
                exp.fourier_variant, exp.max_epoch, args.suite_stop_after_epoch)
    launch(worker,n_gpus,num_machines=args.num_machines,machine_rank=args.machine_rank,
           dist_url=args.dist_url,backend=args.dist_backend,args=(exp,args))


if __name__ == '__main__':
    main()
