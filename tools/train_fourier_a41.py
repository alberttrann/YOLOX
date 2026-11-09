"""Dedicated isolated A4.1 Windows single-GPU launcher; never changes native train."""
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
import tools.train as native_train
from yolox.exp import get_exp
from yolox.core.launch import launch
from yolox.core.a41_trainer import A41Trainer, build_a41_provenance, verify_a41_provenance


class LimitedA41Trainer(A41Trainer):
    def train_in_epoch(self):
        limit = int(self.args.stop_after_epoch)
        if not limit:
            return super().train_in_epoch()
        if limit < 0 or limit > self.max_epoch:
            raise ValueError('Invalid A4.1 completed epoch limit')
        for self.epoch in range(self.start_epoch, min(limit, self.max_epoch)):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
        logger.info('A41_STOP_AFTER_COMPLETE_EPOCH={} LR_HORIZON={}',limit,self.max_epoch)


def a41_worker(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
    from yolox.utils import configure_nccl, configure_omp
    configure_nccl()
    configure_omp()
    cudnn.benchmark = False  # Determinism takes priority in the audit gate.
    trainer = LimitedA41Trainer(exp, args)
    trainer.train()


def main(argv=None):
    parser = native_train.make_parser()
    parser.add_argument('--stop-after-epoch', type=int, default=0)
    args = parser.parse_args(argv)
    exp = get_exp(args.exp_file, args.name)
    if getattr(args, 'opts', None):
        exp.merge(args.opts)
    if exp.fourier_variant not in ('a41', 'a41_control', 'a41_mosaic_a4'):
        raise RuntimeError('A4.1 launcher requires A4.1 experiment, not A4')
    if args.experiment_name is None:
        args.experiment_name = exp.exp_name
    if args.experiment_name != exp.exp_name:
        # Separate run dirs need separate immutable provenance; support seed
        # replications only through explicit exp_name override.
        exp.exp_name = args.experiment_name
    if not str(args.experiment_name).startswith('tdevnext3_a41_'):
        raise RuntimeError('A4.1 must write to an A4.1-namespaced directory')
    if args.devices != 1 or int(args.num_machines) != 1:
        raise RuntimeError('This launcher is certified only for -d 1 single GPU')
    if args.stop_after_epoch < 0 or args.stop_after_epoch > exp.max_epoch:
        raise RuntimeError('Invalid --stop-after-epoch')
    dest = ROOT / exp.output_dir / args.experiment_name
    if args.resume:
        provenance_path = dest / 'a41_provenance.json'
        if not provenance_path.is_file():
            raise RuntimeError('Resume requires matching A4.1 provenance: '+str(provenance_path))
        previous = json.loads(provenance_path.read_text(encoding='utf-8'))
        current = build_a41_provenance(ROOT, exp, args)
        verify_a41_provenance(previous, current)
    elif dest.exists() and any(dest.glob('*ckpt.pth')):
        raise RuntimeError('Refusing to overwrite existing A4.1 checkpoints: '+str(dest))
    logger.info('A4.1 TRAIN experiment={} basis={} seed={} stop={}',
                exp.exp_name, exp.a4_filter_basis, exp.seed, args.stop_after_epoch)
    launch(a41_worker, 1, num_machines=1, machine_rank=args.machine_rank,
           dist_url=args.dist_url, backend=args.dist_backend, args=(exp,args))


if __name__ == '__main__':
    main()
