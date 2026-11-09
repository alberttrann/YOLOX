"""Windows single-GPU tdevNext.3 training entry point.

Use native YOLOX argparse/launch setup but call A4Trainer explicitly.
There is deliberately NO mutation of native tools.train.Trainer, and no
assumption that tools.train has imported a Trainer symbol. This preserves
the original tools/train.py, vNext.2 trainer, EMA and training jobs.
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

# Native YOLOX scripts are normally launched by file path. Preserve their
# source-root import behavior when invoked from another working directory.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.backends.cudnn as cudnn
from loguru import logger
import tools.train as native_train
from yolox.exp import get_exp
from yolox.core.launch import launch
from yolox.core.a4_matched_trainer import A4MatchedControlTrainer


class LimitedA4Trainer(A4MatchedControlTrainer):
    def train_in_epoch(self):
        limit=int(self.args.stop_after_epoch)
        if limit == 0:
            return super().train_in_epoch()
        if limit < 0 or limit > self.max_epoch:
            raise ValueError('Invalid completed epoch limit')
        for self.epoch in range(self.start_epoch,min(limit,self.max_epoch)):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
        logger.info('A4_STOP_AFTER_COMPLETE_EPOCH={} LR_HORIZON={}',limit,self.max_epoch)


def fourier_worker(exp, args):
    """Dedicated worker entry point; no native Trainer rebinding required.

    Equivalent to the standard YOLOX single-GPU preamble while ensuring the
    Matched-control trainer subclass, rather than native trainer, is instantiated.
    """
    if getattr(exp, "seed", None) is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        logger.info("Fourier experiment seed: {}", exp.seed)
    from yolox.utils import configure_nccl, configure_omp
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    logger.info("Trainer selected: {}", A4MatchedControlTrainer.__name__)
    trainer = LimitedA4Trainer(exp, args)
    trainer.train()


def main(argv=None):
    if not callable(getattr(native_train, "make_parser", None)):
        raise RuntimeError("Installed tools.train does not provide make_parser().")
    parser=native_train.make_parser()
    parser.add_argument('--stop-after-epoch',type=int,default=0,
                        help='0 full horizon; otherwise stop after this completed epoch without changing LR horizon')
    args=parser.parse_args(argv)
    if not args.exp_file:
        raise ValueError("Specify -f exps/default/tde_yolox_s_fourier_spectral.py")
    exp = get_exp(args.exp_file, args.name)
    if getattr(args, "opts", None):
        exp.merge(args.opts)
    if getattr(args, "experiment_name", None) is None:
        raise RuntimeError('Must explicitly set -expn tdevnext3_matched_<variant>_seedN')
    if not getattr(args, "experiment_name", "").startswith("tdevnext3_matched_"):
        raise RuntimeError(
            "Refusing to write to a non-matched-control experiment directory: "
            + str(args.experiment_name))
    if getattr(exp, "fourier_variant", None) not in ("off","spatial","spectral","spectral_memory"):
        raise RuntimeError("This control launcher only accepts A0/A1/A2/A3 experiment definitions")
    num_gpu = (torch.cuda.device_count() if args.devices is None
               else int(args.devices))
    if num_gpu != 1 or int(getattr(args, "num_machines", 1)) != 1:
        raise RuntimeError(
            "This launcher is validated only for one CUDA GPU; "
            "pass -d 1 and --num_machines 1")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > exp.max_epoch:
        raise RuntimeError('Invalid --stop-after-epoch')
    out_dir=_REPO/exp.output_dir/args.experiment_name
    if args.resume:
        manifest=out_dir/'fourier_architecture_manifest.json'
        if not manifest.exists():
            raise RuntimeError('Cannot resume A4 without matching provenance: '+str(manifest))
        import json
        data=json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('variant') != exp.fourier_variant:
            raise RuntimeError('Checkpoint provenance differs in control variant')
    elif any(out_dir.glob('*_ckpt.pth')):
        raise RuntimeError('Refusing to overwrite existing A4 checkpoints at '+str(out_dir))
    logger.info("Starting matched control trainer: experiment={} variant={} GPUs={}",
                args.experiment_name, exp.fourier_variant, num_gpu)
    launch(fourier_worker, num_gpu,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           backend=args.dist_backend,
           args=(exp, args))


if __name__ == "__main__":
    main()
