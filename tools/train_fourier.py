"""Windows single-GPU tdevNext.3 training entry point.

Use native YOLOX argparse/launch setup but call FourierTrainer explicitly.
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
from yolox.core.fourier_trainer import FourierTrainer


def fourier_worker(exp, args):
    """Dedicated worker entry point; no native Trainer rebinding required.

    Equivalent to the standard YOLOX single-GPU preamble while ensuring the
    FourierTrainer subclass, rather than the native trainer, is instantiated.
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
    logger.info("Trainer selected: {}", FourierTrainer.__name__)
    trainer = FourierTrainer(exp, args)
    trainer.train()


def main(argv=None):
    if not callable(getattr(native_train, "make_parser", None)):
        raise RuntimeError("Installed tools.train does not provide make_parser().")
    args = native_train.make_parser().parse_args(argv)
    if not args.exp_file:
        raise ValueError("Specify -f exps/default/tde_yolox_s_fourier_spectral.py")
    exp = get_exp(args.exp_file, args.name)
    if getattr(args, "opts", None):
        exp.merge(args.opts)
    if getattr(args, "experiment_name", None) is None:
        args.experiment_name = exp.exp_name
    if not getattr(args, "experiment_name", "").startswith("tdevnext3_"):
        raise RuntimeError(
            "Refusing to write to a non-vNext3 experiment directory: "
            + str(args.experiment_name))
    if not hasattr(exp, "fourier_variant"):
        raise RuntimeError("This launcher only accepts a vNext3 Fourier experiment")
    num_gpu = (torch.cuda.device_count() if args.devices is None
               else int(args.devices))
    if num_gpu != 1 or int(getattr(args, "num_machines", 1)) != 1:
        raise RuntimeError(
            "This launcher is validated only for one CUDA GPU; "
            "pass -d 1 and --num_machines 1")
    logger.info("Starting FourierTrainer: experiment={} variant={} GPUs={}",
                args.experiment_name, exp.fourier_variant, num_gpu)
    launch(fourier_worker, num_gpu,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           backend=args.dist_backend,
           args=(exp, args))


if __name__ == "__main__":
    main()
