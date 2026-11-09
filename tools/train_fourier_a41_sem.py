"""Single-GPU standalone semantic-auxiliary training launcher.

Inherits A4.1 Mosaic, LR, EMA, terminal curriculum and pretrained migration.
New directories and immutable extra provenance; original runs untouched.
"""
import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
import tools.train as native_train
from yolox.exp import get_exp
from yolox.core.launch import launch
from yolox.core.a41_sem_trainer import A41SemTrainer,sem_manifest
from yolox.core.a41_trainer import build_a41_provenance,verify_a41_provenance

class LimitedA41SemTrainer(A41SemTrainer):
    def train_in_epoch(self):
        limit=int(self.args.stop_after_epoch)
        if not limit:return super().train_in_epoch()
        if limit<0 or limit>self.max_epoch:raise ValueError('Invalid epoch limit')
        for self.epoch in range(self.start_epoch,min(limit,self.max_epoch)):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
        logger.info('A41_SEM_STOP_AFTER_COMPLETE_EPOCH={} LR_HORIZON={}',limit,self.max_epoch)

def worker(exp,args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic=True
    from yolox.utils import configure_nccl,configure_omp
    configure_nccl();configure_omp();cudnn.benchmark=False
    LimitedA41SemTrainer(exp,args).train()

def main(argv=None):
    parser=native_train.make_parser()
    parser.add_argument('--stop-after-epoch',type=int,default=0)
    args=parser.parse_args(argv)
    exp=get_exp(args.exp_file,args.name)
    if getattr(args,'opts',None):exp.merge(args.opts)
    if getattr(exp,'a41_sem_mode',None) not in (
            'legacy','gaussian','gaussian_harmonic','gaussian_phase_probe'):
        raise RuntimeError('Must use semantic-aux experiment, not historical A4.1')
    if args.experiment_name is None:args.experiment_name=exp.exp_name
    exp.exp_name=args.experiment_name
    if not exp.exp_name.startswith('tdevnext3_a41_sem_'):
        raise RuntimeError('Semantic experiment must have isolated namespace')
    if args.devices!=1 or args.num_machines!=1:
        raise RuntimeError('Certified launcher supports single GPU only')
    if not 0<=args.stop_after_epoch<=exp.max_epoch:raise ValueError('Invalid epoch limit')
    dst=ROOT/exp.output_dir/exp.exp_name
    if args.resume:
        for file_name,expected in (
          ('a41_provenance.json',build_a41_provenance(ROOT,exp,args)),
          ('a41_sem_provenance.json',sem_manifest(ROOT,exp,args))):
            p=dst/file_name
            if not p.is_file():raise RuntimeError('Missing provenance '+str(p))
            prev=json.loads(p.read_text(encoding='utf8'))
            if file_name=='a41_provenance.json':verify_a41_provenance(prev,expected)
            elif prev!=expected:raise RuntimeError('Semantic provenance drift')
    elif dst.exists() and any(dst.glob('*ckpt.pth')):
        raise RuntimeError('Refusing to overwrite checkpoints in '+str(dst))
    logger.info('A41_SEM_TRAIN mode={} exp={} grad_scale={} weight={}',
                exp.a41_sem_mode,exp.exp_name,exp.a41_sem_grad_scale,exp.a41_sem_loss_weight)
    launch(worker,1,num_machines=1,machine_rank=args.machine_rank,
           dist_url=args.dist_url,backend=args.dist_backend,args=(exp,args))

if __name__=='__main__':main()
