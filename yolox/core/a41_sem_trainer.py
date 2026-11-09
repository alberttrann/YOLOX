"""Additive transitive semantic provenance; retains the certified A4.1 trainer."""
import hashlib
import json
from pathlib import Path
from loguru import logger
from .a41_trainer import A41Trainer

SEM_FILES=(
 'yolox/models/a41_sem_aux.py',
 'yolox/models/a41_sem_detector.py',
 'yolox/core/a41_sem_trainer.py',
 'tools/train_fourier_a41_sem.py',
 'exps/default/tde_yolox_s_fourier_a41_sem_box.py',
 'exps/default/tde_yolox_s_fourier_a41_sem_gaussian.py',
 'exps/default/tde_yolox_s_fourier_a41_sem_harmonic.py',
 'exps/default/tde_yolox_s_fourier_a41_sem_phase_probe.py',
)

def sem_manifest(repo,exp,args):
    hashes={}
    for rel in SEM_FILES:
        path=Path(repo)/rel
        if not path.is_file():
            raise RuntimeError('Missing semantic research source '+str(path))
        hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'schema':'a41_aux_sem_v1', 'experiment':exp.exp_name,
        'aux_mode':exp.a41_sem_mode,
        'sem_loss_weight':float(exp.a41_sem_loss_weight),
        'backbone_grad_scale':float(exp.a41_sem_grad_scale),
        'sigma_divisor':float(exp.a41_sem_sigma_divisor),
        'harmonic_weight':float(exp.a41_sem_harmonic_weight),
        'seed':exp.seed,'batch_size':int(args.batch_size),
        'max_epoch':int(exp.max_epoch),
        'train_ann_hash':hashlib.sha256(Path(exp.train_ann).read_bytes()).hexdigest(),
        'source_sha256':hashes,
        'inference_semantic_decoder':'never_executed',
    }

class A41SemTrainer(A41Trainer):
    def before_train(self):
        super().before_train()
        if self.rank!=0:
            return
        root=Path(__file__).resolve().parents[2]
        manifest=sem_manifest(root,self.exp,self.args)
        path=Path(self.file_name)/'a41_sem_provenance.json'
        if self.args.resume:
            if not path.is_file():
                raise RuntimeError('Missing immutable semantic provenance on resume')
            prev=json.loads(path.read_text(encoding='utf8'))
            mismatches=[k for k,v in manifest.items() if prev.get(k)!=v]
            if mismatches:
                raise RuntimeError('Semantic provenance changed: '+str(mismatches))
        else:
            if path.exists():
                raise RuntimeError('Refusing to overwrite existing semantic provenance')
            path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
        logger.info('A41_SEM_PROVENANCE_PASS mode={} files={} path={}',
                    self.exp.a41_sem_mode,len(SEM_FILES),path)
