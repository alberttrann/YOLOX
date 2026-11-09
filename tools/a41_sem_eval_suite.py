"""Native COCO point evaluation for auxiliary semantic experiments.

No bootstrap. Strict sem-provenance check before invoking existing A4.1
strict-checkpoint evaluator, which bypasses the train-only semantic head.
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from types import SimpleNamespace
from yolox.exp import get_exp
from yolox.core.a41_sem_trainer import sem_manifest

EXPS={
 'box':'exps/default/tde_yolox_s_fourier_a41_sem_box.py',
 'gaussian':'exps/default/tde_yolox_s_fourier_a41_sem_gaussian.py',
 'harmonic':'exps/default/tde_yolox_s_fourier_a41_sem_harmonic.py',
 'phase_probe':'exps/default/tde_yolox_s_fourier_a41_sem_phase_probe.py',
}
DATA={
 'clean':'exps/default/tde_yolox_s_fourier_off.py',
 'fog':'exps/default/test_acdc_fog.py',
 'rain':'exps/default/test_acdc_rain.py',
 'snow':'exps/default/test_acdc_snow.py',
 'night':'exps/default/test_acdc_night.py',
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default='.')
    p.add_argument('--modes',choices=EXPS,nargs='+',default=['box','gaussian','harmonic'])
    p.add_argument('--domains',choices=DATA,nargs='+',default=['clean','fog'])
    p.add_argument('--checkpoint-name',default='epoch_2_ckpt.pth')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--run',action='store_true')
    p.add_argument('--fp32',action='store_true')
    p.add_argument('--test-ann',action='store_true')
    a=p.parse_args()
    root=Path(a.repo).resolve()
    for mode in a.modes:
        exp_path=EXPS[mode]
        exp=get_exp(exp_path,None)
        ck=root/'YOLOX_outputs'/exp.exp_name/a.checkpoint_name
        for dom in a.domains:
            dest=root/'YOLOX_outputs'/'a41_sem_evals'/mode/dom
            cmd=[sys.executable,'tools/a41_eval_counterfactual.py',
                 '--model-exp',exp_path,'--data-exp',DATA[dom],
                 '--checkpoint',str(ck),'--weights','ema',
                 '--branches','on','--batch-size',str(a.batch_size),
                 '--conf','0.01','--nms','0.60','--device','cuda',
                 '--output-dir',str(dest),'--skip-existing']
            if not a.fp32:cmd.append('--fp16')
            if a.test_ann:cmd.append('--test')
            print('SEM_EVAL_PLAN',mode,dom,' '.join(cmd),flush=True)
            if not a.run:continue
            if not ck.is_file():raise FileNotFoundError(ck)
            sem_path=ck.parent/'a41_sem_provenance.json'
            if not sem_path.is_file():raise RuntimeError('Missing sem provenance '+str(sem_path))
            saved=json.loads(sem_path.read_text(encoding='utf8'))
            exp.exp_name=saved['experiment']
            now=sem_manifest(root,exp,SimpleNamespace(batch_size=saved['batch_size']))
            if now!=saved:raise RuntimeError('Semantic source/config provenance changed; refuse evaluation')
            subprocess.run(cmd,cwd=root,check=True)
    print('SEM_EVAL_SUITE_COMPLETE')

if __name__=='__main__':main()
