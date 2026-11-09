"""Atomic full-overlay installer; NEVER rewrites installed YOLOX model source.

Understands older failed tdevNext.3 Phase-1 files and backs them up first.
Run from extracted bundle's overlay/tools path:
 python overlay/tools/install_full_fourier.py --repo . --dry-run
 python overlay/tools/install_full_fourier.py --repo . --apply
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REL_FILES=(
    'yolox/models/fourier_invariance.py',
    'yolox/models/fourier_backbone.py',
    'yolox/models/fourier_head.py',
    'yolox/models/fourier_detector.py',
    'yolox/core/fourier_trainer.py',
    'exps/default/tde_yolox_s_fourier.py',
    'exps/default/tde_yolox_s_fourier_off.py',
    'exps/default/tde_yolox_s_fourier_spatial.py',
    'exps/default/tde_yolox_s_fourier_spectral.py',
    'exps/default/tde_yolox_s_fourier_spectral_memory.py',
    'tools/train_fourier.py',
    'tools/check_fourier_integration.py',
    'tools/export_full_fourier.py',
    'tools/profile_full_fourier.py',
)
BASE_REQUIREMENTS={
    'yolox/models/engram_head.py':'TDE_Head',
    'yolox/models/yolo_pafpn.py':'YOLOPAFPN',
    'yolox/models/darknet.py':'CSPDarknet',
    'yolox/models/ttt_modules.py':'mHC_AttnRes_SPI',
    'exps/default/tde_yolox_s.py':'Exp',
}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def verify_source(repo):
    for rel,class_name in BASE_REQUIREMENTS.items():
        p=repo/rel
        if not p.is_file():raise RuntimeError(f'Missing required vNext2 source: {p}')
        tree=ast.parse(p.read_text(encoding='utf-8'))
        classes=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
        if class_name not in classes:
            raise RuntimeError(f'{rel}: expected class {class_name}, found {classes}')


def install(repo, overlay, *, apply=False, verify_import=True):
    repo=Path(repo).resolve(); overlay=Path(overlay).resolve()
    verify_source(repo)
    inputs=[]
    for rel in REL_FILES:
        src=overlay/rel
        if not src.is_file():raise RuntimeError(f'Bundle incomplete; missing {src}')
        ast.parse(src.read_text(encoding='utf-8'))
        dst=repo/rel
        inputs.append({'path':rel,'old_sha256':digest(dst),'new_sha256':digest(src),
                       'change':('unchanged' if dst.exists() and digest(dst)==digest(src)
                                 else 'replace' if dst.exists() else 'new')})
    for item in inputs:print(f"{item['change']:9} {item['path']}")
    if not apply:
        print('DRY RUN: source type checks passed; no files changed')
        return inputs
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=repo/'tdevnext3_fourier_backups'/timestamp
    backup.mkdir(parents=True,exist_ok=False)
    applied=[]
    try:
        for item in inputs:
            rel=item['path'];dst=repo/rel;src=overlay/rel
            if item['change']=='unchanged':continue
            if dst.exists():
                prev=backup/rel;prev.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(dst,prev)
            dst.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(dst.parent),delete=False,
                                             suffix='.py') as temp:
                temp.write(src.read_bytes());temp.flush()
                temp_name=temp.name
            try:os.replace(temp_name,dst)
            finally:
                if os.path.exists(temp_name):os.unlink(temp_name)
            applied.append(item)
        if verify_import:
            imports=[f'exps/default/tde_yolox_s_fourier_{v}.py'
                     for v in ('off','spatial','spectral','spectral_memory')]
            script='from yolox.exp import get_exp\n'
            script+='for x in '+repr(imports)+':\n'
            script+='    e=get_exp(x,None)\n'
            script+='    assert e.fourier_variant in '+repr(('off','spatial','spectral','spectral_memory'))+'\n'
            subprocess.run([sys.executable,'-c',script],cwd=repo,check=True,
                           timeout=90)
        manifest={'version':'tdevnext3_fourier_full_fidelity_1.0',
                  'utc':timestamp,'changes':inputs,
                  'source_base_sha256':{rel:digest(repo/rel) for rel in BASE_REQUIREMENTS}}
        (backup/'install_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('INSTALL OK. Backups and hashes:',backup)
        return inputs
    except Exception:
        # Transactional rollback, leaving original source intact.
        for item in reversed(applied):
            dst=repo/item['path'];original=backup/item['path']
            if original.exists():shutil.copy2(original,dst)
            elif dst.exists():dst.unlink()
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True,type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--apply',action='store_true')
    p.add_argument('--no-import-check',action='store_true',
                   help='Skip native-loader check only in offline environments')
    a=p.parse_args()
    install(a.repo,Path(__file__).resolve().parents[2],apply=a.apply,
            verify_import=not a.no_import_check)
