"""Conservative vNext.3 installer. Defaults to dry-run; refuses conflicts.

Use `--check` before `--apply`. Backups of existing files are created on
explicit apply, and files different from the archived base are never overwritten.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--repo',required=True,type=Path)
    mode=cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    args=cli.parse_args()
    bundle=Path(__file__).resolve().parents[1]
    source=bundle/'overlay'
    manifest=json.loads((bundle/'manifest.json').read_text())
    repo=args.repo.resolve()
    if not repo.is_dir() or not (repo/'yolox/models/engram_head.py').is_file():
        raise SystemExit('The target must be an existing full YOLOX/vNext.2 checkout')
    changes=[]; conflicts=[]
    for f in sorted(source.rglob('*.py')):
        relative=f.relative_to(source)
        dest=repo/relative
        prior=manifest['archived_sha256'].get(str(relative).replace('\\','/'))
        if dest.exists():
            digest=sha256(dest)
            if digest==sha256(f):
                changes.append((f,dest,'already-installed'))
            elif prior and digest==prior:
                changes.append((f,dest,'replace-archived'))
            else:
                conflicts.append(str(relative))
        else:
            if prior:
                conflicts.append(str(relative)+' (archived baseline expected but file missing)')
            else:
                changes.append((f,dest,'new'))
    for _,dst,operation in changes:
        print(f'{operation:18s} {dst.relative_to(repo)}')
    if conflicts:
        print('\nCONFLICTS. NOT overwriting user hotfixes:')
        for item in conflicts: print('  ',item)
        raise SystemExit(2)
    if args.apply:
        backup=repo/('vnext3_pgsma_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        for src,dst,operation in changes:
            if operation=='already-installed':continue
            if operation=='replace-archived':
                saved=backup/dst.relative_to(repo)
                saved.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(dst,saved)
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
        print('\nInstallation complete. Backup directory:',backup)
        print('Full training remains OFF until real checkout preflight and data audit pass.')
    else:
        print('\nDry run only. No files changed.')

if __name__=='__main__':main()
