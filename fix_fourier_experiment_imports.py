"""Idempotent import hotfix for vNext.3 phase-1 YOLOX experiment files.

YOLOX's get_exp_by_file imports experiment modules directly by filename, with
exps/default on sys.path. `from exps.default...` fails when the repo root is
not on sys.path. This script changes only the FOUR problematic import lines.

Usage:
  python tools/fix_fourier_experiment_imports.py --repo .
  python tools/fix_fourier_experiment_imports.py --repo . --check
"""
import argparse
from pathlib import Path

REPLACEMENTS = {
    "tde_yolox_s_fourier.py": (
        "from exps.default.tde_yolox_s import Exp as BaseExp",
        "from tde_yolox_s import Exp as BaseExp",
    ),
    "tde_yolox_s_fourier_off.py": (
        "from exps.default.tde_yolox_s_fourier import Exp as BaseExp",
        "from tde_yolox_s_fourier import Exp as BaseExp",
    ),
    "tde_yolox_s_fourier_spatial.py": (
        "from exps.default.tde_yolox_s_fourier import Exp as BaseExp",
        "from tde_yolox_s_fourier import Exp as BaseExp",
    ),
    "tde_yolox_s_fourier_spectral.py": (
        "from exps.default.tde_yolox_s_fourier import Exp as BaseExp",
        "from tde_yolox_s_fourier import Exp as BaseExp",
    ),
}


def fix(repo: Path, check: bool = False):
    exps = repo / "exps" / "default"
    patched = []
    for name, (old, new) in REPLACEMENTS.items():
        path = exps / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing experiment file: {path}")
        text = path.read_text(encoding="utf-8")
        if text.count(new) == 1 and old not in text:
            print(f"OK (already fixed): {name}")
            continue
        if text.count(old) != 1:
            raise RuntimeError(
                f"Cannot safely identify import in {name}; expected exactly one: {old}"
            )
        patched.append((path, text.replace(old, new)))
        print(f"{'NEEDS FIX' if check else 'PATCH'}: {name}")
    if check:
        if patched:
            raise SystemExit(f"{len(patched)} experiment imports still require patching")
        print("All four experiment imports are compatible with the YOLOX loader.")
        return
    for path, text in patched:
        # Import-only edit; no model/loss/optimizer modifications.
        path.write_text(text, encoding="utf-8")
    print(f"Applied {len(patched)} import-only corrections. No training code changed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fix(args.repo.resolve(), args.check)
