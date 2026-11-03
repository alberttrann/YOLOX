#!/usr/bin/env python3
"""vNext.2 TTT per-image eval installer (syntax-aware, source preserving).

Designed to accept BOTH original single-line vNext.2 and the multiline
frozen-EMA hotfix for `_forward_residual_ttt`. Does not replace or edit the
existing training/eval implementation; inserts one new method and one dispatch
statement after the original no-TTT early return.

Default action: check only. --diff: show proposed changes. --apply: backup and
atomically write the update. Running repeatedly is safe (idempotent).
"""
import argparse
import ast
import difflib
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
METHOD_FILE = HERE / 'vnext2_ttt_per_image_method.txt'
METHOD_NAME = '_forward_residual_ttt_per_image_eval'
TARGET_NAME = '_forward_residual_ttt'
CLASS_NAME = 'TTTAdaptiveStage'
DISPATCH = (
    "        if (not self.training and "
    "getattr(self, 'eval_ttt_batch_policy', 'per_image') == 'per_image'):\n"
    "            return self._forward_residual_ttt_per_image_eval(base_feat)\n"
)


def _class_and_method(source):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS_NAME]
    if len(classes) != 1:
        raise RuntimeError('Expected exactly one TTTAdaptiveStage class; no changes made')
    cls = classes[0]
    target = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == TARGET_NAME]
    injected = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == METHOD_NAME]
    if len(target) != 1 or len(injected) > 1:
        raise RuntimeError('Ambiguous residual TTT methods; no changes made')
    fn = target[0]
    if not isinstance(fn, ast.FunctionDef):
        raise RuntimeError('Unexpected async TTT method; no changes made')
    argnames = [a.arg for a in fn.args.args]
    if argnames != ['self', 'x_in', 'run_ttt', 'precomputed_feat']:
        raise RuntimeError(f'Unexpected {TARGET_NAME} signature: {argnames}; no changes made')
    # Ignore the docstring, then require precisely the known native-output
    # assignment and no-TTT early return. AST handles formatting variations.
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    if len(body) < 2:
        raise RuntimeError('TTT body is incomplete; no changes made')
    assign, early = body[:2]
    if not isinstance(assign, ast.Assign) or len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name) or assign.targets[0].id != 'base_feat':
        raise RuntimeError('Expected base_feat assignment after method docstring; no changes made')
    if not (
        isinstance(early, ast.If)
        and isinstance(early.test, ast.UnaryOp)
        and isinstance(early.test.op, ast.Not)
        and isinstance(early.test.operand, ast.Name)
        and early.test.operand.id == 'run_ttt'
        and len(early.body) == 1
        and isinstance(early.body[0], ast.Return)
        and isinstance(early.body[0].value, ast.Name)
        and early.body[0].value.id == 'base_feat'
        and len(early.orelse) == 0
    ):
        raise RuntimeError('Expected `if not run_ttt: return base_feat` early return; no changes made')
    has_dispatch = any(METHOD_NAME in ast.unparse(x) for x in body[2:])
    return tree, cls, fn, early, injected, has_dispatch


def rewrite(source):
    # All slicing performed on normalized LF while I/O retains original CRLF.
    source = source.replace('\r\n', '\n')
    _, _, fn, early, injected, has_dispatch = _class_and_method(source)
    if injected:
        if not has_dispatch:
            raise RuntimeError('Partial installation: helper exists without dispatch. No changes made.')
        return source
    if has_dispatch:
        raise RuntimeError('Partial installation: dispatch exists without helper. No changes made.')

    method_text = METHOD_FILE.read_text(encoding='utf-8').replace('\r\n', '\n').rstrip() + '\n\n'
    # Validate method source and indentation before inserting anything.
    parsed_method = ast.parse('class _Sentinel:\n' + method_text)
    if (len(parsed_method.body[0].body) != 1 or
        parsed_method.body[0].body[0].name != METHOD_NAME):
        raise RuntimeError('Unexpected helper method file; no changes made')
    original_lines = source.splitlines(keepends=True)
    if original_lines[fn.lineno - 1] != '    def _forward_residual_ttt(self, x_in, run_ttt=True, precomputed_feat=None):\n':
        raise RuntimeError('Method declaration does not match known indentation/signature; no changes made')
    # Insert helper BEFORE old method. Reparse to recover shifted line numbers.
    original_lines.insert(fn.lineno - 1, method_text)
    intermediate = ''.join(original_lines)
    _, _, new_fn, new_early, new_injected, _ = _class_and_method(intermediate)
    if len(new_injected) != 1 or new_early.end_lineno is None:
        raise RuntimeError('Internal method insertion mismatch; no changes made')
    lines = intermediate.splitlines(keepends=True)
    # Insert after the entire early-return block, without touching training flow.
    lines.insert(new_early.end_lineno, DISPATCH)
    result = ''.join(lines)
    ast.parse(result)

    # Security/correctness assertion: the existing method's old AST body must
    # be entirely preserved, except for exactly the newly inserted dispatch.
    _, _, post_fn, post_early, post_injected, post_dispatch = _class_and_method(result)
    if not (post_injected and post_dispatch):
        raise RuntimeError('Post-insertion verification failed; no changes made')
    expected_body = list(fn.body)
    new_body = list(post_fn.body)
    dispatch_nodes = [i for i, n in enumerate(new_body) if METHOD_NAME in ast.unparse(n)]
    if len(dispatch_nodes) != 1:
        raise RuntimeError('Unexpected dispatch count; no changes made')
    del new_body[dispatch_nodes[0]]
    if ast.dump(ast.Module(body=expected_body, type_ignores=[]), include_attributes=False) != ast.dump(ast.Module(body=new_body, type_ignores=[]), include_attributes=False):
        raise RuntimeError('Existing TTT method changed unexpectedly; no changes made')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='yolox/models/ttt_modules.py')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--apply', action='store_true')
    group.add_argument('--diff', action='store_true')
    args = parser.parse_args()
    path = Path(args.source)
    if not path.is_file():
        parser.error(f'Source file not found: {path}')
    raw = path.read_bytes()
    if raw.startswith(b'\xef\xbb\xbf'):
        raise RuntimeError('Unexpected UTF-8 BOM; refusing to rewrite source')
    text = raw.decode('utf-8')
    newline = '\r\n' if b'\r\n' in raw else '\n'
    before = text.replace('\r\n', '\n')
    after = rewrite(before)
    if after == before:
        print('ALREADY_INSTALLED: no changes made')
        return
    if args.diff:
        print(''.join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=str(path) + ' (original)', tofile=str(path) + ' (proposed)'
        )), end='')
        return
    if not args.apply:
        print(f'CHECK_PASSED: {path}; syntax-aware anchor verified; no changes made')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = path.with_name(path.name + '.before_per_image_' + stamp + '.bak')
    shutil.copy2(path, backup)
    contents = after.replace('\n', newline).encode('utf-8')
    # Atomic replacement: never leave a partially written Python module.
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + '.', suffix='.tmp',
                                         dir=path.parent, delete=False) as temp:
            tmppath = Path(temp.name)
            temp.write(contents)
        os.replace(tmppath, path)
    finally:
        if 'tmppath' in locals() and tmppath.exists():
            tmppath.unlink()
    print(f'PATCHED: {path}\nBACKUP: {backup}\nEXISTING_TRAINING_METHOD: preserved')


if __name__ == '__main__':
    main()
