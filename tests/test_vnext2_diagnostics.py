"""Pure reporting-math sanity; no YOLOX repository required."""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import torch

stub = types.ModuleType('yolox')
stub.__path__=[]
sub = types.ModuleType('yolox.exp')
sub.get_exp = lambda *_: None
sys.modules['yolox'] = stub
sys.modules['yolox.exp'] = sub
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diag_v2', root/'tools/diagnose_vnext2.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
a = torch.tensor([[1., 2.]])
b = torch.tensor([[1., 2.]])
assert mod.relative_l2(a,b) == 0
assert mod.deltas(a,b)['max_abs'] == 0
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'ann.json'
    path.write_text(json.dumps({
        'images': [{'id':1},{'id':2}],
        'categories':[{'id':1,'name':'a'},{'id':2,'name':'b'}],
        'annotations':[{'image_id':1,'category_id':1},{'image_id':1,'category_id':1}]}))
    report=mod.annotation_report(path)
    assert report['images'] == 2
    assert report['images_without_annotations'] == 1
    assert report['objects_per_image']['max'] == 2
    assert report['class_support'][0]['num_annotations'] == 2
    assert report['class_support'][1]['num_annotations'] == 0
print('PASS: delta metrics and exact dataset class support')
