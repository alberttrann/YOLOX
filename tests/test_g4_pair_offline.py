import sys
from pathlib import Path
import json
import unittest
import tempfile
import importlib.util

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('g4pair',HERE/'tools/audit_pgsma_g4_pair.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class PairTests(unittest.TestCase):
    def make(self, path, objective, ap=0.2, hashed='X'):
        path.mkdir()
        m={k:0.2 for k in mod.COCO}
        r={'status':'CHECKS_PASSED','native_state_sha256':hashed,
           'training_batch_stream_sha256':'Y','first_training_batch_sha256':'Z',
           'successful_native_updates':4,'successful_semantic_updates':4,
           'clean_subset_coco_native_final':m,
           'clean_subset_coco_final':{**m,'AP50_95':ap}}
        (path/'result.json').write_text(json.dumps(r))
        rows=[{'step':1,'native_loss':4.0,'semantic_utility':{'candidate_fg_improvement':0.01}},
              {'step':4,'native_loss':3.0,'semantic_utility':{'candidate_fg_improvement':0.02}}]
        (path/'train_metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))

    def test_paired_valid(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'v2',Path(d)/'margin'
            self.make(a,'v2');self.make(b,'margin',0.201)
            r=mod.compare(a,b)
            self.assertAlmostEqual(r['performance']['AP50_95']['margin_minus_v2'],0.001)
            self.assertEqual(r['training']['common_logged_steps'],[1,4])

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'v2',Path(d)/'margin'
            self.make(a,'v2');self.make(b,'margin',hashed='DIFFERENT')
            with self.assertRaises(ValueError):mod.compare(a,b)

    def test_not_equal_native_eval_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'v2',Path(d)/'margin'
            self.make(a,'v2');self.make(b,'margin')
            p=b/'result.json';r=json.loads(p.read_text());r['clean_subset_coco_native_final']['AP50_95']=.21
            p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):mod.compare(a,b)

if __name__=='__main__':unittest.main()
