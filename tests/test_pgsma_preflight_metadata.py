import json
from pathlib import Path
from tools.audit_pgsma_dataset import inspect_coco_file
from tools.audit_pgsma_onnx_runtime import parse_shapes


def test_coco_valid_metadata(tmp_path):
    file=tmp_path/'source.json'
    file.write_text(json.dumps({'images':[{'id':1,'width':100,'height':80,'file_name':'a.jpg'}],
       'categories':[{'id':5,'name':'car'}],
       'annotations':[{'id':3,'image_id':1,'category_id':5,'bbox':[10,5,6,4],'iscrowd':0}]}))
    d=inspect_coco_file(file,expected_classes=1)
    assert d['max_non_crowd_annotations_per_image']==1
    assert not d['issues']
    assert d['categories']['5']=='car'


def test_coco_crowd_and_bbox_problems_flagged(tmp_path):
    file=tmp_path/'broken.json'
    file.write_text(json.dumps({'images':[{'id':1,'width':40,'height':20}],
       'categories':[{'id':9,'name':'vehicle'}],
       'annotations':[{'id':0,'image_id':1,'category_id':9,'bbox':[1,1,3,3],'iscrowd':1},
                      {'id':1,'image_id':1,'category_id':9,'bbox':[37,10,5,5],'iscrowd':0},
                      {'id':2,'image_id':1,'category_id':9,'bbox':[2,2,-1,2],'iscrowd':0}]}))
    d=inspect_coco_file(file,expected_classes=1)
    assert d['crowd']==1
    assert d['outside_boxes']==1
    assert d['invalid_ann']==1
    assert d['under_4px_original']==1
    assert d['issues']


def test_dynamic_onnx_shapes():
    assert parse_shapes(['64x96','128x128'])==[(64,96),(128,128)]
    try:parse_shapes(['65x96'])
    except ValueError:pass
    else:raise AssertionError('Misaligned shapes should fail')
