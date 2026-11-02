import importlib.util
import json
import runpy
import tempfile
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
ctx = runpy.run_path(str(root / 'tests' / 'test_full_inventory.py'))
model = ctx['model']
head = model.head
# Trainer enables rich assignment collection only on telemetry iterations.
head.collect_assignment_telemetry = True
model.train()
model.zero_grad(set_to_none=True)
_ = model(torch.randn(2, 3, 64, 64), ctx['labels'])
stats = head.last_assignment_stats
required = {
    'num_fg_total', 'num_gt_total', 'size_gt_counts', 'size_positive_counts',
    'size_covered_gt_counts', 'size_zero_positive_gt_counts', 'fpn_positive_counts',
    'class_gt_counts', 'class_positive_counts', 'matched_iou_count',
    'dynamic_k_count', 'geometry_fallback_gt_total', 'pre_conflict_positive_total',
}
missing = required - set(stats)
assert not missing, missing
assert int(stats['num_gt_total']) == int(stats['size_gt_counts'].sum().item())
assert int(stats['num_fg_total']) == int(stats['size_positive_counts'].sum().item())
assert int(stats['num_fg_total']) == int(stats['fpn_positive_counts'].sum().item())
assert int(stats['num_gt_total']) == int(stats['class_gt_counts'].sum().item())
assert int(stats['num_fg_total']) == int(stats['class_positive_counts'].sum().item())
assert int(stats['num_fg_total']) == int(stats['matched_iou_count'].item())
assert int(stats['invalid_matched_iou_count'].item()) == 0
print('PASS: assignment telemetry conservation identities')

# Load telemetry module explicitly because test_full_inventory installs a utils stub.
spec = importlib.util.spec_from_file_location(
    'yolox.utils.telemetry', root / 'yolox' / 'utils' / 'telemetry.py'
)
telemetry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry_module)
TelemetryRecorder = telemetry_module.TelemetryRecorder

# model already has gradients from test_full_inventory second backward.
with tempfile.TemporaryDirectory() as tmp:
    rec = TelemetryRecorder(tmp, model, record_interval=1, rank=0, heavy_update_interval=1)
    pre = rec.capture_gradient_state(1)
    raw = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
    post = rec.capture_gradient_state(1)
    rec.record_gradient_flow(1, 1, pre, post, 10.0, float(raw))
    snap = rec.snapshot_parameter_updates(1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    opt.step()
    rec.record_parameter_updates(1, 1, snap, True, lr_used=1e-4,
                                 clip_coefficient=min(1.0, 10.0/(float(raw)+1e-12)))
    rec.record_simota_assignment(1, 1, stats)
    rec.record_architecture_state(1, 1)
    rec.record_iteration_losses(1, 1, 2e-4, {'total_loss': 1.0}, runtime={
        'lr_used': 1e-4, 'optimizer_step': True,
    })
    rec.record_prototype_geometry(1)
    rec.close()

    tdir = Path(tmp) / 'telemetry'
    expected = [
        'gradient_flow_telemetry.jsonl', 'parameter_update_telemetry.jsonl',
        'simota_telemetry.jsonl', 'architecture_state_telemetry.jsonl',
        'train_iter_telemetry.jsonl', 'prototype_geometry.jsonl',
    ]
    for name in expected:
        lines = (tdir / name).read_text().strip().splitlines()
        assert lines, name
        payload = json.loads(lines[-1])
        assert payload.get('telemetry_schema') == 3, (name, payload.keys())

    grad = json.loads((tdir / 'gradient_flow_telemetry.jsonl').read_text().splitlines()[-1])
    assert grad['pre_clip_total_grad_norm'] >= grad['post_clip_total_grad_norm']
    assert 0.0 < grad['clip_coefficient'] <= 1.0
    update = json.loads((tdir / 'parameter_update_telemetry.jsonl').read_text().splitlines()[-1])
    assert update['mode'] == 'full_group_exact'
    assert update['groups']
    assign = json.loads((tdir / 'simota_telemetry.jsonl').read_text().splitlines()[-1])
    assert set(assign['by_size']) == {'micro', 'medium', 'macro'}
    assert assign['matched_iou']['invalid_or_out_of_range'] == 0
    print('PASS: telemetry v3 JSONL schema, clipping, exact updates, assignment detail')
