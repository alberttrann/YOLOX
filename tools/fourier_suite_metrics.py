"""Pure-stdlib parser for YOLOX's COCO evaluator summaries.

Keeps area-conditioned metrics and per-class values.  Returns None for metrics
not present in a log; never infers them or converts nan into 0.
"""
from __future__ import annotations
import re

VARIANTS = {
    'A0': ('off', 'tdevnext3_a0_off', 'tde_yolox_s_fourier_off.py'),
    'A1': ('spatial', 'tdevnext3_a1_spatial', 'tde_yolox_s_fourier_spatial.py'),
    'A2': ('spectral', 'tdevnext3_a2_spectral', 'tde_yolox_s_fourier_spectral.py'),
    'A3': ('spectral_memory', 'tdevnext3_a3_spectral_memory', 'tde_yolox_s_fourier_spectral_memory.py'),
}
METRIC = re.compile(
    r'Average\s+(Precision|Recall)\s+\((AP|AR)\)\s+@\[\s*IoU=\s*'
    r'(0\.50:0\.95|0\.50|0\.75)\s*\|\s*area=\s*(all|small|medium|large)\s*'
    r'\|\s*maxDets=\s*(\d+)\s*\]\s*=\s*(\d+(?:\.\d+)?|nan|-1(?:\.0+)?)',
    flags=re.IGNORECASE,
)
TRAIN_EPOCH = re.compile(r'epoch:\s*\[(\d+)\/\d+\]\[(\d+)\/(\d+)\]')
INFER_TIME = re.compile(r'Average forward time:\s*([\d.]+) ms,\s*Average NMS time:\s*([\d.]+) ms,\s*Average inference time:\s*([\d.]+) ms')
CLASS_RE = re.compile(r'\|\s*([^|]+?)\s*\|\s*(\d+(?:\.\d+)?|nan)\s*(?=\|)', flags=re.IGNORECASE)


def metric_key(group, iou, area, maxdets):
    if group == 'AP':
        key = 'AP' if iou == '0.50:0.95' else ('AP50' if iou == '0.50' else 'AP75')
        return key if area == 'all' else key+'_'+area
    key = 'AR'+str(maxdets)
    return key if area == 'all' else key+'_'+area


def parse_coco_summary(text):
    result = {}
    for match in METRIC.finditer(text):
        _,group,iou,area,maxdets,value = match.groups()
        key=metric_key(group.upper(),iou,area,int(maxdets))
        try: v=float(value)
        except ValueError: continue
        if v < 0 or v != v: v = None
        result[key]=v
    time_match = INFER_TIME.search(text)
    if time_match:
        for label,value in zip(('forward_ms','nms_ms','inference_ms'),time_match.groups()):
            result[label]=float(value)
    # Parse YOLOX summary's per-class tables without inferring absent classes.
    for tag in ('AP','AR'):
        m=re.search(r'per class '+tag+r':\s*\n((?:\|[^\n]*\n)+)',text,re.IGNORECASE)
        if m:
            lines=m.group(1).splitlines()
            classes={}
            for line in lines:
                tokens=[t.strip() for t in line.split('|')[1:-1]]
                for i in range(0,len(tokens)-1,2):
                    name=tokens[i]; value=tokens[i+1]
                    if not name or name.lower() in ('class','ap','ar') or set(name)<=set(':- '):continue
                    try:v=float(value)/100.0
                    except ValueError:continue
                    if v == v: classes[name]=v
            if classes:result['per_class_'+tag]=classes
    return result


def parse_training_evaluations(log):
    """Collect each COCO summary and attach the latest *completed* train epoch.

    If logs contain multiple runs, the last evaluation per epoch is selected,
    but callers should verify source/variant manifest first.
    """
    matches=list(TRAIN_EPOCH.finditer(log))
    apstarts=list(re.finditer(r'Average\s+Precision\s+\(AP\)\s+@\[\s*IoU=\s*0\.50:0\.95\s*\|\s*area=\s*all',log))
    entries={}
    for ap in apstarts:
        preceding=[m for m in matches if m.start() < ap.start()]
        if not preceding:continue
        epoch=int(preceding[-1].group(1))
        # The whole metric block includes recall, per-class, and inference
        # timing immediately before it; stop at the next epoch's training log.
        next_train=next((m.start() for m in matches if m.start()>ap.start()),len(log))
        start=max(log.rfind('Average forward time:',0,ap.start()),preceding[-1].end())
        if start<0:start=ap.start()
        content=log[start:next_train]
        metrics=parse_coco_summary(content)
        if metrics.get('AP') is not None:
            entries[epoch]=metrics
    return entries


def delta(a, b):
    if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):
        return None
    return round(a-b,6)
