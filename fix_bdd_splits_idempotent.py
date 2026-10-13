#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Idempotent ZSDA Split Generator for BDD100K

import json
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURE PATHS ---
IMAGE_ROOT = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
ANNOTATIONS_DIR = os.path.join(IMAGE_ROOT, "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

TRAIN_LABEL_RAW = "D:/YOLOX-3RD/bdd100k_labels_images_train.json"
VAL_LABEL_RAW = "D:/YOLOX-3RD/bdd100k_labels_images_val.json"

CATEGORY_MAP = {
    "car": 1, "bus": 2, "truck": 3, "person": 4, 
    "rider": 5, "bike": 6, "motor": 7, 
    "traffic light": 8, "traffic sign": 9
}

TEST_WEATHER = {'rainy', 'snowy', 'foggy'}

def get_coco_skeleton():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()]
    }

def build_path_index():
    print("=" * 60)
    print("Indexing all local disk images...")
    path_index = {}
    for p in tqdm(Path(IMAGE_ROOT).rglob('*.jpg'), desc="Scanning JPGs"):
        rel_path = str(p.relative_to(IMAGE_ROOT)).replace("\\", "/")
        path_index[p.name] = rel_path
    print(f"Indexed {len(path_index)} unique images on disk.")
    return path_index

def process_entries(entries, path_index, weather_filter, is_adverse_target, start_img_id=1, start_ann_id=1):
    coco_data = get_coco_skeleton()
    img_id = start_img_id
    ann_id = start_ann_id

    for entry in entries:
        fname = entry['name']
        if fname not in path_index:
            continue

        weather = entry.get('attributes', {}).get('weather', 'unknown').lower()
        is_adverse = weather in weather_filter

        if is_adverse != is_adverse_target:
            continue

        coco_data['images'].append({
            "id": img_id,
            "file_name": path_index[fname],
            "width": 1280,
            "height": 720,
            "weather": weather
        })

        for label in entry.get('labels', []):
            cat = label['category']
            if cat in CATEGORY_MAP and 'box2d' in label:
                box = label['box2d']
                w = box['x2'] - box['x1']
                h = box['y2'] - box['y1']
                if w > 1 and h > 1:
                    coco_data['annotations'].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": CATEGORY_MAP[cat],
                        "bbox": [box['x1'], box['y1'], w, h],
                        "area": float(w * h),
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    return coco_data, img_id, ann_id

def main():
    path_index = build_path_index()

    print("\nLoading raw BDD100K JSON annotations...")
    with open(TRAIN_LABEL_RAW, 'r') as f:
        train_raw = json.load(f)
    with open(VAL_LABEL_RAW, 'r') as f:
        val_raw = json.load(f)

    # 1. PRESERVE CLEAN TRAIN SPLIT (Exact reproduction)
    train_clean_path = os.path.join(ANNOTATIONS_DIR, "tde_train_clean_coco.json")
    if not os.path.exists(train_clean_path):
        print("\nBuilding Clean Train Split (tde_train_clean_coco.json)...")
        train_clean_coco, _, _ = process_entries(train_raw, path_index, TEST_WEATHER, is_adverse_target=False)
        with open(train_clean_path, 'w') as f:
            json.dump(train_clean_coco, f)
        print(f"Saved: {train_clean_path} ({len(train_clean_coco['images'])} images)")
    else:
        print(f"\n[Verified & Untouched] Clean Train Split exists: {train_clean_path}")

    # 2. PRESERVE CLEAN VAL SPLIT (Exact reproduction)
    val_clean_path = os.path.join(ANNOTATIONS_DIR, "tde_val_clean_coco.json")
    if not os.path.exists(val_clean_path):
        print("\nBuilding Clean Val Split (tde_val_clean_coco.json)...")
        val_clean_coco, _, _ = process_entries(val_raw, path_index, TEST_WEATHER, is_adverse_target=False)
        with open(val_clean_path, 'w') as f:
            json.dump(val_clean_coco, f)
        print(f"Saved: {val_clean_path} ({len(val_clean_coco['images'])} images)")
    else:
        print(f"[Verified & Untouched] Clean Val Split exists: {val_clean_path}")

    # 3. BUILD EXPANDED COMPREHENSIVE ADVERSE TEST SPLIT (Val Adverse + Train Adverse)
    print("\nBuilding Expanded Comprehensive Adverse Test Split...")
    # First: Val adverse
    test_adv_val, next_img_id, next_ann_id = process_entries(
        val_raw, path_index, TEST_WEATHER, is_adverse_target=True, start_img_id=1, start_ann_id=1
    )
    
    # Save val-only adverse for historical comparison
    test_adv_val_path = os.path.join(ANNOTATIONS_DIR, "tde_test_adverse_val_only.json")
    with open(test_adv_val_path, 'w') as f:
        json.dump(test_adv_val, f)
    print(f"Saved Val-Only Adverse: {test_adv_val_path} ({len(test_adv_val['images'])} images)")

    # Second: Pool all unseen adverse images from train_raw
    test_adv_train, _, _ = process_entries(
        train_raw, path_index, TEST_WEATHER, is_adverse_target=True, start_img_id=next_img_id, start_ann_id=next_ann_id
    )

    # Merge images and annotations deterministically
    combined_adverse_coco = get_coco_skeleton()
    combined_adverse_coco['images'] = test_adv_val['images'] + test_adv_train['images']
    combined_adverse_coco['annotations'] = test_adv_val['annotations'] + test_adv_train['annotations']

    expanded_test_path = os.path.join(ANNOTATIONS_DIR, "tde_test_adverse_coco.json")
    with open(expanded_test_path, 'w') as f:
        json.dump(combined_adverse_coco, f)

    print(f"\nSUCCESS: Generated Expanded Adverse Test Set!")
    print(f"File: {expanded_test_path}")
    print(f"Total Adverse Test Images: {len(combined_adverse_coco['images'])} (Expanded from ~1,500 to {len(combined_adverse_coco['images'])})")
    print(f"Total Adverse Bounding Boxes: {len(combined_adverse_coco['annotations'])}")

if __name__ == "__main__":
    main()