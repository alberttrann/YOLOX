import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURE PATHS ---
# Root directory where images folder lives
IMAGE_ROOT = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"
# Output directory for the generated annotations
ANNOTATIONS_DIR = os.path.join(IMAGE_ROOT, "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# BDD100K Raw Labels
TRAIN_LABEL_RAW = "D:/YOLOX-3RD/bdd100k_labels_images_train.json"
VAL_LABEL_RAW = "D:/YOLOX-3RD/bdd100k_labels_images_val.json"

# Target 9 Detection Categories
CATEGORY_MAP = {
    "car": 1, "bus": 2, "truck": 3, "person": 4, 
    "rider": 5, "bike": 6, "motor": 7, 
    "traffic light": 8, "traffic sign": 9
}

# Adverse Weather Set (Test Target Domain)
TEST_WEATHER = {'rainy', 'snowy', 'foggy'}

def get_coco_skeleton():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()]
    }

def main():
    print("=" * 60)
    print("Step 1: Building Global Relative Path Index...")
    print(f"Scanning: {IMAGE_ROOT}")
    print("=" * 60)
    
    path_index = {}
    for p in tqdm(Path(IMAGE_ROOT).rglob('*.jpg'), desc="Indexing JPGs"):
        # Store relative path from IMAGE_ROOT with forward slashes (e.g., '100k/train/trainA/xxx.jpg')
        rel_path = str(p.relative_to(IMAGE_ROOT)).replace("\\", "/")
        path_index[p.name] = rel_path

    print(f"Found {len(path_index)} unique image files on disk.")

    # -------------------------------------------------------------
    # 2. Build Clean Training Split (Strictly from BDD train.json)
    # -------------------------------------------------------------
    print("\nStep 2: Building Clean Training Split (from train.json)...")
    with open(TRAIN_LABEL_RAW, 'r') as f:
        train_raw = json.load(f)

    train_clean_coco = get_coco_skeleton()
    img_id = 1
    ann_id = 1
    train_skipped = 0

    for entry in tqdm(train_raw, desc="Processing Train Clean"):
        fname = entry['name']
        if fname not in path_index:
            train_skipped += 1
            continue

        weather = entry.get('attributes', {}).get('weather', 'unknown').lower()
        # Clean training includes clear, overcast, partly cloudy, undefined
        if weather in TEST_WEATHER:
            continue

        train_clean_coco['images'].append({
            "id": img_id,
            "file_name": path_index[fname],  # Exact relative subfolder path!
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
                    train_clean_coco['annotations'].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": CATEGORY_MAP[cat],
                        "bbox": [box['x1'], box['y1'], w, h],
                        "area": float(w * h),
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    train_out_path = os.path.join(ANNOTATIONS_DIR, "tde_train_clean_coco.json")
    with open(train_out_path, 'w') as f:
        json.dump(train_clean_coco, f)
    print(f"Saved: {train_out_path} ({len(train_clean_coco['images'])} images, {len(train_clean_coco['annotations'])} boxes)")

    # -------------------------------------------------------------
    # 3. Build Clean Val & Adverse Test Splits (from val.json)
    # -------------------------------------------------------------
    print("\nStep 3: Building Clean Val & Adverse Test Splits (from val.json)...")
    with open(VAL_LABEL_RAW, 'r') as f:
        val_raw = json.load(f)

    val_clean_coco = get_coco_skeleton()
    test_adverse_coco = get_coco_skeleton()
    
    val_clean_img_id, val_clean_ann_id = 1, 1
    test_adv_img_id, test_adv_ann_id = 1, 1
    val_skipped = 0

    for entry in tqdm(val_raw, desc="Processing Val Splits"):
        fname = entry['name']
        if fname not in path_index:
            val_skipped += 1
            continue

        weather = entry.get('attributes', {}).get('weather', 'unknown').lower()
        is_adverse = weather in TEST_WEATHER

        target_coco = test_adverse_coco if is_adverse else val_clean_coco
        current_img_id = test_adv_img_id if is_adverse else val_clean_img_id

        target_coco['images'].append({
            "id": current_img_id,
            "file_name": path_index[fname],  # Exact relative subfolder path!
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
                    current_ann_id = test_adv_ann_id if is_adverse else val_clean_ann_id
                    target_coco['annotations'].append({
                        "id": current_ann_id,
                        "image_id": current_img_id,
                        "category_id": CATEGORY_MAP[cat],
                        "bbox": [box['x1'], box['y1'], w, h],
                        "area": float(w * h),
                        "iscrowd": 0
                    })
                    if is_adverse:
                        test_adv_ann_id += 1
                    else:
                        val_clean_ann_id += 1

        if is_adverse:
            test_adv_img_id += 1
        else:
            val_clean_img_id += 1

    val_out_path = os.path.join(ANNOTATIONS_DIR, "tde_val_clean_coco.json")
    test_out_path = os.path.join(ANNOTATIONS_DIR, "tde_test_adverse_coco.json")

    with open(val_out_path, 'w') as f:
        json.dump(val_clean_coco, f)
    with open(test_out_path, 'w') as f:
        json.dump(test_adverse_coco, f)

    print(f"Saved Clean Val: {val_out_path} ({len(val_clean_coco['images'])} images)")
    print(f"Saved Adverse Test: {test_out_path} ({len(test_adverse_coco['images'])} images)")
    print("\nAll ZSDA splits generated with verified relative disk paths!")

if __name__ == "__main__":
    main()