import json
import os
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

ROOT_DIR = "C:/TDE-YOLOX"
BDD_LABEL_FILES = [
    os.path.join(ROOT_DIR, "bdd100k_labels_images_train.json"),
    os.path.join(ROOT_DIR, "bdd100k_labels_images_val.json")
]
IMAGE_ROOT = os.path.join(ROOT_DIR, "bdd100k/bdd100k/images")

# Adverse OOD categories for the TEST set
TEST_WEATHER = ['rainy', 'snowy', 'foggy']

# BDD100K Detection categories to keep
CATEGORY_MAP = {
    "car": 1, "bus": 2, "truck": 3, "person": 4, 
    "rider": 5, "bike": 6, "motor": 7, 
    "traffic light": 8, "traffic sign": 9
}

def get_coco_skeleton():
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()]
    }

def preprocess():
    # --- STEP 1: GLOBAL INDEXING WITH COLLISION GUARD ---
    print("Step 1: Building Global Image Index...")
    path_index = defaultdict(list)
    collision_count = 0
    
    for path in Path(IMAGE_ROOT).rglob('*.jpg'):
        # Normalize path to relative string with forward slashes
        rel_path = str(path.relative_to(IMAGE_ROOT)).replace("\\", "/")
        path_index[path.name].append(rel_path)
        if len(path_index[path.name]) > 1:
            collision_count += 1

    print(f"Index built. Found {len(path_index)} unique filenames.")
    if collision_count > 0:
        print(f"WARNING: Found {collision_count} filename collisions.")
        print("Strategy: Prioritizing the first physical path found.")

    train_coco = get_coco_skeleton()
    test_coco = get_coco_skeleton()
    
    img_id_counter = 1
    ann_id_counter = 1
    skipped_no_file = 0

    # --- STEP 2: JSON CONVERSION ---
    for label_file in BDD_LABEL_FILES:
        with open(label_file, 'r') as f:
            bdd_data = json.load(f)

        for entry in tqdm(bdd_data, desc=f"Converting {os.path.basename(label_file)}"):
            file_name = entry['name']
            
            # Use the index to find where the file actually lives
            if file_name not in path_index:
                skipped_no_file += 1
                continue

            # Selection logic
            weather = entry.get('attributes', {}).get('weather', 'unknown').lower()
            if weather in TEST_WEATHER:
                target_dataset = test_coco
            else:
                target_dataset = train_coco

            # Register Image
            target_dataset['images'].append({
                "id": img_id_counter,
                "file_name": path_index[file_name][0], # Use the first path instance
                "width": 1280,
                "height": 720,
                "weather": weather
            })

            # Register Annotations
            for label in entry.get('labels', []):
                cat = label['category']
                if cat not in CATEGORY_MAP or 'box2d' not in label:
                    continue
                
                box = label['box2d']
                x = box['x1']
                y = box['y1']
                w = box['x2'] - box['x1']
                h = box['y2'] - box['y1']

                if w <= 1 or h <= 1: # Filter out noise/too small
                    continue

                target_dataset['annotations'].append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": CATEGORY_MAP[cat],
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0
                })
                ann_id_counter += 1
            
            img_id_counter += 1

    # --- STEP 3: OUTPUT ---
    print(f"Step 3: Writing JSONs...")
    with open(os.path.join(ROOT_DIR, 'tde_train_coco.json'), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(ROOT_DIR, 'tde_test_adverse_coco.json'), 'w') as f:
        json.dump(test_coco, f)

    print("\n" + "="*30)
    print("PREPROCESSING SUMMARY")
    print("="*30)
    print(f"Total Unique Images Registered: {img_id_counter - 1}")
    print(f"Clean/Undefined Training Images: {len(train_coco['images'])}")
    print(f"Adverse Testing Images (Rain/Snow/Fog): {len(test_coco['images'])}")
    print(f"Images skipped (no local file found): {skipped_no_file}")
    print(f"Final labels saved to {ROOT_DIR}")

if __name__ == "__main__":
    preprocess()