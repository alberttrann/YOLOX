import json
import os
from tqdm import tqdm
from pathlib import Path

IMAGE_ROOT_DIR = r"D:\ACDC\rgb_anon"
LABEL_ROOT_DIR = r"D:\gt_detection_trainval\gt_detection"

OUTPUT_DIR = r"D:\YOLOX-3rd\acdc_labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- BDD TARGET DEFINITION ---
BDD_NAME_TO_ID = {
    "car": 1, "bus": 2, "truck": 3,
    "person": 4, "rider": 5, "bicycle": 6, # ACDC uses 'bicycle', BDD uses 'bike'
    "bike": 6,                             # Handling both just in case
    "motorcycle": 7, # ACDC uses 'motorcycle', BDD uses 'motor'
    "motor": 7,
    "traffic light": 8, "traffic sign": 9
}

def convert_acdc_condition(condition):
    print(f"\n Processing ACDC Condition: {condition.upper()}...")
    
    json_path = os.path.join(LABEL_ROOT_DIR, condition, f"instancesonly_{condition}_val_gt_detection.json")
    
    if not os.path.exists(json_path):
        print(f" CRITICAL ERROR: Could not find file at: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build Dynamic ID Map 
    acdc_id_to_bdd_id = {}
    print("Mapping Categories found in JSON:")
    for cat in data['categories']:
        original_name = cat['name']
        original_id = cat['id']
        
        # Name Matching
        if original_name in BDD_NAME_TO_ID:
            target_id = BDD_NAME_TO_ID[original_name]
            acdc_id_to_bdd_id[original_id] = target_id
            print(f"  - '{original_name}' (ID {original_id}) -> BDD ID {target_id}")
        else:
            print(f"  - WARNING: Skipping category '{original_name}' (Not in BDD)")

    # 3. Index Images 
    print("Indexing local images...")
    image_search_dir = os.path.join(IMAGE_ROOT_DIR, condition, "val")
    path_map = {}
    for p in Path(image_search_dir).rglob('*.png'):
        rel_path = str(p.relative_to(IMAGE_ROOT_DIR)).replace("\\", "/")
        path_map[p.name] = rel_path

    # 4. Construct Output JSON
    out_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": v, "name": k} for k, v in BDD_NAME_TO_ID.items() if k in ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]]
    }

    # Map Images
    valid_image_ids = set()
    for img in data['images']:
        fname = img['file_name']
        fname_base = os.path.basename(fname)
        
        if fname_base in path_map:
            new_img = img.copy()
            new_img['file_name'] = path_map[fname_base]
            out_json['images'].append(new_img)
            valid_image_ids.add(img['id'])
        else:
            pass
            
    print(f"Matched {len(valid_image_ids)} / {len(data['images'])} images.")

    # Map Annotations
    for ann in data['annotations']:
        if ann['image_id'] not in valid_image_ids:
            continue
            
        old_cat = ann['category_id']
        if old_cat in acdc_id_to_bdd_id:
            new_ann = ann.copy()
            new_ann['category_id'] = acdc_id_to_bdd_id[old_cat]
            out_json['annotations'].append(new_ann)

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"acdc_{condition}_bdd.json")
    with open(out_path, 'w') as f:
        json.dump(out_json, f)
    print(f" Saved: {out_path}")

if __name__ == "__main__":
    # Process all 4 conditions
    for cond in ['fog', 'night', 'rain', 'snow']:
        convert_acdc_condition(cond)