import json, os
from pathlib import Path

ROOT_DIR = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/annotations"
IMAGE_ROOT = "D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images"

CATEGORY_MAP = {
    "car": 1, "bus": 2, "truck": 3, "person": 4, 
    "rider": 5, "bike": 6, "motor": 7, 
    "traffic light": 8, "traffic sign": 9
}
TEST_WEATHER = ['rainy', 'snowy', 'foggy']

def get_coco_skeleton():
    return {"images": [], "annotations": [], "categories": [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()]}

def build_split(bdd_json_path, out_clean_path, out_adverse_path=None):
    with open(bdd_json_path, 'r') as f:
        bdd_data = json.load(f)
    
    clean_coco = get_coco_skeleton()
    adverse_coco = get_coco_skeleton() if out_adverse_path else None
    
    img_id = 1
    ann_id = 1
    for entry in bdd_data:
        weather = entry.get('attributes', {}).get('weather', 'unknown').lower()
        is_adverse = weather in TEST_WEATHER
        
        target = adverse_coco if (is_adverse and out_adverse_path) else clean_coco
        if is_adverse and not out_adverse_path:
            continue
            
        target['images'].append({
            "id": img_id,
            "file_name": entry['name'],
            "width": 1280, "height": 720, "weather": weather
        })
        for label in entry.get('labels', []):
            cat = label['category']
            if cat in CATEGORY_MAP and 'box2d' in label:
                box = label['box2d']
                w = box['x2'] - box['x1']
                h = box['y2'] - box['y1']
                if w > 1 and h > 1:
                    target['annotations'].append({
                        "id": ann_id, "image_id": img_id,
                        "category_id": CATEGORY_MAP[cat],
                        "bbox": [box['x1'], box['y1'], w, h],
                        "area": float(w * h), "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1
        
    with open(out_clean_path, 'w') as f:
        json.dump(clean_coco, f)
    if out_adverse_path:
        with open(out_adverse_path, 'w') as f:
            json.dump(adverse_coco, f)

# Generate distinct splits:
build_split("D:/YOLOX-3RD/bdd100k_labels_images_train.json", os.path.join(ROOT_DIR, "tde_train_clean_coco.json"))
build_split("D:/YOLOX-3RD/bdd100k_labels_images_val.json", os.path.join(ROOT_DIR, "tde_val_clean_coco.json"), os.path.join(ROOT_DIR, "tde_test_adverse_coco.json"))
print("ZSDA splits successfully created!")