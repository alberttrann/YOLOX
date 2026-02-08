import json
import os
import cv2
from tqdm import tqdm
from pathlib import Path

DAWN_ROOT = r"D:\DAWN"
IMAGE_DIR = os.path.join(DAWN_ROOT, "images")
LABEL_DIR = os.path.join(DAWN_ROOT, "labels")
OUTPUT_JSON = r"D:\YOLOX-3rd\dawn_bdd_compatible.json"

# Target BDD Categories
BDD_CATEGORIES = [
    {"id": 1, "name": "car"}, {"id": 2, "name": "bus"}, {"id": 3, "name": "truck"},
    {"id": 4, "name": "person"}, {"id": 5, "name": "rider"}, {"id": 6, "name": "bike"},
    {"id": 7, "name": "motor"}, {"id": 8, "name": "traffic light"}, {"id": 9, "name": "traffic sign"}
]

# DAWN (YOLO) -> BDD ID Map
DAWN_MAP = {
    1: 4, # person -> person (BDD 4)
    3: 1, # car -> car (BDD 1)
    6: 3, # truck/van -> truck (BDD 3)
    # The following are inferred based on the 1/3/6 sequence
    0: 6, # bicycle -> bike (BDD 6)
    2: 7, # motorcycle -> motor (BDD 7)
    4: 2, # bus -> bus (BDD 2)
    5: 3  # truck -> truck (BDD 3)
}

def convert_dawn():
    print(f" Converting DAWN Dataset to BDD-Compatible COCO...")
    
    out_json = {
        "images": [],
        "annotations": [],
        "categories": BDD_CATEGORIES
    }
    
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    img_id = 0
    ann_id = 0
    
    for label_name in tqdm(label_files):
        # 1. Find corresponding image
        base_name = os.path.splitext(label_name)[0]
        # Search for common extensions
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            test_path = os.path.join(IMAGE_DIR, base_name + ext)
            if os.path.exists(test_path):
                img_file = base_name + ext
                break
        
        if img_file is None: continue
        
        # 2. Get image dimensions () YOLO -> COCO)
        img_path = os.path.join(IMAGE_DIR, img_file)
        im = cv2.imread(img_path)
        if im is None: continue
        h, w, _ = im.shape
        
        out_json["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": w,
            "height": h
        })
        
        # 3. Parse YOLO labels
        with open(os.path.join(LABEL_DIR, label_name), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            cls_id = int(float(parts[0]))
            # Center Coordinates (Normalized)
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
            
            # Convert to COCO [x_min, y_min, width, height]
            coco_w = bw * w
            coco_h = bh * h
            coco_x = (cx * w) - (coco_w / 2)
            coco_y = (cy * h) - (coco_h / 2)
            
            if cls_id in DAWN_MAP:
                out_json["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": DAWN_MAP[cls_id],
                    "bbox": [coco_x, coco_y, coco_w, coco_h],
                    "area": float(coco_w * coco_h),
                    "iscrowd": 0
                })
                ann_id += 1
                
        img_id += 1

    print(f"Saving JSON to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f" DONE! Converted {img_id} images and {ann_id} annotations.")

if __name__ == "__main__":
    convert_dawn()