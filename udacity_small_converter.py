import json
import os
from tqdm import tqdm
from pathlib import Path

UDACITY_JSON_PATH = r"D:\Self Driving Car.v3-fixed-small.coco\export\_annotations.coco.json"
IMAGE_ROOT_DIR = r"D:\Self Driving Car.v3-fixed-small.coco\export" 
OUTPUT_JSON_PATH = r"D:\YOLOX-3RD\udacity_small_bdd_compatible.json"

# BDD100K Class Map 
BDD_CATEGORIES = [
    {"id": 1, "name": "car"}, {"id": 2, "name": "bus"}, {"id": 3, "name": "truck"},
    {"id": 4, "name": "person"}, {"id": 5, "name": "rider"}, {"id": 6, "name": "bike"},
    {"id": 7, "name": "motor"}, {"id": 8, "name": "traffic light"}, {"id": 9, "name": "traffic sign"}
]

# Udacity ID -> BDD ID Mapper
MAPPER = {
    1: 6, 2: 1, 3: 4, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8, 11: 3
}

def convert_udacity_small():
    print(f" Loading Udacity v3 Small JSON...")
    if not os.path.exists(UDACITY_JSON_PATH):
        print(f"FATAL: File not found at {UDACITY_JSON_PATH}")
        return

    with open(UDACITY_JSON_PATH, 'r') as f:
        data = json.load(f)

    out_json = {
        "images": [],
        "annotations": [],
        "categories": BDD_CATEGORIES
    }
    
    print("Indexing local images...")
    local_files = {p.name: str(p.relative_to(IMAGE_ROOT_DIR)).replace("\\", "/") 
                   for p in Path(IMAGE_ROOT_DIR).rglob('*.jpg')}
    
    # Process Images
    print(f"Found {len(local_files)} images on disk. Matching with JSON...")
    valid_image_ids = {}
    
    for img in data['images']:
        json_fname = img['file_name']
        fname_base = os.path.basename(json_fname)
        
        if fname_base in local_files:
            out_json['images'].append({
                "id": img['id'],
                "file_name": local_files[fname_base], 
                "width": img.get('width', 512),
                "height": img.get('height', 512)
            })
            valid_image_ids[img['id']] = True
        else:
            if len(valid_image_ids) < 5:
                print(f"Skipping: {fname_base} (Not found in {IMAGE_ROOT_DIR})")

    print(f"Successfully matched {len(out_json['images'])} images.")

    # Process Annotations
    print("Mapping categories...")
    mapped_count = 0
    
    for ann in tqdm(data['annotations']):
        if ann['image_id'] not in valid_image_ids:
            continue
            
        u_cat = ann['category_id']
        if u_cat in MAPPER:
            new_ann = ann.copy()
            new_ann['id'] = len(out_json['annotations']) 
            new_ann['category_id'] = MAPPER[u_cat]
            out_json['annotations'].append(new_ann)
            mapped_count += 1

    print(f"Saving to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(out_json, f)

    print(f" DONE! Mapped {mapped_count} annotations for {len(out_json['images'])} images.")

if __name__ == "__main__":
    convert_udacity_small()