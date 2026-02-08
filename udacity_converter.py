import json
import os
from tqdm import tqdm

UDACITY_JSON_PATH = r"D:\Self Driving Car.v2-fixed-large.coco\export\_annotations.coco.json"
OUTPUT_JSON_PATH = r"D:\YOLOX-3RD\udacity_bdd_compatible.json"
IMAGE_ROOT_DIR = r"D:\Self Driving Car.v2-fixed-large.coco\export" 

# BDD100K Class Map 
BDD_CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "truck"},
    {"id": 4, "name": "person"},
    {"id": 5, "name": "rider"},
    {"id": 6, "name": "bike"},
    {"id": 7, "name": "motor"},
    {"id": 8, "name": "traffic light"},
    {"id": 9, "name": "traffic sign"}
]

# Udacity ID -> BDD ID Mapper
# 0: obstacles -> IGNORE
MAPPER = {
    1: 6,  # biker -> bike
    2: 1,  # car -> car
    3: 4,  # pedestrian -> person
    4: 8,  # trafficLight -> traffic light
    5: 8,  # trafficLight-Green -> traffic light
    6: 8,  # trafficLight-GreenLeft -> traffic light
    7: 8,  # trafficLight-Red -> traffic light
    8: 8,  # trafficLight-RedLeft -> traffic light
    9: 8,  # trafficLight-Yellow -> traffic light
    10: 8, # trafficLight-YellowLeft -> traffic light
    11: 3  # truck -> truck
}

def convert_udacity():
    print(f"Loading {UDACITY_JSON_PATH}...")
    with open(UDACITY_JSON_PATH, 'r') as f:
        data = json.load(f)

    out_json = {
        "images": [],
        "annotations": [],
        "categories": BDD_CATEGORIES
    }
    
    # Process Images
    print("Processing images...")
    valid_image_ids = set()
    skipped_images = 0
    
    for img in data['images']:
        fname = img['file_name']
        w = img.get('width', 0)
        h = img.get('height', 0)
        
        # --- Filter corrupted metadata ---
        if w <= 0 or h <= 0:
            skipped_images += 1
            continue
            
        full_path = os.path.join(IMAGE_ROOT_DIR, fname)
        if not os.path.exists(full_path):
             skipped_images += 1
             continue

        out_json['images'].append({
            "id": img['id'],
            "file_name": fname,
            "width": w,
            "height": h
        })
        valid_image_ids.add(img['id'])
        
    print(f"Skipped {skipped_images} corrupted/missing images.")


    # Process Annotations
    print("Processing annotations...")
    mapped_count = 0
    skipped_count = 0
    
    for ann in tqdm(data['annotations']):
        if ann['image_id'] not in valid_image_ids:
            continue
            
        u_cat = ann['category_id']
        
        if u_cat in MAPPER:
            new_ann = ann.copy()
            new_ann['category_id'] = MAPPER[u_cat]
            out_json['annotations'].append(new_ann)
            mapped_count += 1
        else:
            skipped_count += 1

    print(f"Conversion Complete.")
    print(f"Mapped Annotations: {mapped_count}")
    print(f"Skipped Annotations: {skipped_count}")
    print(f"Saving to {OUTPUT_JSON_PATH}...")
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(out_json, f)

if __name__ == "__main__":
    convert_udacity()