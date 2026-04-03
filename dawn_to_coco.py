import json
import os
import cv2
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
DAWN_ROOT = "D:/DAWN"
OUTPUT_JSON = "D:/DAWN/dawn_adverse_coco.json"

# FINAL FORENSIC REMAP (Verified by foggy-001, 002, 007 and rain-297)
REMAP = {
    "3": 1, # car
    "6": 2, # bus
    "1": 3, # truck
    "2": 6, # bike
    "4": 7, # motor
    "8": 9  # traffic sign
}

CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bus"},
    {"id": 3, "name": "truck"},
    {"id": 6, "name": "bike"},
    {"id": 7, "name": "motor"},
    {"id": 9, "name": "traffic sign"}
]

def convert():
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    
    # 1. Global Indexing: Find every JPG and every TXT
    print(f"Scanning {DAWN_ROOT} for files...")
    all_jpgs = {p.stem: p for p in Path(DAWN_ROOT).rglob("*.jpg")}
    all_txts = {p.stem: p for p in Path(DAWN_ROOT).rglob("*.txt") if "_YOLO_darknet" in str(p)}

    # 2. Pairing logic
    common_stems = set(all_jpgs.keys()) & set(all_txts.keys())
    print(f"Found {len(all_jpgs)} images and {len(all_txts)} label files.")
    print(f"Matched {len(common_stems)} valid image-label pairs.")

    img_id, ann_id = 1, 1

    for stem in tqdm(sorted(list(common_stems)), desc="Converting"):
        img_path = all_jpgs[stem]
        label_path = all_txts[stem]

        # Get Dimensions
        img_cv = cv2.imread(str(img_path))
        if img_cv is None: continue
        h, w, _ = img_cv.shape

        # Register Image
        # file_name is relative to the DAWN_ROOT for YOLOX compatibility
        rel_path = img_path.relative_to(DAWN_ROOT).as_posix()
        
        # Determine weather from parent folder
        weather_label = "unknown"
        for w_type in ["fog", "rain", "sand", "snow"]:
            if w_type in str(img_path).lower():
                weather_label = w_type
                break

        coco["images"].append({
            "id": img_id,
            "file_name": rel_path,
            "width": w,
            "height": h,
            "weather": weather_label
        })

        # Process YOLO Labels
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                d_id = parts[0]
                if d_id not in REMAP: continue
                
                # YOLO Normalized -> COCO Absolute
                xc, yc, bw, bh = map(float, parts[1:5])
                width = bw * w
                height = bh * h
                xmin = (xc - bw / 2) * w
                ymin = (yc - bh / 2) * h

                # Bound check for robustness
                xmin, ymin = max(0, xmin), max(0, ymin)
                width = min(width, w - xmin)
                height = min(height, h - ymin)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": REMAP[d_id],
                    "bbox": [round(xmin, 2), round(ymin, 2), round(width, 2), round(height, 2)],
                    "area": round(float(width * height), 2),
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1

    # 3. Final Write
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco, f)
    
    print("\n" + "="*40)
    print("CONVERSION SUMMARY")
    print("="*40)
    print(f"Total Images: {len(coco['images'])}")
    print(f"Total Annotations: {len(coco['annotations'])}")
    print(f"JSON Saved to: {OUTPUT_JSON}")
    print("="*40)

if __name__ == "__main__":
    convert()