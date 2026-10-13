import json
import os

MASTER_JSON = r"D:\DAWN\dawn_adverse_coco.json"
OUTPUT_DIR = r"D:\DAWN\weather_splits"

def split_weather():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(MASTER_JSON, 'r') as f:
        data = json.load(f)

    weather_types = ["fog", "rain", "snow", "sand"]
    
    for target in weather_types:
        new_coco = {
            "images": [],
            "annotations": [],
            "categories": data["categories"]  # Preserves all 9 categories
        }
        
        target_img_ids = set()
        for img in data["images"]:
            if img["weather"] == target:
                new_coco["images"].append(img)
                target_img_ids.add(img["id"])
        
        for ann in data["annotations"]:
            if ann["image_id"] in target_img_ids:
                new_coco["annotations"].append(ann)
        
        output_path = os.path.join(OUTPUT_DIR, f"dawn_{target}.json")
        with open(output_path, 'w') as f:
            json.dump(new_coco, f)
        print(f"Saved: {output_path} ({len(new_coco['images'])} images, {len(new_coco['annotations'])} annotations)")

if __name__ == "__main__":
    split_weather()