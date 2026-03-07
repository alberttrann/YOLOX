import json
import os

MASTER_JSON = r"D:\DAWN\dawn_adverse_coco.json"
OUTPUT_DIR = r"D:\DAWN\weather_splits"

def split_weather():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    with open(MASTER_JSON, 'r') as f:
        data = json.load(f)

    weather_types = ["fog", "rain", "snow", "sand"]
    
    for target in weather_types:
        print(f"Filtering for: {target.upper()}")
        
        # Create a new COCO structure
        new_coco = {
            "images": [],
            "annotations": [],
            "categories": data["categories"]
        }
        
        # 1. Find images for this weather
        target_img_ids = []
        for img in data["images"]:
            if img["weather"] == target:
                new_coco["images"].append(img)
                target_img_ids.append(img["id"])
        
        # 2. Find annotations for these images
        for ann in data["annotations"]:
            if ann["image_id"] in target_img_ids:
                new_coco["annotations"].append(ann)
        
        # 3. Save
        output_path = os.path.join(OUTPUT_DIR, f"dawn_{target}.json")
        with open(output_path, 'w') as f:
            json.dump(new_coco, f)
        print(f"  -> Saved {len(new_coco['images'])} images to {output_path}")

if __name__ == "__main__":
    split_weather()