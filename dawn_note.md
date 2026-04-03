This is the final "Data Engineering" step to ensure your research paper has a professional, multi-domain breakdown. 

Here is how you handle the paths and the per-weather evaluation.

---

### 1. The Experiment File Configuration

In your `test_dawn.py` (or whatever you named the DAWN eval exp file), point the paths like this. **Do not use subfolders like `\images` if your JSON paths already include the weather names.**

```python
# In your test_dawn.py or tde_yolox_s.py (for eval)
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ... other settings ...
        
        # 1. POINT TO THE DAWN ROOT
        # Because the JSON file_names are 'Fog/Fog/001.jpg', 
        # YOLOX will look for D:\DAWN\Fog\Fog\001.jpg
        self.data_dir = r"D:\DAWN" 
        
        # 2. POINT TO YOUR GENERATED JSON
        self.val_ann = r"D:\DAWN\dawn_adverse_coco.json"
        self.test_ann = r"D:\DAWN\dawn_adverse_coco.json"
        
        self.num_classes = 9 # Keep this synced with your BDD training
```

---

### 2. The Per-Weather JSON Splitter

To get the breakdown (e.g., mAP for just Fog), run this script. It reads your master DAWN JSON and generates four specialized versions. This allows you to prove your **TTT-Stage 1** clears Fog and your **Engram Head** restores identities in Snow separately.

**Save as `split_dawn_weather.py`:**

```python
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
```

---

### 3. How to Run the "Forensic" Evaluation

Once you run the splitter, you will have 4 files in `D:\DAWN\weather_splits\`. 

To get the results for your paper's "Adverse Weather Performance Table," run the `eval.py` command four times, overriding the JSON path on the command line using the `-opts` argument.

**Example Command (For Fog):**
```bash
python tools/eval.py -f exps/default/test_dawn.py -c YOLOX_outputs/tde_yolox_s/latest_ckpt.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse val_ann D:/DAWN/weather_splits/dawn_fog.json
```

**Example Command (For Snow):**
```bash
python tools/eval.py -f exps/default/test_dawn.py -c YOLOX_outputs/tde_yolox_s/latest_ckpt.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse val_ann D:/DAWN/weather_splits/dawn_snow.json
```

### Why this is the "Expert" approach:
1.  **Scientific Depth:** You can now see if your **Physics Cocktail** (Contrast/Bias/Noise) is equally effective across all types.
2.  **No Data Duplication:** You aren't moving images around; you are just changing the "filter" (the JSON) through which the model sees the data.
3.  **Traceability:** If a specific class (like "Motorcycle") is failing in "Sand," you can quickly find the images using the specific `dawn_sand.json` to see what's happening.

**Proceed with splitting the JSON and running the individual evals. This is the final step of the natural OOD validation.**