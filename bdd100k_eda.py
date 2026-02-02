import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

def perform_bdd100k_eda(json_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Analyzing {len(data)} images...")
    
    image_metadata = []
    all_categories = []
    object_attributes = {
        "occluded": [],
        "truncated": [],
        "trafficLightColor": []
    }
    box_dimensions = [] # To check for small object distribution (relevant for DSA)

    for img in tqdm(data):
        # 1. Image Level Metadata
        attr = img.get('attributes', {})
        image_metadata.append({
            "name": img['name'],
            "weather": attr.get('weather', 'unknown'),
            "scene": attr.get('scene', 'unknown'),
            "timeofday": attr.get('timeofday', 'unknown'),
            "label_count": len(img.get('labels', []))
        })

        # 2. Label Level Data
        for label in img.get('labels', []):
            cat = label['category']
            all_categories.append(cat)
            
            # Focus on 2D Bounding Boxes (ignore drivable area/lanes for detection stats)
            if 'box2d' in label:
                box = label['box2d']
                w = box['x2'] - box['x1']
                h = box['y2'] - box['y1']
                box_dimensions.append({"w": w, "h": h, "category": cat})

                # Attributes
                l_attr = label.get('attributes', {})
                object_attributes['occluded'].append(l_attr.get('occluded', False))
                object_attributes['truncated'].append(l_attr.get('truncated', False))
                if cat == "traffic light":
                    object_attributes['trafficLightColor'].append(l_attr.get('trafficLightColor', 'none'))

    # Convert to DataFrames
    df_img = pd.DataFrame(image_metadata)
    df_boxes = pd.DataFrame(box_dimensions)

    # --- REPORTING ---
    
    print("\n" + "="*30)
    print("GLOBAL IMAGE METADATA")
    print("="*30)
    for field in ['weather', 'scene', 'timeofday']:
        print(f"\nDistribution for {field.upper()}:")
        print(df_img[field].value_counts())

    print("\n" + "="*30)
    print("OBJECT CATEGORIES (DETECTION)")
    print("="*30)
    cat_counts = Counter(all_categories)
    for cat, count in cat_counts.most_common():
        print(f"{cat}: {count}")

    # --- OOD VALIDATION FOR YOUR RESEARCH PLAN ---
    train_weather = ['clear', 'overcast', 'partly cloudy']
    test_weather = ['rainy', 'snowy', 'undefined']
    
    train_count = df_img[df_img['weather'].isin(train_weather)].shape[0]
    test_count = df_img[df_img['weather'].isin(test_weather)].shape[0]
    fog_count = df_img[df_img['weather'] == 'foggy'].shape[0]

    print("\n" + "="*30)
    print("OOD SPLIT CHECK")
    print("="*30)
    print(f"Potential 'Clean' Training Images ({train_weather}): {train_count}")
    print(f"Potential 'Adverse' Testing Images ({test_weather}): {test_count}")
    print(f"There are {fog_count} 'foggy' images")

    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 10))
    
    # Weather Plot
    plt.subplot(2, 2, 1)
    sns.countplot(data=df_img, x='weather')
    plt.title("Weather Distribution")
    plt.xticks(rotation=45)

    # Category Plot
    plt.subplot(2, 2, 2)
    sns.barplot(x=list(cat_counts.values()), y=list(cat_counts.keys()))
    plt.title("Object Category Distribution")

    # Box Size Distribution (Scatter)
    plt.subplot(2, 2, 3)
    sns.histplot(data=df_boxes, x='w', y='h', bins=50)
    plt.title("Box Size Density (Width vs Height)")
    
    plt.tight_layout()
    plt.show()

perform_bdd100k_eda('C:/TDE-YOLOX/bdd100k_labels_images_val.json')