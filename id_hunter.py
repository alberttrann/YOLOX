import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Point this to the top-most DAWN folder
DAWN_ROOT = "D:/DAWN"
# We'll check for all IDs to confirm the script is working
TARGET_IDS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

def find_elusive_ids():
    print(f"Scanning all folders under {DAWN_ROOT} for labels...")
    
    # Initialize storage
    found_files = {tid: [] for tid in TARGET_IDS}
    
    # 1. Find all .txt files recursively
    all_txt_files = list(Path(DAWN_ROOT).rglob("*.txt"))
    print(f"Found {len(all_txt_files)} total .txt files. Analyzing contents...")

    for txt_file in tqdm(all_txt_files):
        # Ignore common non-label files
        if txt_file.name in ["classes.txt", "notes.txt"]:
            continue
            
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    id_val = parts[0]
                    if id_val in found_files:
                        # Convert absolute path to a readable relative path for you
                        rel_path = txt_file.relative_to(DAWN_ROOT)
                        found_files[id_val].append(rel_path)
                        # We only need to register the file once per ID
                        # but check if other IDs are in the same file
        except Exception as e:
            continue

    # 2. Report Results
    print("\n" + "="*50)
    print("DAWN ID SCAN RESULTS")
    print("="*50)
    
    for tid in TARGET_IDS:
        paths = list(set(found_files[tid])) # Unique files
        print(f"\nID {tid}: Found in {len(paths)} files")
        if paths:
            print("First 3 examples to inspect:")
            for p in paths[:3]:
                # Attempt to find the matching .jpg
                img_name = p.stem + ".jpg"
                # Search the parent's sibling or parent's parent for the image
                print(f"  - Label: {p}")
                print(f"    Check for image: {img_name} in the adjacent 'Fog', 'Rain', etc. folder")

if __name__ == "__main__":
    find_elusive_ids()