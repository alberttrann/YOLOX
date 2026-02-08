import torch
import cv2
import os
import json
import numpy as np
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

DAWN_IMG_DIR = r"D:\DAWN\images"
DAWN_JSON = r"D:\YOLOX-3rd\dawn_bdd_compatible.json"
CKPT_PATH = r"YOLOX_outputs/tde_yolox_s/epoch_73_ckpt.pth"
EXP_FILE = r"exps/default/tde_yolox_s.py"

# BDD Class Names 
COCO_CLASSES = ("car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign")

def debug_dawn():
    # 1. Load Model
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    model.eval()
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()

    # 2. Load GT JSON
    with open(DAWN_JSON, 'r') as f:
        coco_gt = json.load(f)
    
    # Map Image ID to Annotations
    img_to_anns = {}
    for ann in coco_gt['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns: img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    # 3. Iterate through a few images
    # pick indices that likely have trucks/buses
    sample_indices = [0, 10, 25, 50, 100] 
    
    for i in sample_indices:
        img_info = coco_gt['images'][i]
        file_name = img_info['file_name']
        img_path = os.path.join(DAWN_IMG_DIR, file_name)
        
        # Read Image
        img = cv2.imread(img_path)
        if img is None: continue
        
        # --- GROUND TRUTH (GREEN) ---
        if img_info['id'] in img_to_anns:
            for ann in img_to_anns[img_info['id']]:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cls_id = ann['category_id'] # 1-based in JSON
                # Map 1-based ID to 0-based name index
                # BDD IDs: 1=car, 2=bus, 3=truck...
                # COCO_CLASSES index: 0=car, 1=bus, 2=truck...
                cls_name = COCO_CLASSES[cls_id - 1]
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"GT: {cls_name}", (x, y-5), 0, 0.5, (0, 255, 0), 2)

        # --- PREDICTION (RED) ---
        img_tensor, ratio = preprocess(img, (640, 640))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).cuda()
        
        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = postprocess(outputs, 9, 0.25, 0.45, class_agnostic=True) # Low conf to see all

        if outputs[0] is not None:
            output = outputs[0].cpu()
            bboxes = output[:, 0:4]
            bboxes /= ratio
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            
            for box, score, cls_id in zip(bboxes, scores, cls):
                x1, y1, x2, y2 = box.int().numpy()
                cls_name = COCO_CLASSES[int(cls_id)]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"PR: {cls_name} {score:.2f}", (x1, y1+15), 0, 0.5, (0, 0, 255), 2)

        cv2.imshow("DAWN Debug", img)
        key = cv2.waitKey(0)
        if key == 27: break 

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

if __name__ == "__main__":
    debug_dawn()