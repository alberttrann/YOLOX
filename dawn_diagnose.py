import cv2
import os

DAWN_IMAGE = r"D:\DAWN\images\dusttornado-001.jpg" 
DAWN_LABEL = r"D:\DAWN\labels\dusttornado-001.txt"

def verify_dawn_ids():
    img = cv2.imread(DAWN_IMAGE)
    h, w, _ = img.shape

    with open(DAWN_LABEL, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls_id = parts[0]
        cx, cy, bw, bh = [float(x) for x in parts[1:]]

        # Convert to pixels
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID: {cls_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("DAWN ID Verification", img)
    print(f"Check the image. What object is labeled as ID {cls_id}?")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_dawn_ids()