import os
import cv2
from pathlib import Path
from ultralytics import RTDETR

# Must match training order
CLASS_NAMES = {0: "fire", 1: "firearm", 2: "grenade", 3: "knife", 4: "pistol", 5: "rocket"}

MODEL_PATH = "models/rtdetr_weapon.pt"
RAW_DIR = Path("datasets/raw")
LABELS_DIR = Path("datasets/labels")

CONF_THRESHOLD = 0.4

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h

def main():
    print("[Auto-Label] Loading model...")
    model = RTDETR(MODEL_PATH)

    total = 0
    saved = 0

    for class_folder in RAW_DIR.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        print(f"\n[CLASS] {class_name}")

        for img_path in class_folder.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            total += 1

            results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)
            detections = []

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue
                    detections.append((cls_id, conf, box.xyxy[0].tolist()))

            if not detections:
                print(f"  [SKIP] No detection for {img_path.name}")
                continue

            # take highest confidence
            detections.sort(key=lambda x: x[1], reverse=True)
            cls_id, conf, xyxy = detections[0]

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)

            label_dir = LABELS_DIR / CLASS_NAMES[cls_id]
            label_dir.mkdir(parents=True, exist_ok=True)

            label_file = label_dir / f"{img_path.stem}.txt"
            with open(label_file, "w") as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            print(f"  [OK] {img_path.name} → {label_file.name}")
            saved += 1

    print(f"\nDone. {saved}/{total} images labeled.")

if __name__ == "__main__":
    main()