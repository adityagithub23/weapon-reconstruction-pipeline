import os
import shutil
from ultralytics import RTDETR

# ---------------- SETTINGS ----------------

DATASET_DIR = "Data"
SPLITS = ["train", "test", "valid"]

OUTPUT_DIR = "datasets/raw"
MODEL_PATH = "models/rtdetr_weapon.pt"

CONF_THRESHOLD = 0.25

CLASS_NAMES = {
    0: "fire",
    1: "firearm",
    2: "grenade",
    3: "knife",
    4: "pistol",
    5: "rocket"
}

# ------------------------------------------

print("\nLoading RT-DETR model...\n")
model = RTDETR(MODEL_PATH)

# create output folders
for cls in CLASS_NAMES.values():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# image counters
class_counts = {cls: 0 for cls in CLASS_NAMES.values()}

total_images = 0

for split in SPLITS:

    img_folder = os.path.join(DATASET_DIR, split, "images")

    if not os.path.exists(img_folder):
        continue

    print(f"\nScanning {img_folder}\n")

    images = [f for f in os.listdir(img_folder)
              if f.lower().endswith((".jpg",".jpeg",".png"))]

    for img in images:

        total_images += 1

        img_path = os.path.join(img_folder, img)

        results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)

        detected_class = None
        best_conf = 0

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > best_conf:
                    best_conf = conf
                    detected_class = CLASS_NAMES.get(cls)

        if detected_class is None:
            print(f"⚠ No detection → {img}")
            continue

        class_counts[detected_class] += 1

        ext = os.path.splitext(img)[1]

        new_name = f"{detected_class}{class_counts[detected_class]}{ext}"

        dest_path = os.path.join(OUTPUT_DIR, detected_class, new_name)

        shutil.copy(img_path, dest_path)

        print(f"{img} → datasets/raw/{detected_class}/{new_name}")

print("\n-----------------------------------")
print(f"Processed {total_images} images")
print("Dataset sorting complete\n")