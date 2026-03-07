import cv2
from ultralytics import RTDETR

model = RTDETR("models/rtdetr_weapon.pt")

input_path  = "datasets/raw/firearm/vid1.mp4"
output_path = "output/vid1_detected.mp4"

cap = cv2.VideoCapture(input_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])

            label = f"{cls} {conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()

print("Done. Saved to:", output_path)