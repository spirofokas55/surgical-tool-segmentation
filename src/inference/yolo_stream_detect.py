from ultralytics import YOLO
import cv2
import random
import time
import os
import numpy as np

# ========= CONFIG =========
webcamIndex = 0
output_path = "output_detected.mp4"
confidence_threshold = 0.7
desired_fps = 30
imgsz = 768

SNAPSHOT_PATH = "detected_object.png"
SNAPSHOT_COOLDOWN_SEC = 5.0   # saves at most once per second while target is visible

# ---- User command (typed once at start) ----
command = input("Type command (example: detect scalpel): ").strip().lower()
target = ""
if command.startswith("detect"):
    target = command.replace("detect", "", 1).strip()
else:
    print("Command must start with 'detect'. Example: detect scalpel")
    raise SystemExit

if not target:
    print("No target provided. Example: detect scalpel")
    raise SystemExit

print(f"Will detect any label containing: '{target}'")

# Load YOLO model
yolo = YOLO("best.pt")  # your segmentation model

def mask_centroid(mask_u8: np.ndarray):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

# Open webcam
cap = cv2.VideoCapture(webcamIndex)
if not cap.isOpened():
    raise Exception("Could not open video source")

# Get frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# If the camera didn't report size, read one frame to get it
if frame_width == 0 or frame_height == 0:
    ret, tmp = cap.read()
    if not ret:
        raise Exception("Failed to read a frame for size detection")
    frame_height, frame_width = tmp.shape[:2]

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, desired_fps, (frame_width, frame_height))

# OpenCV window
win_name = "YOLO Tracking"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Force the window to appear immediately (prevents "only appears later")
dummy = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.imshow(win_name, dummy)
cv2.waitKey(1)

frame_delay = 1.0 / desired_fps
prev_frame_time = 0.0
last_snapshot_time = 0.0

print("Tracking started... Press 'q' to quit.")
print(f"Live target: '{target}' (mask centroid)")
print(f"Snapshot overwrites: {os.path.abspath(SNAPSHOT_PATH)}")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected or end of feed.")
        break

    # FPS control
    now = time.time()
    if now - prev_frame_time < frame_delay:
        # Still show something so the window stays responsive
        cv2.imshow(win_name, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            print("Stopping.")
            break
        continue
    prev_frame_time = now

    # YOLO detection
    results = yolo.predict(source=frame, imgsz=imgsz, conf=confidence_threshold, verbose=False)
    r = results[0]

    # Plot (boxes + masks) (live feed stays live)
    vis = r.plot().copy()

    names_map = getattr(r, "names", None) or getattr(yolo, "names", None) or {}

    # Look for a matching class name and compute centroid
    if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
        H, W = frame.shape[:2]

        masks = r.masks.data.cpu().numpy()               # (N, h, w)
        clss  = r.boxes.cls.cpu().numpy().astype(int)    # (N,)
        confs = r.boxes.conf.cpu().numpy()               # (N,)

        best = None  # (conf, cls_name, centroid)

        for i in range(len(clss)):
            cls_id = int(clss[i])
            cls_name = str(names_map.get(cls_id, cls_id)).lower()
            conf = float(confs[i])

            if target in cls_name:
                mask = masks[i].astype(np.uint8)
                if mask.shape[:2] != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                c = mask_centroid(mask)
                if c is None:
                    continue

                if best is None or conf > best[0]:
                    best = (conf, cls_name, c)

        if best is not None:
            conf, cls_name, (cx, cy) = best

            # Draw centroid on live view (optional)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)

            # Save snapshot (throttled) WITHOUT freezing window
            if (time.time() - last_snapshot_time) >= SNAPSHOT_COOLDOWN_SEC:
                cv2.imwrite(SNAPSHOT_PATH, vis)
                last_snapshot_time = time.time()
                print(f"[SNAPSHOT] matched '{cls_name}' centroid=({cx},{cy}) conf={conf:.2f}")
                print(f"          saved -> {os.path.abspath(SNAPSHOT_PATH)}")

    # Show live output continuously
    cv2.imshow(win_name, vis)

    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Stopping.")
        break

    # Save video
    out.write(vis)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete! Output saved to: {os.path.abspath(output_path)}")