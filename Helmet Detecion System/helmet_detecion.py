import cv2
import math
import time
import cvzone
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "Media/bike_3.mp4"
WEIGHTS_PATH = "Weights/best.pt"
CONF_THRESH  = 0.18   # lower → more recall (catches more unhelmeted riders)
                       # raise  → more precision (fewer false positives)
USE_WEBCAM   = False   # set True to use live camera instead of video file
WEBCAM_ID    = 0

CLASS_NAMES = ["With Helmet", "Without Helmet"]

# Per-class colors in BGR: green for safe, red for violation
CLASS_COLORS = {
    "With Helmet":    (0, 200, 0),    # green
    "Without Helmet": (0, 0, 220),    # red
}
# ───────────────────────────────────────────────────────────────────────────────


def draw_overlay(img, fps, counts):
    """Draw semi-transparent HUD with FPS and per-class counts."""
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (260, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, f"FPS: {fps:.1f}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(img, f"With Helmet:    {counts['With Helmet']}",
                (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS["With Helmet"], 2)
    cv2.putText(img, f"Without Helmet: {counts['Without Helmet']}",
                (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS["Without Helmet"], 2)


def main():
    source = WEBCAM_ID if USE_WEBCAM else VIDEO_PATH
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        return

    model = YOLO(WEIGHTS_PATH)
    print(f"[INFO] Model loaded. Press 'q' to quit.")

    prev_time = time.time()

    try:
        while True:
            success, img = cap.read()
            if not success:
                # End of video file — loop back, or break for webcam
                if USE_WEBCAM:
                    print("[WARN] Failed to grab frame from webcam.")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
                continue

            counts = {"With Helmet": 0, "Without Helmet": 0}

            results = model(img, stream=True, conf=CONF_THRESH, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    cls  = int(box.cls[0])
                    name = CLASS_NAMES[cls]
                    conf = round(float(box.conf[0]), 2)
                    color = CLASS_COLORS[name]

                    counts[name] += 1

                    # Colored bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Corner decorations via cvzone (uses its own color arg)
                    cvzone.cornerRect(img, (x1, y1, w, h),
                                      colorR=color, colorC=color, l=12, t=2)

                    # Label with class-matched color
                    cvzone.putTextRect(
                        img,
                        f"{name}  {conf:.2f}",
                        (max(0, x1), max(35, y1)),
                        scale=0.8, thickness=1,
                        colorR=color,
                        offset=5,
                    )

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            draw_overlay(img, fps, counts)

            cv2.imshow("Helmet Detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit requested.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released.")


if __name__ == "__main__":
    main()