import cv2
import argparse
from ultralytics import YOLO
from pathlib import Path


# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "best.pt"       # ✏️ path to your trained weights
CONF        = 0.25            # confidence threshold
IOU         = 0.45            # NMS IoU threshold
IMG_SIZE    = 640             # must match training size
SAVE_OUTPUT = False            # save annotated output to disk
OUTPUT_DIR  = "outputs"       # folder to save results
# ──────────────────────────────────────────────────────────────────────────────


def load_model(weights: str) -> YOLO:
    print(f"[INFO] Loading model from: {weights}")
    model = YOLO(weights)
    print(f"[INFO] Classes: {model.names}")
    return model


def predict_image(model: YOLO, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    results = model.predict(
        source  = image_path,
        conf    = CONF,
        iou     = IOU,
        imgsz   = IMG_SIZE,
        verbose = False,
    )[0]

    # Draw boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf)
        cls   = int(box.cls)
        label = f"{model.names[cls]} {conf:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 220), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    count = len(results.boxes)
    print(f"[RESULT] {count} pothole(s) detected in {image_path}")

    if SAVE_OUTPUT:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        out_path = str(Path(OUTPUT_DIR) / Path(image_path).name)
        cv2.imwrite(out_path, img)
        print(f"[SAVED]  {out_path}")

    cv2.imshow("Pothole Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_video(model: YOLO, video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if SAVE_OUTPUT:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        out_path = str(Path(OUTPUT_DIR) / Path(video_path).stem) + "_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO]  Saving output to: {out_path}")

    frame_idx = 0
    print("[INFO] Running inference on video... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source  = frame,
            conf    = CONF,
            iou     = IOU,
            imgsz   = IMG_SIZE,
            verbose = False,
        )[0]

        # Draw boxes on frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf)
            cls   = int(box.cls)
            label = f"{model.names[cls]} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 220), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_idx}  Potholes: {len(results.boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Pothole Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped by user.")
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_idx} frames.")


def predict_webcam(model: YOLO, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index: {cam_index}")
        return

    print("[INFO] Webcam running... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source  = frame,
            conf    = CONF,
            iou     = IOU,
            imgsz   = IMG_SIZE,
            verbose = False,
        )[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf)
            cls   = int(box.cls)
            label = f"{model.names[cls]} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 220), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Potholes: {len(results.boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pothole Detection — Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pothole Detection — YOLOv8")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to image or video file (omit for webcam)")
    parser.add_argument("--weights", type=str, default=MODEL_PATH,
                        help="Path to model weights (default: best.pt)")
    parser.add_argument("--conf", type=float, default=CONF,
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    CONF = args.conf
    model = load_model(args.weights)

    if args.source is None:
        # No source → use webcam
        predict_webcam(model, cam_index=0)

    else:
        ext = Path(args.source).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            predict_image(model, args.source)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            predict_video(model, args.source)
        else:
            print(f"[ERROR] Unsupported file type: {ext}")