import cv2
import numpy as np
from ultralytics import YOLO

# ── SEVERITY GRADING ─────────────────────────────────────
def get_severity(confidence: float):
    """Return (severity_label, BGR_color) based on confidence score."""
    if confidence >= 0.80:
        return "HIGH",   (0, 0, 255)      # Red
    elif confidence >= 0.50:
        return "MEDIUM", (0, 165, 255)    # Orange
    else:
        return "LOW",    (0, 255, 255)    # Yellow

# ── MAIN DETECTION FUNCTION ──────────────────────────────
def detect_and_annotate(image_path: str):
    """
    Run YOLOv8 inference on *image_path*, draw colour-coded bounding boxes,
    add a summary banner, and return (annotated_rgb_ndarray, report_list).

    Parameters
    ----------
    image_path : str
        Absolute path to the input image.

    Returns
    -------
    img_rgb : np.ndarray  (H, W, 3)  uint8  — RGB for Streamlit / PIL display
    report  : list[dict]              — one dict per detection
    """
    # 1. Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]

    # 2. Load & run YOLOv8
    model = _get_model()
    results = model(image_path, verbose=False)
    detections = results[0].boxes

    report = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id   = int(box.cls[0])
        label      = model.names[class_id]
        severity, color = get_severity(confidence)

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Filled label background
        text = f"{label} | {severity} | {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(y1 - th - 12, 0)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        report.append({
            "object":     label,
            "severity":   severity,
            "confidence": round(confidence, 4),
            "location":   f"({x1},{y1}) → ({x2},{y2})"
        })

    # 3. Summary banner at bottom
    high   = sum(1 for d in report if d["severity"] == "HIGH")
    medium = sum(1 for d in report if d["severity"] == "MEDIUM")
    low    = sum(1 for d in report if d["severity"] == "LOW")
    banner = (f"Total: {len(report)}  |  "
              f"HIGH: {high}  MEDIUM: {medium}  LOW: {low}")

    banner_h = 44
    cv2.rectangle(img, (0, height - banner_h), (width, height), (20, 20, 20), -1)
    cv2.putText(img, banner, (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)

    # 4. BGR → RGB for Streamlit / PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, report


# ── MODEL LOADER (singleton-like) ────────────────────────
_yolo_model = None

def _get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("neu_defect_best.pt")
    return _yolo_model


# ── STANDALONE TEST ──────────────────────────────────────
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    img_rgb, report = detect_and_annotate(image_path)

    # Save result
    cv2.imwrite("output.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"\n✅ Saved annotated image → output.jpg")

    print("\n📋 DETECTION REPORT")
    print("=" * 55)
    for i, d in enumerate(report):
        print(f"\nDetection #{i+1}")
        print(f"  Object     : {d['object']}")
        print(f"  Severity   : {d['severity']}")
        print(f"  Confidence : {d['confidence']:.4f}")
        print(f"  Location   : {d['location']}")
    print("\n" + "=" * 55)
    print(f"Total: {len(report)} detection(s)")