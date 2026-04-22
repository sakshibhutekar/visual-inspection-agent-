import cv2
import numpy as np

# ── 1. READ & DISPLAY IMAGE ──────────────────────────────
img = cv2.imread("test_image.jpg")
print(f"Image shape: {img.shape}")  
# Output: (height, width, channels)
# channels = 3 means BGR (Blue, Green, Red)

# ── 2. DRAW A RECTANGLE (Bounding Box) ───────────────────
# cv2.rectangle(image, top-left corner, bottom-right corner, color, thickness)
img_boxes = img.copy()
cv2.rectangle(img_boxes, (50, 50), (300, 300), (0, 255, 0), 2)
# (0, 255, 0) = Green in BGR

# ── 3. WRITE TEXT ON IMAGE ───────────────────────────────
cv2.putText(img_boxes, 
            "Defect: Scratch",          # text
            (50, 45),                    # position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.8,                         # font size
            (0, 255, 0),                 # color (green)
            2)                           # thickness

# ── 4. DRAW A CIRCLE ─────────────────────────────────────
cv2.circle(img_boxes, (400, 200), 50, (0, 0, 255), 2)
# (0, 0, 255) = Red in BGR
cv2.putText(img_boxes, "Defect: Hole", (360, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# ── 5. SAVE THE ANNOTATED IMAGE ──────────────────────────
cv2.imwrite("annotated.jpg", img_boxes)
print("Saved annotated.jpg ✅")

# ── 6. CROP A REGION (Region of Interest) ────────────────
# img[y1:y2, x1:x2]
cropped = img[50:300, 50:300]
cv2.imwrite("cropped.jpg", cropped)
print("Saved cropped.jpg ✅")

# ── 7. CONVERT TO GRAYSCALE ──────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayscale.jpg", gray)
print("Saved grayscale.jpg ✅")

# ── 8. BASIC IMAGE INFO ──────────────────────────────────
height, width, channels = img.shape
print(f"\nImage Info:")
print(f"Width: {width}px")
print(f"Height: {height}px")
print(f"Channels: {channels} (BGR)")
print(f"Total pixels: {height * width}")