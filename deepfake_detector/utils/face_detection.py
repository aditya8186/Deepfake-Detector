import cv2
import numpy as np
from typing import List, Tuple


_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_primary_face(image_rgb: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    faces = detect_faces(image_rgb)
    h, w, _ = image_rgb.shape
    if not faces:
        # fallback center crop square
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = image_rgb[y0:y0 + side, x0:x0 + side]
    else:
        # choose largest face
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        pad = int(0.25 * max(fw, fh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + fw + pad)
        y1 = min(h, y + fh + pad)
        crop = image_rgb[y0:y1, x0:x1]

    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

