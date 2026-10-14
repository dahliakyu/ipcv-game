"""Webcam face tracking with a hat and a mustache.

Standalone version, improved step by step (see docs/phase-1/face-script-steps.md)
so that each function maps onto the face/ module of the pipeline.

Setup:  conda activate ipcvgame
        wget -P models https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Run:    python hat_tracker.py              (webcam, press q to quit)
        python hat_tracker.py clip.mp4     (video file instead of the webcam)
"""

import os
import sys
import time

import cv2
import mediapipe as mp  # this is the AI model that finds face landmarks
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ipcvgame.core.filters import OneEuro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")

# Step 4: centralized variables
FOREHEAD, LEFT_SIDE, RIGHT_SIDE, UPPER_LIP = 10, 234, 454, 164   # Face Mesh landmark indices

HAT_WIDTH = 1.5        # hat width in face widths, tuned by eye
HAT_DROP = 0.40        # landmark 10 is at the hairline; brim this many face widths lower
MUSTACHE_WIDTH = 0.80  # mustache width in face widths
DETECT_CONF = 0.5      # MediaPipe default for detection, presence and tracking


# Step 1: one loader and one paste function instead of draw_hat + draw_mustache
def load_sprite(name):
    """Load an RGBA sprite that sits next to this script."""
    img = cv2.imread(os.path.join(SCRIPT_DIR, name), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise FileNotFoundError(f"{name} missing or has no transparency (needs RGBA)")
    return img


def paste(img, sprite, at, pivot, angle, width):
    """Draw `sprite` on `img`: sprite point `pivot` (x, y in sprite pixels)
    lands on image point `at`, rotated by `angle` (rad), scaled to `width` px."""
    sh, sw = sprite.shape[:2]
    M = cv2.getRotationMatrix2D(pivot, -np.degrees(angle), width / sw)
    M[:, 2] += np.asarray(at, dtype=float) - pivot   # move the pivot onto the anchor
    # Box the sprite covers in the image, clipped to the frame
    corners = cv2.transform(np.float32([[[0, 0], [sw, 0], [0, sh], [sw, sh]]]), M)[0]
    H, W = img.shape[:2]
    x1, y1 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(corners.max(0)).astype(int), (W, H))
    if x2 <= x1 or y2 <= y1:
        return                                       # fully off-screen
    M[:, 2] -= (x1, y1)                              # warp into that box only
    warped = cv2.warpAffine(sprite, M, (int(x2 - x1), int(y2 - y1)))
    alpha = warped[:, :, 3:4] / 255.0
    roi = img[y1:y2, x1:x2]
    roi[:] = (warped[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


HAT = load_sprite("hat.png")            # the image you use for your hat
MUSTACHE = load_sprite("mustache.png")  # the image you use for your mustache


# Step 3: the face geometry in one function that returns numbers
def measure(face, w_img, h):
    """Face geometry in pixels from one face's landmarks."""
    def pt(i):
        return np.array([face[i].x * w_img, face[i].y * h])

    left, right = pt(LEFT_SIDE), pt(RIGHT_SIDE)
    d = right - left
    return {
        "width": float(np.linalg.norm(d)),       # face width in px
        "angle": float(np.arctan2(d[1], d[0])),  # head roll, rad, 0 = upright
        "forehead": pt(FOREHEAD),
        "lip": pt(UPPER_LIP),
    }


# Step 6: smoothing, one filter set per face slot
def new_filters():
    return {
        # Positions must follow fast moves: speed term beta > 0
        "points": OneEuro(min_cutoff=1.0, beta=0.05),
        # Width jitter makes the hat pulse; real changes are slow
        "width": OneEuro(min_cutoff=0.5),
        # Angle comes from only two landmarks, so it is noisy
        "angle": OneEuro(min_cutoff=1.0, beta=0.5),
    }


def draw_face(frame, forehead, lip, width, angle):
    # Step 5: move the hat along the face's own "down", so it stays on a tilted head
    down = np.array([-np.sin(angle), np.cos(angle)])   # face's "down" in image coords (y down)
    hh, hw = HAT.shape[:2]
    paste(frame, HAT, forehead + down * HAT_DROP * width,
          (hw / 2, hh), angle, HAT_WIDTH * width)              # pivot: bottom-middle
    mh, mw = MUSTACHE.shape[:2]
    paste(frame, MUSTACHE, lip, (mw / 2, mh / 2), angle, MUSTACHE_WIDTH * width)  # pivot: centre


def main(source=0):
    # Step 2: MediaPipe Tasks API (mp.solutions is gone in mediapipe 1.0)
    if not os.path.exists(MODEL):
        raise FileNotFoundError(f"{MODEL} not found; see the download command at the top")
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,      # video: tracks between frames
        num_faces=2,
        min_face_detection_confidence=DETECT_CONF,
        min_face_presence_confidence=DETECT_CONF,
        min_tracking_confidence=DETECT_CONF))

    # Step 7: clear error when the camera or file does not open
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source!r}")

    filters = [None, None]    # slot 0 = left face, slot 1 = right face
    t0 = prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w_img = frame.shape[:2]
        t = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((t - t0) * 1000)   # VIDEO mode: must go up every call
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        # Step 6: sort faces left to right for now, so each keeps its own filters
        faces = sorted(result.face_landmarks, key=lambda f: f[LEFT_SIDE].x)[:2]
        for slot, face in enumerate(faces):
            m = measure(face, w_img, h)
            if filters[slot] is None:
                filters[slot] = new_filters()
            f = filters[slot]
            forehead, lip = f["points"](np.stack([m["forehead"], m["lip"]]), t)
            width = float(f["width"](m["width"], t))
            angle = float(f["angle"](m["angle"], t))
            draw_face(frame, forehead, lip, width, angle)
        for slot in range(len(faces), 2):
            filters[slot] = None    # face gone: start fresh, so the hat does not slide in

        # Step 7: FPS in the window, to see the speed-up from step 1
        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Facial tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 0)
