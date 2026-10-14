Oct 14, 2026 · @el created with help of Claude Opus 5.5

# Improving `hat_tracker.py` before the port

These are seven small steps that improve `hat_tracker.py` while it stays a single, runnable script. After them, each function in the script matches a piece of the `face/` module almost one to one, so the port into the pipeline is mostly moving code, not rewriting it.

Steps 1 to 5 are refactors and bug fixes; step 6 is the only new behaviour. Run the script after every step to check that the hat and mustache still look the same.

| Step | What | Why | Becomes in `face/` |
| --- | --- | --- | --- |
| 1 | `load_sprite` + `paste` | One function instead of two copies; faster; works from any folder | `effects.py`: `load_sprite`, `paste` |
| 2 | MediaPipe Tasks API | Pinned `mediapipe==1.0.1` has no `mp.solutions` | `tracker.py`: per-player landmarker |
| 3 | `measure()` | Geometry in one testable function | `tracker.py`: `measure` |
| 4 | Named constants | Every number explained in one place | `configs/default.yaml`, `face:` section |
| 5 | Hat along the face's "down" | Hat stays on a tilted head | `effects.py`: `FaceEffects.draw` |
| 6 | Left/right order + One Euro | Steady hat, no jitter | `tracker.py`: filters per player |
| 7 | FPS, errors, cleanup | Measure the speed-up; clear failures | `ms_face` in the run log |

Work in the shared env so the script runs on the pinned versions: `conda activate ipcvgame`.

## Step 1: `load_sprite` and `paste`

`draw_hat` and `draw_mustache` are the same code with three numbers changed: the sprite, the pivot point on the sprite, and the width. One `paste` function replaces both and fixes three things on the way:

- **Paths.** `cv2.imread("hat.png")` reads from the folder you run the script from. Run it from anywhere else and it silently returns `None`, which crashes later with an unclear `NoneType` error. `load_sprite` reads next to the script and fails with a clear message.
- **Speed.** Today each sprite is warped and blended over the whole frame: 2 sprites × 2 faces = 4 full-frame blends per frame. `paste` warps only the box the sprite covers.
- **Off-screen sprites.** A sprite that is partly or fully outside the frame is clipped or skipped.

```python
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


HAT = load_sprite("hat.png")
MUSTACHE = load_sprite("mustache.png")
```

The two old functions become two calls. The pivot and the maths are the same as before, so the result looks identical:

```python
hh, hw = HAT.shape[:2]
paste(frame, HAT, (top[0], top[1] + 0.40 * face_width), (hw / 2, hh), angle, 1.5 * face_width)   # pivot: bottom-middle
mh, mw = MUSTACHE.shape[:2]
paste(frame, MUSTACHE, lip, (mw / 2, mh / 2), angle, 0.80 * face_width)                       # pivot: centre
```

## Step 2: MediaPipe Tasks API

The pinned `mediapipe==1.0.1` no longer has `mp.solutions`. The new API needs a model file, which you download once (`models/` is gitignored):

```bash
wget -P models https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

The landmark indices (10, 234, 454, 164) are the same in the new model. Only `main()` changes:

| Old (`mp.solutions`) | New (Tasks API) |
| --- | --- |
| `mp.solutions.face_mesh.FaceMesh(...)` | `vision.FaceLandmarker.create_from_options(...)` |
| `face_mesh.process(rgb)` | `landmarker.detect_for_video(mp.Image(...), timestamp_ms)` |
| `result.multi_face_landmarks or []` | `result.face_landmarks` (empty list, never `None`) |
| `face.landmark[i].x` | `face[i].x` |

```python
import time

from mediapipe.tasks.python import BaseOptions, vision

MODEL = os.path.join(SCRIPT_DIR, "models", "face_landmarker.task")


def main():
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,      # video: tracks between frames
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5))
    cap = cv2.VideoCapture(0)
    t0 = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w_img = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int((time.perf_counter() - t0) * 1000)   # VIDEO mode: must go up every call
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        for face in result.face_landmarks:
            ...   # same as before, with face[i] instead of face.landmark[i]
```

`refine_landmarks` is gone: the new model always returns 478 landmarks.

## Step 3: `measure()`

Move the geometry out of the loop into one function that returns numbers instead of drawing. It has no side effects, so it can be checked on its own (for example: a face rotated by 10° must give `angle` = 10°).

```python
def measure(face, w_img, h):
    """Face geometry in pixels from one face's landmarks."""
    pt = lambda i: np.array([face[i].x * w_img, face[i].y * h])
    left, right = pt(LEFT_SIDE), pt(RIGHT_SIDE)
    d = right - left
    return {
        "width": float(np.linalg.norm(d)),       # face width in px
        "angle": float(np.arctan2(d[1], d[0])),  # head roll, rad, 0 = upright
        "forehead": pt(FOREHEAD),
        "lip": pt(UPPER_LIP),
    }

# in the loop:
for face in result.face_landmarks:
    m = measure(face, w_img, h)
    ...
```

## Step 4: Named constants

Give every number a name and a one-line reason at the top of the file. Each of these becomes a key in the `face:` section of `configs/default.yaml` when the script is ported, and each is something you may be asked to justify.

```python
FOREHEAD, LEFT_SIDE, RIGHT_SIDE, UPPER_LIP = 10, 234, 454, 164   # Face Mesh landmark indices

HAT_WIDTH = 1.5        # hat width in face widths, tuned by eye
HAT_DROP = 0.40        # landmark 10 is at the hairline; brim this many face widths lower
MUSTACHE_WIDTH = 0.80  # mustache width in face widths
DETECT_CONF = 0.5      # MediaPipe default for detection, presence and tracking
```

## Step 5: Move the hat along the face's own "down"

`top[1] + 0.40 * face_width` moves the hat straight down the screen. When the head tilts, the face's own "down" is no longer screen-down, so the hat slides sideways off the forehead. Rotating the offset by the same angle keeps it on the head:

```python
down = np.array([-np.sin(m["angle"]), np.cos(m["angle"])])   # face's "down" in image coords (y down)
hat_at = m["forehead"] + down * HAT_DROP * m["width"]
paste(frame, HAT, hat_at, (hw / 2, hh), m["angle"], HAT_WIDTH * m["width"])
```

When the head is upright (`angle` = 0), `down` is `(0, 1)` and the result is exactly what the script did before.

## Step 6: Left/right order and smoothing

With `num_faces=2`, MediaPipe returns faces in no fixed order. A filter per face would mix the two people when the order flips, so first give each face a stable slot: sort by x, left face first. Then keep one set of One Euro filters per slot. Use the shared filter from the repo instead of writing a new one; it uses real time, so it works at any FPS.

```python
from ipcvgame.core.filters import OneEuro


def new_filters():
    return {
        # Positions must follow fast moves: speed term beta > 0
        "points": OneEuro(min_cutoff=1.0, beta=0.05),
        # Width jitter makes the hat pulse; real changes are slow
        "width": OneEuro(min_cutoff=0.5),
        # Angle comes from only two landmarks, so it is noisy
        "angle": OneEuro(min_cutoff=1.0, beta=0.5),
    }


filters = [None, None]        # slot 0 = left face, slot 1 = right face

# in the loop:
t = time.perf_counter()
faces = sorted(result.face_landmarks, key=lambda f: f[LEFT_SIDE].x)[:2]
for slot, face in enumerate(faces):
    m = measure(face, w_img, h)
    if filters[slot] is None:
        filters[slot] = new_filters()
    f = filters[slot]
    forehead, lip = f["points"](np.stack([m["forehead"], m["lip"]]), t)
    width = float(f["width"](m["width"], t))
    angle = float(f["angle"](m["angle"], t))
    ...   # draw with the filtered values
for slot in range(len(faces), 2):
    filters[slot] = None      # face gone: start fresh, so the hat does not slide in
```

Filter the few measured values, not the 478 landmarks: that is cheaper, and every drawn point is derived from them.

This shows the limit of a standalone script: if only one person is in view, they always get slot 0, whichever side they stand on. In the pipeline this is solved properly, because Task 3 gives each player a fixed ID and the face module runs one landmarker per player.

## Step 7: FPS, errors and cleanup

- Show the FPS in the window, so the speed-up from step 1 is visible.
- Fail with a clear message when the camera does not open.
- Close the landmarker at the end.

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam 0")
prev = time.perf_counter()

# in the loop, after drawing:
now = time.perf_counter()
fps = 1.0 / max(now - prev, 1e-6)
prev = now
cv2.putText(frame, f"{fps:.0f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# after the loop:
landmarker.close()
cap.release()
cv2.destroyAllWindows()
```

## After these steps

Moving the script into the pipeline is then mostly moving code:

| In the script | Goes to |
| --- | --- |
| `load_sprite`, `paste`, the two `paste` calls | `face/effects.py`, `FaceEffects.draw` |
| `measure` | `face/tracker.py`, module level |
| Constants | `configs/default.yaml`, `face:` section |
| `new_filters`, the filter calls | `face/tracker.py`, one set per player ID instead of per slot |
| Landmarker setup | `face/tracker.py`, one landmarker per player, run on a head crop |
| Camera, flip, window, FPS | Removed: the pipeline does these |
