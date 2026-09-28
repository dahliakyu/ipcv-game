# Candidate Modules and Libraries per Task

Sep 28, 2026 · @el

## At a glance

One pose model (Ultralytics YOLO-pose) feeds every task, and each task adds a light, explainable layer on top. These are starting points for phase 1; each owner confirms their choice on the shared test clips.

| Task | Recommended | Fallback | Why |
| --- | --- | --- | --- |
| 1 Face + AR | MediaPipe Face Landmarker on per-player head crops + One Euro filter | OpenCV YuNet detector | Landmarks, blendshapes and head pose in one call |
| 2 Pose + motion | Ultralytics YOLO26-pose or YOLO11-pose + One Euro / Kalman filters | RTMPose via `rtmlib` (CPU) | Native multi-person detection in one pass |
| 3 Identity | Own tracker: Kalman + Hungarian + colour histogram | Ultralytics ByteTrack as baseline | Every step explainable; baseline gives a comparison |
| 4 Interaction | NumPy rules + state machine; YOLO detection for props | `scikit-learn` gesture classifier | No training data needed; props use the same library |
| 5 Scene + integration | `pygame-ce` renderer, YOLO-seg for background, threaded capture | OpenCV-only rendering | Full game UI and sound; per-player masks |

## Task 1 — Face tracking and AR effects

The main pick is MediaPipe Face Landmarker, run once per player on a head crop. With `num_faces` above 1 it applies no temporal smoothing, so one instance per tracked player gives smoother output and correct player association by construction.

| Library / model | What it gives | Fit for us | Notes |
| --- | --- | --- | --- |
| **MediaPipe Face Landmarker** | 478 3D landmarks, 52 blendshape scores, face transformation matrix | Main pick | Blendshapes such as `jawOpen` also serve Task 4 ("eat the apple") |
| MediaPipe Face Detector (BlazeFace) | Face boxes + 6 keypoints, very fast | Fallback or re-detection | No orientation beyond roll from eye keypoints |
| OpenCV YuNet (`cv2.FaceDetectorYN`) | Face boxes + 5 landmarks, built into OpenCV | Lightweight alternative detector | No extra dependency |
| InsightFace / RetinaFace | Accurate detection + face embeddings | Only if face identity is needed | Heavier; face embeddings could help Task 3 re-entry |
| dlib 68-point | Classic landmark regressor | Not recommended | Slower, older, awkward to install |

**Alignment and drawing tools:**

- `cv2.warpAffine` for 2D sprites (masks, HP bars): roll from the eye line, scale from inter-ocular distance.
- `cv2.solvePnP` with \~6 landmarks and a canonical face model for 3D head pose; good pinhole-camera theory to explain.
- Alpha blending of RGBA sprites with NumPy for overlays on the camera image.
- One Euro filter (in `core/filters.py`) on anchor points or pose parameters, not on all 478 landmarks.

## Task 2 — Pose estimation and motion tracking

The main pick is a YOLO-pose model from Ultralytics, because it detects every person and their 17 keypoints in one pass. MediaPipe Pose is designed for one person and drops or swaps the second player, so use it only per crop.

| Library / model | What it gives | Fit for us | Notes |
| --- | --- | --- | --- |
| **Ultralytics YOLO26-pose / YOLO11-pose** | Boxes + 17 COCO keypoints for every person, one-stage | Main pick | Runs on CUDA; nano/small sizes for real time; AGPL-3.0 |
| RTMPose / RTMO (via `rtmlib` or MMPose) | Top-down (RTMPose) or one-stage (RTMO) multi-person pose | Best CPU fallback, most accurate real-time option | `rtmlib` avoids the heavy MMPose install |
| MediaPipe Pose Landmarker | 33 landmarks incl. hands/feet + 3D world coordinates | Per-crop refiner only | Model card lists multiple people as out of scope |
| MoveNet MultiPose Lightning | Up to 6 people, 17 keypoints | Browser/CPU fallback | Less accurate on fast limbs |
| OpenPose | Bottom-up multi-person with Part Affinity Fields | Not recommended to run | Heavy install; still good theory to explain |
| MediaPipe Hand Landmarker | 21 hand landmarks per hand | Only if finger gestures are needed | Run on wrist crops from the pose keypoints |

**Smoothing and signals:**

- One Euro filter per keypoint per player: adaptive, so still poses stay steady and fast punches stay responsive.
- Kalman filter (`filterpy`) for position + velocity, and for predicting through short keypoint dropouts.
- Confidence gating: ignore keypoints under a threshold and hold the last valid value for a few frames.
- `scipy.signal.savgol_filter` only for offline analysis of logged data; it needs future frames, so it adds lag live.

## Task 3 — Multiplayer detection and identity tracking

The main pick is a tracker we write ourselves from standard parts, compared against an off-the-shelf tracker as a baseline. Writing it keeps every design choice explainable in the individual questions.

| Library / tool | What it gives | Fit for us | Notes |
| --- | --- | --- | --- |
| **`scipy.optimize.linear_sum_assignment`** | Hungarian algorithm for matching detections to tracks | Core of our own tracker | Cost from 1 − IoU or keypoint similarity (OKS) |
| **`filterpy` Kalman filter** | Constant-velocity prediction of each player's box | Core of our own tracker | Keeps a predicted box while a player is occluded |
| **OpenCV `calcHist` + `compareHist`** | HSV colour histogram of the torso, Bhattacharyya distance | Appearance model for re-entry | Update only when the player is not overlapping the other |
| Ultralytics `model.track()` (ByteTrack, BoT-SORT) | Track IDs out of the box | Baseline to compare against | Configurable track buffer and matching threshold |
| `supervision` (ByteTrack) | Standalone ByteTrack on any detector output | Alternative baseline | Also has drawing and annotation helpers |
| `boxmot` | Many trackers incl. ReID-based ones (e.g. StrongSORT, DeepOCSORT) | Optional ablation | Deep ReID is overkill for 2 players |

**Spatial information for Task 4:**

- Monocular depth from apparent size: shoulder width or box height with the pinhole model, after a one-time calibration at a known distance.
- Ground-plane x position from the hip midpoint, converted to metres with the same calibration.
- Depth Anything V2 or similar monocular depth networks only as an experiment; they cost a lot of GPU time per frame.

**Evaluation:**

- `py-motmetrics` or TrackEval for ID switches and IDF1 on the hand-labelled test clips.
- A small labelling script of our own (click Player 1 / Player 2 per frame) is often enough for two players.

## Task 4 — Player interaction and game control

The main pick is rule-based gesture logic in plain NumPy with a small state machine per player. Rules on joint angles and velocities are fast, need no training data, and are easy to defend; a learned classifier is the fallback when rules become too fiddly.

| Library / approach | What it gives | Fit for us | Notes |
| --- | --- | --- | --- |
| **NumPy rules on keypoints** | Joint angles, distances normalised by shoulder width, wrist velocities | Main pick for gestures | Add hold times, cooldowns and hysteresis per gesture |
| **Hand-written state machine** (or `transitions`) | Idle → charging → fired → cooldown per action | Prevents repeated or accidental actions | A small Python class is usually enough |
| `scikit-learn` (e.g. random forest, k-NN) | Classifier on keypoint features | Fallback for gestures rules can't separate | Needs recorded, labelled examples |
| **Ultralytics YOLO detection** | COCO classes include apple, banana, orange, bottle, cup, sports ball | Real-world props ("eat an apple for HP") | Same library as the pose model |
| MediaPipe Face blendshapes | `jawOpen`, `mouthSmile` and others per player | Mouth-based actions | Comes from Task 1's output |
| `pymunk` | 2D physics: projectiles, collisions, bouncing | Thrown objects, hit detection | Optional; simple circle collisions also work |
| `pygame.mixer` | Sound effects on hits and actions | Clear player feedback | Already part of Pygame |

## Task 5 — Scene processing and real-time integration

The main pick is Pygame (the `pygame-ce` fork) for the game window, with OpenCV windows kept for debug views. For background replacement, YOLO instance segmentation gives a separate mask per player, which fits the identity tracker.

| Library / tool | What it gives | Fit for us | Notes |
| --- | --- | --- | --- |
| **`pygame-ce`** | Game window, sprites, text, sound, input | Main renderer | Draw the camera frame as a surface, effects on top |
| OpenCV `imshow` | Fast debug windows | Debug views and `run_module.py` | Too limited for the final game UI |
| **Ultralytics YOLO-seg** | Instance masks per person | Background replacement per player | Can share the tracker's IDs |
| MediaPipe Selfie / Image Segmenter | Person-vs-background mask | Simple background replacement | One mask for all people, no per-player split |
| ONNX Runtime / TensorRT export (via Ultralytics `export`) | Faster inference on the GPU | Speed-ups if FPS is too low | Measure before and after |
| Python `threading` + `queue` | Camera capture in a background thread | Removes capture wait from the loop | Keep only the latest frame |

**Profiling and project tooling:**

- `time.perf_counter` stage timers in `core/timing.py`; `py-spy` or `cProfile` to find hot spots.
- `pandas` + `matplotlib` for analysing the per-frame CSV logs into report figures.
- `PyYAML` (or OmegaConf) for `configs/default.yaml`.
- `pytest` for unit tests of filters, geometry and the tracker; `ruff` for linting and formatting.

## Compatibility and licences

Before pinning `requirements.txt`, check that one Python version installs every library on every member's machine. MediaPipe wheels in particular lag behind new Python releases.

- [ ] Pick one Python version that `mediapipe`, `ultralytics`, `torch` and `pygame-ce` all ship wheels for.
- [ ] Install the CUDA build of PyTorch on the demo laptop and confirm the GPU is used (`torch.cuda.is_available()`).
- [ ] Confirm a CPU-only path works on members' laptops without a GPU (smaller model or `rtmlib` fallback).
- [ ] Pin exact versions and model weight files; the README gives download commands for the weights.
- [ ] Note licences in the AI and external-tools statement: Ultralytics is AGPL-3.0, MediaPipe is Apache 2.0; both are fine for coursework.

## Sources

From the earlier research survey for this project:

- [MediaPipe Face Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) — landmarks, blendshapes, smoothing only when `num_faces` is 1
- [BlazePose GHUM 3D model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) — multiple people out of scope
- [MediaPipe issue #5842](https://github.com/google-ai-edge/mediapipe/issues/5842) — Pose Landmarker with `num_poses` above 1
- [Ultralytics YOLO26 paper](https://arxiv.org/pdf/2606.03748)
- [Ultralytics tracking docs](https://docs.ultralytics.com/modes/track) — ByteTrack and BoT-SORT
- [RTMPose paper](https://arxiv.org/html/2303.07399v2)
- [MoveNet in TensorFlow.js](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/src/movenet)
- [OpenPose paper](https://arxiv.org/html/1812.08008v2)

Library descriptions without a link above (OpenCV, SciPy, filterpy, supervision, boxmot, pymunk, pygame-ce and tooling) are from general knowledge; check each project's own documentation before relying on a specific feature.
