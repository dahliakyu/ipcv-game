Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Merge log: Colour Clash prototype into the shared codebase

This log records every change made while merging the Colour Clash prototype (the `Colour Clash/` folder) into the shared codebase. The merge is done by the Task 5 owner on branch `face-tracking-eline` and goes to `main` in a single PR when it is complete.

Each entry names the files it touched, so owners can check what changed in their own code.

## Pre-merging Inits

### Clean-up of the `Colour Clash/` folder

The folder arrived as a Windows zip of 1.4 GB and ~33,800 files; after the clean-up it holds 24 MB and 15 files. No source code was changed in this step.


**Deleted:**

| Path | What it was | Why it can go |
| --- | --- | --- |
| `Colour Clash/Project/Lib/`, `Scripts/`, `Include/`, `share/`, `pyvenv.cfg` | A Windows Python 3.11 virtual environment (torch, scipy, ultralytics, …) | Packages come from `requirements.txt` and the `ipcvgame` conda env; ~99% of the folder's size |
| `Colour Clash/Project/personreid/` | A git clone of [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (torchreid) | Unmodified upstream code; the 154 "modified" files only had Windows line endings |
| `Colour Clash/__pycache__/`, `Colour Clash/Project/__pycache__/` | Python bytecode caches | Regenerated automatically |
| `Colour Clash/.vscode/`, `Colour Clash/Project/.vscode/` | Personal editor settings | Not shared project config |
| `Colour Clash/Project/Project description.pdf` | The project brief | Byte-identical to `docs/phase-1/IPCV_2026-1A_Project_Description_v1-1.pdf` |
| `Colour Clash/yolo26n-pose.pt` | Pose model weights | Byte-identical to `Colour Clash/Project/yolo26n-pose.pt`, which is the copy `main.py` loads |

**Kept:** all `.py` files, `botsort.yaml`, `notes.txt`, the palette `.jpg` (loaded by `renderer.py`), and the three model weight files: `Project/yolo26n-pose.pt`, `Project/models/yolo11n.pt` and `Project/models/osnet_x1_0_msmt17.pth`. The weights are ignored by git (`*.pt`, `models/`) and must stay out of the repo.

### New dependency: `lap`

**Changed:** `requirements.txt`

- Added `lap==0.5.13`. Ultralytics' BoT-SORT tracker (`person_detector.track(...)` in `Colour Clash/main.py`) imports it, and if it is missing Ultralytics pip-installs it silently at runtime.
- Every other import in the prototype (`ultralytics`, `torch`, `torchvision`, `cv2`, `numpy`) was already pinned.

**Action for everyone:** run `pip install -r requirements.txt` in the `ipcvgame` env.

### OSNet copied into the code; `torchreid` dropped

The identity code (`PlayerIdentityTracker` in `Colour Clash/Project/bodypose.py`) uses OSNet to recognise a player who comes back after the tracker lost them. It crops the person, turns the crop into a 512-number appearance vector, and compares that vector with each known player's (cosine similarity, threshold 0.6). OSNet came from the `torchreid` package, but the code used only two of its functions: building the network and loading the weights. Importing `torchreid` also needs `gdown` and `tensorboard`. The one file that defines the network is now copied into the project, so none of the three packages is needed.

**Added:** `Colour Clash/Project/osnet.py`

- Copied from `torchreid/models/osnet.py` at upstream commit `f8cd150f` (MIT licence; the licence text and source are in the file header).
- Trimmed to `osnet_x1_0` and the layers it uses. The other model sizes and the Google Drive weight download were removed.
- New `load_weights(model, path)` replaces `torchreid.utils.load_pretrained_weights`. It loads with `torch.load(..., weights_only=True)` and raises an error if no layer matches. The upstream version only printed a warning and kept running with random weights.
- Layer names are unchanged, so `osnet_x1_0_msmt17.pth` loads as before. Only the classifier layer is skipped, as before; it isn't used at inference.

**Changed:** `Colour Clash/Project/osnet_embedding.py`

- `import torchreid` replaced by `from osnet import load_weights, osnet_x1_0`, with the same `if __package__:` pattern as `bodypose.py`. Both `python main.py` (from `Colour Clash/`) and running scripts inside `Project/` still work.
- `torchreid.models.build_model(name="osnet_x1_0", num_classes=4101, pretrained=False)` became `osnet_x1_0(num_classes=4101)`.
- `torchreid.utils.load_pretrained_weights(self.model, model_path)` became `load_weights(self.model, model_path)`.
- Nothing else in the file changed; `OSNetEmbeddingExtractor` behaves as before.

**Changed:** `requirements.txt`: no `torchreid`, `gdown` or `tensorboard` entries.

**Checked:**

- On the same input with the same weights, the copied network gives the same output as upstream torchreid (max difference 0.0).
- `OSNetEmbeddingExtractor` returns a unit-length 512-d vector on CUDA.
- `Colour Clash/main.py` imports without `torchreid`.
- `ruff` finds no errors in the two files.

**Action for anyone who installed it:** `torchreid`, `gdown` and `tensorboard` are no longer needed and can be removed with `pip uninstall torchreid gdown tensorboard tensorboard-data-server`.

### Open issues found during the merge

These are not fixed yet; each belongs to the owner of the code.

- **Two detectors per frame (Task 3, Task 2).** `Colour Clash/main.py` runs `yolo11n.pt` as a separate person detector for tracking and then `yolo26n-pose.pt` for keypoints. The architecture rule is one pose model per frame. YOLO-pose already returns person boxes, so tracking can run on those and `yolo11n.pt` can go.
- **Stored appearance is never updated (Task 3).** In `PlayerIdentityTracker`, each player's OSNet vector is stored once, when the player is created. Lighting changes or a jacket coming off can then break re-identification. Matching is also greedy per track, not a joint assignment (Hungarian).
- **`Colour Clash/Project/test_osnet.py` fails as written.** It calls `OSNetEmbeddingExtractor()` without the required `model_path` argument, and it opens the webcam.
- **Working directory.** `Colour Clash/` has a space in its name and `Project/` has no `__init__.py`, so `main.py` only runs with `Colour Clash/` as the working directory. This goes away when the code moves into `src/ipcvgame/`.

## Change Overviews

A quick reference for each teammate: where your prototype code now lives, and what was changed beyond moving it. The task sections below have the step-by-step details and the reasons.

### Refactor: where to find your code

#### Task 1: `face_track/facial_tracking_yunet.py`

**Functions and classes**

| In the prototype | Now in | Note |
| --- | --- | --- |
| `measure()` | `face/tracker.py`, `measure()` | Same maths; constants come from config |
| `measure()`, the `lk` grid part | `face/tracker.py`, `flow_points()` | Built from the 4 key points |
| `geometry()` | `face/tracker.py`, `geometry()` | Also returns `forehead` and `above_head`; since step 9 the fallback when the head pose is off or fails |
| (new, steps 9–10) head pose | `face/head_pose.py`: `camera_matrix()`, `head_pose()`, `pose_angles()`, `rotation()`, `mediapipe_angles()`, `translation()`, `anchor_points_mm()`; used in `face/tracker.py`, `FaceTracker._smoothed_geometry()` | Not in the prototype |
| (new, step 10) mouth, blinks, MediaPipe pose | `face/landmarker.py`, `FaceLandmarker`; `face/tracker.py`, `FaceTracker._landmark()` | Not in the prototype |
| `OneEuro` | `core/filters.py`, `OneEuro` | Shared filter, same maths |
| `Track` (state) | `face/tracker.py`, `_FaceState` | One per player ID |
| `Track.update()` | `face/tracker.py`, `FaceTracker.update()`, "detect" branch | |
| `Track.flow()` | `face/tracker.py`, `estimate_motion()` + the "flow" branch of `update()` | |
| `Track._apply()` | `face/tracker.py`, `FaceTracker.update()` | Filter, then `geometry()` |
| `Track.alpha()` | `face/tracker.py`, `fade()` | In seconds; result in `Face.alpha` |
| `Track.draw()`, `overlay()` | `face/effects.py`, `FaceEffects.draw()` + `paste()` | |
| `load_detector()` | `face/tracker.py`, `FaceTracker.__init__()` | |
| `update_tracks()` | `face/tracker.py`, `head_box()` + `FaceTracker._detect()` | Replaced; see Modifications |
| Main loop: detect, flow, delete tracks | `face/tracker.py`, `FaceTracker.update()` | |
| `Metrics` | `face/tracker.py`, `FaceTracker.log_fields()` + `tools/eval_face.py`, `evaluate()` | Jitter = `second_difference()` |
| `Metrics.report()` | `tools/eval_face.py`, `main()` + `markdown()` | Usage: `docs/evaluation.md` |
| `open_camera()`, camera loop, flip, window, FPS text | `io/source.py`, `app/main.py`, `scene/renderer.py` | Shared pipeline (Task 5) |
| Keys `q` | `app/main.py` (`q`/`Esc`; also `p` pause, `r` reset) | |
| `draw_status()` (name + HP bar) | Not ported | Game UI; planned in `docs/extra_features.md` |
| `Track.predicted()`, `Track.vel` | Removed | Only used for matching |
| Keys `s`, `d`, `l`, `x`, `1`/`2` | Removed | Conditions come from clip names; HP demo is game logic |

**Constants**

All of these are in `configs/default.yaml`, `face:` section, unless another section is named.

| In the prototype | Now |
| --- | --- |
| `MODEL`, `MODEL_URL` | `model` (download command in `README.md`) |
| `SCORE_THRESH` | `score_threshold` |
| `MIN_FACE` | `min_face` |
| `MAX_SIZE_JUMP` | `max_size_jump` |
| `HOLD_FRAMES`, `FLOW_FRAMES`, `MAX_MISSED` | `hold_s`, `flow_max_s`, `lost_after_s` (seconds) |
| `LK_GRID` | `flow_grid_x`, `flow_grid_y`, `flow_grid_n` |
| `LK_PARAMS` | `flow_win`, `flow_levels`, `flow_criteria` |
| Numbers inside `Track.flow()` (8 points, 6 inliers, 3.0 px, 0.7–1.4) | `flow_min_points`, `flow_min_inliers`, `flow_ransac_px`, `flow_scale_range` |
| Numbers inside `measure()` / `geometry()` (0.20, 0.50, 0.35, 0.40) | `center_drop`, `forehead_rise`, `lip_to_nose`; the 0.40 became `hat_rise` (0.11) in step 10, see Modifications |
| Numbers inside `Track.draw()` (1.5, 0.80) | `hat_width`, `mustache_width` |
| `OneEuro(1.5, 0.1)` | `filter_min_cutoff`, `filter_beta`, `filter_d_cutoff` |
| `HAT`, `MUSTACHE` (`hat.png`, `mustache.png`) | `hat_sprite`, `mustache_sprite` → `assets/face/` |
| `CAP_W`, `CAP_H`, `CAMERA_INDEX` | `source:` section (`width`, `height`, `uri`) |
| `MAX_PLAYERS` | `identity:` section (`num_players`), Task 3 |
| `COLORS` | `scene/draw.py`, `PLAYER_COLORS`, Task 5 |
| `MATCH_FACTOR`, `LIGHTS`, `EXPOSURE` | Removed (matching replaced; conditions from clip names; exposure not ported) |

#### Task 2 and Task 4: `Project/bodypose.py` (smoothing and gestures)

**Functions and classes**

| In the prototype | Now in | Note |
| --- | --- | --- |
| `KeypointTracker` | `pose/motion.py`, `MotionFilter._smooth()` + `_PlayerState` | Task 2; same rules, in seconds |
| `KeypointTracker.reset()` | `pose/motion.py`, `MotionFilter.reset(pid)` | Also on re-entry |
| `draw_pose()` | `pose/motion.py`, `MotionFilter.draw_debug()` | Shared `draw_skeleton()` |
| `angle_at()` | `core/geometry.py`, `angle_at()` | Shared helper |
| `lower_body_y()`, `detect_hands_up()`, `detect_crossed()`, `arm_extended()`, `arm_bent()`, `detect_punch()` | `interaction/gestures.py`, same names | Task 4; numbers from config, keypoints by name |

**Constants** (`configs/default.yaml`)

| In the prototype | Now |
| --- | --- |
| `CONF_T` (keypoints) | `motion.kpt_conf_threshold` |
| `ALPHA` | `motion.tau` (seconds) |
| `MISSING_FRAMES` | `motion.hold_s` (seconds) |
| `CROSS_MAX_WRIST_DIST`, `PUNCH_ELBOW_ANGLE`, `PUNCH_MIN_REACH`, `PUNCH_BENT_ANGLE` | `gestures.cross_max_wrist_dist`, `punch_elbow_angle`, `punch_min_reach`, `punch_bent_angle` |
| Numbers inside the functions (0.3, 2.0, 0.5, 1.0 sw) | `gestures.head_above_shoulders`, `hip_below_shoulders`, `chest_above_shoulders`, `punch_above_shoulders` |
| `NOSE`, `L_SHOULDER`, …, `SKELETON` | `core/skeleton.py` (names and `EDGES`) |
| `CONF_T` (detection), `IMGSZ`, `MODEL_PATH` | `pose.det_conf`, `pose.imgsz`, `pose.weights` (already there; the pipeline runs the pose model once on the full frame) |

#### Task 3: `Project/` (identity)

Not merged into `src/ipcvgame/` yet; only the pre-merge OSNet change below exists. To be added when the identity merge happens.

### Modifications: changes beyond moving code

#### Task 1

- **Head crops instead of full-frame detection.** YuNet runs on a crop around each player's head, taken from the pose keypoints, and the player ID comes from Task 3. `update_tracks()`, greedy matching and `MAX_PLAYERS` are gone, so faces cannot swap between players. It is also faster: 1.2 ms per small crop against 24.5 ms for a full 720p frame.
- **Which detection is used (new).** If a crop holds more than one face, the one closest to the player's head keypoints is used.
- **Seconds instead of frames.** Hold, flow and lost limits are in seconds (`Frame.t`), so they behave the same at any frame rate.
- **Fade-out moved into the contract.** The opacity is now `Face.alpha`, a new `core/types.py` field, so the renderer and the game can see it.
- **More anchors.** `forehead` and `above_head` are added next to `hat` and `mouth`.
- **Head pose and 3D anchors (step 9, new).** `solvePnP` fits a canonical 3D head to the 5 landmarks. The forehead, hat and `above_head` anchors are then 3D points on that head, projected into the image. The prototype's fixed 2D offsets made the hat rise when the chin went up, and sink when it went down. `Face.yaw` is filled from the pose.
- **Head rotation from MediaPipe (step 10, new).** YuNet's 5 landmarks stay near-frontal at strong up/down tilts (−3° measured on a 33° chin-up), so the rotation now comes from MediaPipe Face Landmarker, which runs on one player per frame, in turn. `solvePnP` on YuNet's landmarks plus the pose model's ears is the fallback.
- **Mouth and blinks (step 10, new).** MediaPipe's blendshape scores go into `Face.blendshapes`, a new `core/types.py` field.
- **Hat hangs from its brim (step 10).** The prototype's `overlay()` pinned the hat at the bottom of the PNG, which has 34% transparent padding below the hat, so the visible brim floated half a face width above the anchor. The hat now hangs from its visible brim, and `hat_drop` (0.40 below the forehead point) became `hat_rise` (0.11 above it), which keeps the frontal look the same.
- **Smoothing order (step 9).** The One Euro filter now runs on the face sides and the anchors together, after the 3D projection, instead of on the 4 key points before `geometry()`.
- **Flow also moves the 5 landmarks (new),** not just the key points and flow points.
- **Large head crops are shrunk (step 8, new).** Crops over `crop_max_px` (192) are shrunk before YuNet, and the detection is scaled back to frame pixels: 23.4 → 9.2 ms on a 4K clip with nearly the same detections.
- **Faster drawing.** `paste()` warps only the sprite's box, not the whole frame. It looks the same: 2.6 ms against 85 ms for two faces at 720p.
- **Face state dropped** after `lost_after_s` without any update, and when the player disappears or re-enters (`reset(pid)`).
- **No model download at runtime.** The YuNet model is fetched once with the `README.md` command.
- **The old stand-in stays as `face.backend: stub`,** so the tests run without the model.
- **Evaluation** is per clip, with conditions from clip names instead of key presses. Latency is the face-stage time (`ms_face`). The identity-switch count moved to Task 3.
- **Name and HP bar** are not drawn by the face stage any more; they are game UI.
- **One Euro filter:** the shared one in `core/filters.py`. The only difference is the minimum time step (1 µs instead of 1 ms), which is negligible.

#### Task 2 and Task 4

- **Smoothing moved out of the gesture code** into the motion stage (`pose/motion.py`), so the face stage and the game read the same smoothed keypoints.
- **Seconds instead of frames.** `ALPHA` became a time constant and `MISSING_FRAMES` a hold time, so smoothing and hold behave the same at any frame rate.
- **Velocities (new):** `Motion.vel` is the finite difference of the smoothed points; 0 on the frame a point jumps back.
- **`pose_detector.py` not merged:** it is an older single-player copy of `bodypose.py` with slightly different thresholds.

#### Task 3 (pre-merge)

- **`torchreid` replaced by a copied `osnet.py`.** `osnet_embedding.py` imports it instead of `torchreid`. Embeddings are identical (max difference 0.0).
- **`load_weights()` raises an error** when no layer of the weight file matches, instead of only warning and running with random weights.

## Task 1: Face tracking and AR

### Plan

**Source.** `face_track/facial_tracking_yunet.py` with `hat.png` and `mustache.png`, the Task 1 owner's latest standalone script. The two sprites are byte-identical to the ones already in the repo root.

**What the script does.**

- **Detection:** OpenCV's YuNet face detector runs on the full frame every frame; it returns a box and 5 landmarks per face.
- **Geometry:** `measure()` turns the 5 landmarks into 4 key points (both face sides, forehead, upper lip). From these, `geometry()` gets face width, head roll, a hat anchor and a lip anchor.
- **Fallback:** when YuNet misses a face, optical flow (`calcOpticalFlowPyrLK` plus a RANSAC similarity fit) carries it along.
- **Smoothing and identity:** a One Euro filter smooths the 4 key points; greedy nearest-centre matching gives each face a player ID.
- **Drawing:** the hat and mustache are warped onto the frame and fade out after a face is lost. A name and HP bar are drawn as well.
- **Report metrics:** jitter, alignment error, detection rate per lighting condition, latency.

**Decisions (Task 5 owner, 2026-10-25).**

- **Detector: YuNet**, the Task 1 owner's latest choice. It needs only OpenCV and is fast on small crops. YuNet gives only 5 landmarks, so `Face.yaw` stays 0.0. There are no blendshapes either, so mouth-based game actions would need MediaPipe later.
- **Fade-out kept.** It needs a new `alpha` field on `Face` in `core/types.py`; this log is the team heads-up for that change.
- **The Task 1 owner is informed through this log** rather than reviewing before each step.

**The main change: per-player head crops instead of the script's own identity.** In the pipeline, Task 3 decides who is who. So the face stage crops each player's head from that player's pose keypoints and runs YuNet only inside that crop. As a result:

- Faces cannot swap between players, which is the phase-1 deliverable.
- It is much cheaper. Measured on the demo machine's CPU: a full 1280×720 frame costs 24.5 ms, a 200×200 head crop 1.2 ms per player.
- `update_tracks`, `Track.predicted()`, the velocity and `MATCH_FACTOR` are no longer needed, because they only existed for matching.

The owner's methods stay: YuNet, `measure()`, One Euro and the optical-flow fallback.

**Where each part goes.**

| In the script | Goes to | Change |
| --- | --- | --- |
| `load_detector()` | `face/tracker.py`, `FaceTracker.__init__` | Model path in config, file in `models/` (gitignored); no download at runtime, the README has the command |
| `measure()`, `geometry()` | `face/tracker.py`, module level | Same maths; detections shifted by the crop offset |
| Full-frame detection, de-duplication, `update_tracks`, `MAX_PLAYERS` | Replaced | One head crop per `Player`; the detection closest to the head keypoints is used |
| `Track` | Per-player state in `FaceTracker` | Cleared by `reset(pid)`, which the pipeline calls when a player re-enters |
| Own `OneEuro` class | `core.filters.OneEuro` | Same maths, so `min_cutoff=1.5, beta=0.1` carry over |
| `flow()` | `face/tracker.py` | Same method; limits in seconds instead of frames (filters use `Frame.t`) |
| `HOLD_FRAMES`, `FLOW_FRAMES`, `MAX_MISSED` | `face:` config, in seconds | At 30 FPS: 0.1 s, 1.3 s, 1.0 s |
| Other constants | `face:` config, each with a reason | `MIN_FACE` was tuned at 640×480; re-check at 1280×720 |
| `overlay()` | `face/effects.py`, `FaceEffects.draw` | `paste()` from `docs/phase-1/facetracking-update-guide.md`, which warps only the sprite's box instead of the full frame |
| `draw_status()` (name + HP bar) | Out of Task 1 | Game UI (Tasks 4 and 5); should draw at `faces[pid].anchors["above_head"]` |
| `Metrics` | `log_fields()` + `tools/eval_face.py` | Per-player CSV columns; jitter and alignment computed from the CSV; lighting and motion labels from the shared clip names |
| Camera, flip, window, keys, FPS text | Removed | The pipeline and `io/` already do these |
| `hat.png`, `mustache.png` | `assets/face/` | Paths in config |

**Steps.** Each step gets an entry under Changes below.

1. Contract, config and model: `Face.alpha`, `face:` config keys, YuNet model in `models/`, README download line.
2. Effects: `load_sprite` and `paste` in `effects.py`, with tests.
3. Geometry: `measure()` and `geometry()` in `tracker.py`, with tests.
4. Detection on head crops: YuNet, One Euro, `Face` output, `draw_debug`.
5. Optical-flow fallback, hold time and fade-out.
6. Evaluation: `log_fields` and `tools/eval_face.py`.
7. Cleanup: remove the old root scripts and sprites, mark the update guide as history, update `docs/phase-1/status.md`.

**Added after step 6 (decisions by the Task 5 owner, 2026-10-25).** These steps only touch `face/`, apart from one `Face` field:

8. Speed: scale each head crop down to a maximum size before YuNet; the 4K test clip costs 24 ms in the face stage.
9. Head pose: `cv2.solvePnP` on the 5 YuNet landmarks and a canonical 3D face, which fills `Face.yaw`.
10. Mouth opening and blinking: MediaPipe Face Landmarker on the face box YuNet finds, alternating players frame by frame. Its blendshapes (`jawOpen`, `eyeBlinkLeft`/`Right`, …) go into a new `Face.blendshapes` field (**team heads-up: contract change**, approved). Uses:
    - eating consumables with the mouth;
    - a blink resets the super-charge move;
    - taunt (smile) and cheek bulge effects.

Game features that build on the face stage (team colour, hit reaction, status above the head, taunt, cheek bulge, Smash-Bros-style consumables) are written up for later in `docs/extra_features.md`.

**Verification.** Steps 2 and 3 are covered by unit tests. Steps 4 and 5 are checked by the Task 5 owner on the webcam (single player). The multiplayer cases (two faces, crossing) wait for recorded clips.

### Changes

#### Step 1 · Contract, config and model

**Changed:** `src/ipcvgame/core/types.py`

- `Face` has a new last field, `alpha: float = 1.0`: the effect's opacity, 1 while the face is tracked, fading to 0 once it is lost. It keeps the prototype's fade-out.
- It has a default, so existing code that builds a `Face` without it keeps working.

**Changed:** `configs/default.yaml`, `face:` section

- `backend`: `yunet` (real detector) or `stub` (the old nose-keypoint anchor, no model file needed; used by the tests).
- `model`, `score_threshold` (0.6, prototype `SCORE_THRESH`), `nms_threshold` (0.3) and `top_k` (50) for YuNet.
- The stub keys `above_head_factor` and `default_scale` stay, marked "stub backend only".

**Changed:** `README.md`: download command and SHA-256 for `models/face_detection_yunet_2023mar.onnx`. The script downloaded the model at runtime; the pipeline doesn't download it, and the model stays out of git (`models/` is ignored).

**Checked:** 18 tests pass, `ruff` clean.

#### Step 2 · Effects: `load_sprite` and `paste`

**Changed:** `src/ipcvgame/face/effects.py` (replaces the stub's dot)

- `load_sprite(path)`: loads an RGBA PNG (a relative path is taken from the repo root) and fails with a clear error if it is missing or has no transparency.
- `paste(img, sprite, at, pivot, angle, width, alpha)`: the version from `docs/phase-1/facetracking-update-guide.md` (step 1), plus the prototype's `alpha` fade. It replaces the script's `overlay()`. The geometry is the same: the pivot lands on the anchor, the sprite is rotated by the head roll and scaled to a multiple of the face width. The difference is that only the sprite's box is warped and blended, and a sprite partly off-screen is clipped instead of crashing.
- `FaceEffects.draw`: hat at `anchors["hat"]` (pivot bottom-middle), mustache at `anchors["mouth"]` (pivot centre), both faded by `Face.alpha`. Faces with `valid=False` or `alpha=0` are skipped. The stub tracker has neither anchor, so until step 4 nothing is drawn on faces in the game view.
- The script's `draw_status()` (name + HP bar) is not ported: it is game UI.

**Added:** `assets/face/hat.png`, `assets/face/mustache.png` (copied from `face_track/`). The copies in the repo root stay until step 7, so `hat_tracker_v2.py` keeps working.

**Changed:** `configs/default.yaml`, `face:`: `hat_sprite`, `mustache_sprite`, `hat_width` (1.5) and `mustache_width` (0.8), from the script's `overlay()` calls.

**Added:** `tests/test_effects.py`: 7 tests covering pivot placement, scaling, rotation, alpha, off-screen clipping, sprite errors, and skipping hidden faces.

**Checked:** 25 tests pass, `ruff` clean.

- Drawing two faces at 1280×720 takes 2.6 ms; the script's full-frame `overlay()` takes 85 ms for the same two faces.
- A rendered check image looks right: the hat and mustache turn with a tilted head, and `alpha=0.5` draws them half-transparent.

#### Step 3 · Geometry: `measure()` and `geometry()`

**Changed:** `src/ipcvgame/face/tracker.py`: two module-level functions above the (still stub) `FaceTracker`; the class itself is unchanged until step 4.

- `measure(row, center_drop, forehead_rise, lip_to_nose)`: the prototype's `measure()`. It turns one YuNet detection into the 4 key points (left side, right side, forehead, upper lip), the face centre at cheek level, and the 5 raw landmarks (which become `Face.landmarks`). Same maths; the constants are passed in from config instead of hard-coded.
- `geometry(pts, hat_drop, above_head_rise)`: the prototype's `geometry()`. It gives face width (`Face.scale`), head roll (`Face.roll`) and the anchors effects draw at, offset along the face's own "down" so they follow a tilted head. The prototype returned only the hat and lip points; it now returns all of `forehead`, `hat`, `mouth` and `above_head`.
- New anchor `above_head`: above the hat, where the prototype drew the player name. The game UI (HP bar, name) should draw there.
- Not ported: the flow-point grid (`lk`) the prototype built in `measure()`. It belongs to the optical-flow fallback and comes in step 5.

**Changed:** `configs/default.yaml`, `face:`: `center_drop` (0.20), `forehead_rise` (0.50), `lip_to_nose` (0.35) and `hat_drop` (0.40), all from the prototype, plus `above_head_rise` (1.5, the hat's height in face widths).

**Added:** `tests/test_face.py`, 4 tests on synthetic YuNet rows:

- an upright face gives roll 0, the right width, and anchors in top-to-bottom order;
- a face tilted by +10° and −25° gives that roll, the same width, and anchors rotated with the head;
- swapping the two eyes in the detection changes nothing.

**Checked:**

- 29 tests pass, `ruff` clean.
- On 1000 random detections, the ported functions give exactly the same key points, centre, width, roll, hat and lip anchors as the prototype's own `measure()`/`geometry()` (max difference 0.0).

#### Step 4 · Detection on per-player head crops

**Changed:** `src/ipcvgame/face/tracker.py`: `FaceTracker` now runs YuNet; the old stub is kept as backend `stub`.

- `head_box(player, kpt_conf_threshold, crop_size)`: a square crop around each player's head from that player's pose keypoints. The centre is the mean of the visible nose, eye and ear keypoints, or else the top of the person box. The side is `crop_size` shoulder widths, or person-box widths when a shoulder is not visible.
- `_detect()`: YuNet on the crop. Detections are shifted back to full-frame pixels, and then filtered in this order:
  - narrower than `min_face` (prototype `MIN_FACE`): dropped;
  - width changed by more than `max_size_jump` since the last frame (prototype `MAX_SIZE_JUMP`): dropped;
  - of the rest, the one closest to the crop centre (the head keypoints) is used, so another player's face that falls into the same crop is skipped.
- `update()`: per player, `measure()`, then the One Euro filter on the 4 key points (`core.filters.OneEuro`, same maths and values as the prototype's own filter), then `geometry()`, then `Face`:
  - `landmarks` = the 5 raw YuNet points;
  - `center`, `scale` (face width) and `roll` from the smoothed key points;
  - `yaw = 0.0`, since 5 landmarks give no yaw;
  - `anchors` = `forehead`, `hat`, `mouth`, `above_head`;
  - `alpha = 1.0`.
- Per-player state (`_FaceState`: filter, last width, last detection time) is kept between frames. It is dropped by `reset(pid)` (called when a player re-enters), when the player disappears from `players`, or after `lost_after_s` without a detection (prototype `MAX_MISSED`, now in seconds). A returning face then starts fresh instead of sliding in.
- `draw_debug`: per player, the head crop (dashed) labelled with the YuNet score or "no face", plus the 5 raw landmarks, face centre and scale circle, and every anchor with its name.
- Replaced from the prototype: full-frame detection, de-duplication, `update_tracks()` (greedy matching), `Track.predicted()`, the velocity and `MAX_PLAYERS`. Identity now comes from Task 3 through the head crop.
- Not yet: when YuNet misses a frame, that player has no face that frame (the hat flickers). The optical-flow fallback, hold time and fade-out come in step 5.

**Changed:** `configs/default.yaml`, `face:`: `min_face` (20 px), `max_size_jump` (0.5), `kpt_conf_threshold` (0.5), `crop_size` (1.5 shoulder widths), `filter_min_cutoff` (1.5), `filter_beta` (0.1), `filter_d_cutoff` (1.0), `lost_after_s` (1.0).

**Changed:** `src/ipcvgame/app/pipeline.py`: `FaceTracker(cfg["face"])`. The face stage now reads its own `face.kpt_conf_threshold` instead of borrowing `motion.kpt_conf_threshold`.

**Changed:** `tests/test_pipeline.py`: runs with `face.backend=stub`, because the tests feed black frames and need no model file.

**Added to** `tests/test_face.py`: 5 tests:

- the head crop is centred on the head keypoints and 1.5 shoulder widths wide;
- it falls back to the person box when shoulders or keypoints are not trusted;
- there is no crop without a detection;
- an unknown backend is rejected;
- a missing model file gives an error that points to the README.

**Checked:**

- 34 tests pass, `ruff` clean.
- On two OpenCV sample photos (not in the repo), the full chain of YOLO pose → head crop → YuNet → `Face` → hat and mustache works.
  - A head tilted ~10° gets a tilted hat.
  - A 29 px face of a player far from the camera is found inside its crop.
- Face stage: 1.3 ms for a small crop, 5.8 ms for a ~350 px crop. The cost grows with crop area, so at 1280×720 with a player close to the camera, scaling the crop down before detection may be worth it. To be checked on the webcam (`face` row of the timing HUD).

#### Step 5 · Optical-flow fallback, hold time and fade-out

**Changed:** `src/ipcvgame/face/tracker.py`

- `flow_points(pts, cfg)`: the prototype's grid of points on the face (6×6, eyes to mouth), which `measure()` used to build. It is now built from the 4 key points.
- `estimate_motion(prev_gray, gray, pts, cfg)`: the core of the prototype's `Track.flow()`. Lucas-Kanade optical flow on the grid, then a similarity transform (shift + rotation + scale) fitted with RANSAC. It returns `None`, so the face is not moved, in any of these cases:
  - fewer than `flow_min_points` points tracked;
  - fewer than `flow_min_inliers` inliers;
  - a scale outside `flow_scale_range`.
- `fade(blind_s, hold_s, lost_after_s)`: the prototype's `Track.alpha()` in seconds. Fully visible for `hold_s` after the last update, then a linear fade to 0 at `lost_after_s`.
- `FaceTracker.update()`, per player, each frame:
  - **detect:** YuNet found the face. The key points, landmarks and flow points are reset from the detection.
  - **flow:** no detection, but the last detection is under `flow_max_s` old and the fit is good. The key points, flow points and landmarks are moved by the fitted transform. The prototype moved only the key points and flow points.
  - **hold:** neither. The last smoothed position is shown and fades out.
- After a detect or flow update, the key points go through the One Euro filter and `geometry()`, as in the prototype's `_apply()`.
- `Face.alpha` comes from `fade()`; `Face.valid` is `alpha > 0`. A held face is still valid, so the game can keep drawing at its anchors while it fades.
- State is now dropped after `lost_after_s` without any update (detection or flow), not just without a detection. The size-jump check uses the current width, which flow also updates, as the prototype did.
- The frame is converted to grayscale once per frame (the raw frame, before effects are drawn) and kept for the next frame's flow.
- `draw_debug`: the crop label shows the source (`detect 0.91`, `flow`, `hold`) and `a=` alpha. The flow points are drawn while flowing.
- Dropped from the prototype: `Track.vel` and `predicted()`, only used for matching (see step 4).

**Changed:** `configs/default.yaml`, `face:`

- `hold_s` (0.1 s), from `HOLD_FRAMES` = 3 frames.
- `flow_max_s` (1.3 s), from `FLOW_FRAMES` = 40 frames.
- The flow grid (`flow_grid_x`, `flow_grid_y`, `flow_grid_n`) and the Lucas-Kanade settings (`flow_win` 31, `flow_levels` 4, `flow_criteria` [20, 0.01]), as in the prototype.
- RANSAC limits: `flow_min_points` 8, `flow_min_inliers` 6, `flow_ransac_px` 3.0, `flow_scale_range` [0.7, 1.4], as in the prototype.
- `lost_after_s` comment updated.

**Added to** `tests/test_face.py`: 6 tests.

- `fade()` holds, fades and reaches 0.
- The flow grid covers the face and turns with it.
- `estimate_motion()` recovers a known shift on a textured image, and rejects a zoom outside the allowed range.
- Through `FaceTracker` with a scripted detector (no model file): flow follows a shifted face when detection misses; then hold (anchors frozen, alpha 1), fade (alpha 0.5 halfway) and gone after `lost_after_s`.

**Checked:**

- 40 tests pass, `ruff` clean.
- On `two_people_fitness_challenge.webm`, faces come from detection in 77.3% of player-frames and from flow in 18.8%; in step 4 those 18.8% had no hat. 3.8% are frames before a face was first found. On the 4K `two_people_jump.mp4` it is 99.2% detection, 0.8% flow.
- Face stage: 2.6 ms per frame on the small clip (was 2.1). On the 4K clip 24.0 ms (was 19.1); grayscale conversion of a 4K frame is part of that. Still open from step 4: scaling crops down before detection.

#### Step 6 · Evaluation: `log_fields` and `tools/eval_face.py`

The prototype's `Metrics` class (jitter, alignment error, detection rate per lighting, latency) moves out of the tracker. The tracker now writes per-frame CSV columns, and a separate script computes the metrics from them.

**Changed:** `src/ipcvgame/face/tracker.py`

- `log_fields(state)`: per player, the columns `face_p<pid>_source` (detect/flow/hold, empty while no face), `_score` (YuNet score on detect frames), `_alpha`, `_scale`, `_roll_deg`, `_mouth_x/_y` (smoothed anchor), `_mouth_raw_x/_y` (unsmoothed upper-lip point, on detect frames only) and `_crop_px`. The prototype logged the same raw-versus-smoothed lip pair.
- The raw detection and crop box are now recorded in `update()` instead of `_detect()`, so they are also there when the detector is replaced in tests.

**Added:** `tools/eval_face.py`

- Takes clips, which it runs headless up to the face stage, writing `logs/eval/face-<clip>[-<tag>].csv`; or it takes existing logs.
- Prints one Markdown row per clip and player:
  - coverage: detect / flow / hold / no face %;
  - jitter: the prototype's second-difference RMS, raw versus smoothed, in px and % of face width;
  - lag: the prototype's alignment error (mean, p95, % face width);
  - crop size and face-stage time.
- `--set` and `--tag` run parameter experiments; `--markdown` saves the table.
- Differences from the prototype: conditions come from clip names instead of `s`/`d`/`l` key presses. Latency is the face-stage time from the profiler (`ms_face`). Identity switches are no longer counted here: faces cannot swap with head crops, and switches are Task 3's metric.

**Added:** `docs/evaluation.md`: how to evaluate on the clips for the report. The workflow and run-log columns are for all tasks; the face section has commands, column definitions, how to read the numbers, and an example. It ends with how to add an `eval_<stage>.py`.

**Added:** `tests/test_eval_face.py` (3 tests: the metrics on a synthetic log with known jitter, lag and coverage; missing frames count as "no face"; the Markdown table). Also added to `tests/test_face.py`: a test that raw mouth columns are only written on detect frames.

**Checked:**

- 44 tests pass, `ruff` clean.
- Both test clips evaluate end to end. On the fitness clip the filter cuts jitter from 2.6 to 1.7 px (P1) at about 2 px of lag. A `--tag noflow --set face.flow_max_s=0` run turns the 23.8% flow frames into hold frames and saves about 0.4 ms per frame.

#### Step 7 · Cleanup

**Deleted** (superseded by `src/ipcvgame/face/`; still in git history):

- `hat_tracker.py`, `hat_tracker_v2.py`: the earlier standalone MediaPipe versions of the face script.
- `hat.png`, `mustache.png` in the repo root: byte-identical to `assets/face/`.

**Changed:**

- `docs/phase-1/facetracking-update-guide.md`: a "History" note at the top. It says the guide was for the standalone script, where its `paste()` now lives, and links here.
- `docs/phase-1/status.md`:
  - the module table now shows the face stage as done;
  - Task 1's "Stub today" became a status paragraph and a checklist: done items ticked, next steps 8–10 listed;
  - the old cross-section read note is replaced (the face stage has its own `kpt_conf_threshold`);
  - a face row in the timing table;
  - the anchor-name decision ticked (`forehead`, `mouth`, `hat`);
  - the stub-limitation line about the face dot removed;
  - test count updated to 44.
- `README.md`: `assets/` in the layout; YuNet (MIT, OpenCV model zoo) in the licence section.

**Not changed:** `.gitignore` keeps `garb/` until the whole Colour Clash merge is done, not just Task 1.

**Checked:** nothing outside `garb/` and the history note refers to the deleted files. 44 tests pass, `ruff` clean.

#### Step 8 · Shrink large head crops before detection

**Changed:** `src/ipcvgame/face/tracker.py`, `FaceTracker._detect()`. A crop whose longest side is over `crop_max_px` is shrunk (`cv2.resize`, `INTER_AREA`) before YuNet runs. The detection is scaled back to crop pixels and then shifted to frame pixels. Everything after that (`min_face`, the size-jump check, `measure()`) still works in frame pixels, so nothing else changes.

**Changed:** `configs/default.yaml`, `face:`: `crop_max_px: 192` (`null` = never shrink).

**Measured** with `tools/eval_face.py --tag max<N> --set face.crop_max_px=<N>`. Face-stage time per frame, and the share of player-frames found by detection:

| `crop_max_px` | 4K `two_people_jump.mp4` (crops ~480 px) | `two_people_fitness_challenge.webm` (crops ~100–117 px) |
| --- | --- | --- |
| none | 23.4 ms, detect 98.4 / 100% | 2.59 ms, detect 76.2 / 78.5% |
| 320 | 15.2 ms, 98.4 / 100% | 2.64 ms, unchanged |
| 240 | 11.6 ms, 98.1 / 100% | 2.67 ms, unchanged |
| **192** | **9.2 ms, 97.8 / 100%** | 2.60 ms, unchanged |
| 160 | 8.2 ms, 97.8 / 100% | 2.61 ms, unchanged |
| 128 | 7.3 ms, 97.8 / 100% | 2.46 ms, unchanged |

Why 192:

- Below 192 the gain flattens. What remains at 4K is mostly the full-frame grayscale conversion for optical flow.
- A face is about 0.27 of the crop width, so at 192 it is still ~50 px in YuNet's input. At 128 it would be ~34 px, and the landmarks would lose more precision.
- Small crops (the fitness clip, a webcam far away) are not touched.

Open: the landmark precision cost of shrinking shows best as jitter on a still clip. The two test clips are both dynamic; check on the shared `normal` clip.

**Added to** `tests/test_face.py`: a test with a stand-in detector that finds a face in the middle of whatever crop it gets. Without a limit and with `crop_max_px=64`, the returned face is the same in frame pixels, and YuNet really receives the 64 px crop.

**Changed:** `docs/phase-1/status.md` (face timing row, Task 1 checklist, test count) and the Task 1 modifications in "Change Overviews" above.

**Checked:** 46 tests pass, `ruff` clean.

#### Step 9 · Head pose and 3D anchors

**Why the scope grew.** The plan was to fill `Face.yaw`. Testing step 8 on the webcam showed the hat lifting off the head when the chin goes up. The cause is the prototype's 2D geometry: the forehead and hat sit at fixed offsets above the eyes, whatever way the head points. When the chin goes up, the top of the head turns away from the camera and moves *down* in the image, but the hat stays put. When the chin goes down, the hat sinks onto the forehead. So step 9 estimates the full head pose and places those anchors in 3D.

**Changed:** `src/ipcvgame/face/tracker.py`

- `camera_matrix(w, h, focal_per_width)`: pinhole intrinsics without calibration. The focal length is `focal_per_width` × the image width (1.0 ≈ 53° horizontal field of view), and the principal point is the image centre.
- `head_pose(landmarks, model_mm, K)`: `cv2.solvePnP` with the SQPnP solver, which works from 4 points without an initial guess (the iterative solver needs 6). It fits the canonical 3D head to the 5 YuNet landmarks. Eyes and mouth corners are matched by image position, not by YuNet's labels, so the mirrored frame works. It returns `None` without a valid solution, or when the head would be behind the camera.
- `pose_angles(rvec)`: yaw and pitch from the rotation, as seen on screen: yaw > 0 = face turned to the image right, pitch > 0 = chin up.
- `anchor_points_mm(cfg, model_mm)`: `forehead`, `hat` and `above_head` as 3D points on the head. They come from the existing `forehead_rise`, `hat_drop` and `above_head_rise`, so when the face looks straight at the camera they land where the 2D anchors were. The hat points sit `hat_depth_mm` behind the eyes, so they move like a hat on the head.
- `FaceTracker._smoothed_geometry()`, after every detect or flow update:
  1. the 2D anchors from `geometry()`;
  2. if `head_pose` is on and the pose is within `pose_max_deg`: the forehead, hat and `above_head` anchors are replaced by the projected 3D points (`cv2.projectPoints`);
  3. One Euro on the face sides and the anchors together;
  4. width, roll and centre from the smoothed sides.
  - Yaw and pitch have their own One Euro filter, and `Face.yaw` is filled from it.
  - The mouth anchor and the roll stay as measured (landmarks and eye line).
- `log_fields`: `face_p<pid>_yaw_deg`, `_pitch_deg`, `_pose` (1 = anchors from the head pose, 0 = 2D fallback), and `_hat_x`/`_hat_y`.
- `draw_debug`: yaw and pitch in the label, and a 40 mm head axis triad drawn from the nose (x red, y green, z blue).

**Changed:** `configs/default.yaml`, `face:`

- `head_pose: true` (`false` = the 2D anchors only, for comparison).
- `head_model_mm`: the canonical head. Eyes 63 mm apart. The nose tip (34 mm below the eyes) and mouth corners (62 mm below, 54 mm apart) match YuNet's median layout on 73 detections from the two test clips and two photos. Depths are from a generic face model (nose tip 27 mm in front of the eyes).
- `box_per_eye: 2.16`: YuNet box width in eye distances, the median of the same detections.
- `hat_depth_mm: 50`, `focal_per_width: 1.0`, `pose_max_deg: 60`.
- `pose_filter_min_cutoff: 1.0`, `pose_filter_beta: 0.5`: angle filter values from the update guide.

**Changed:** `tools/eval_face.py`: new columns `pose %` and `hat jitter px`; `docs/evaluation.md` explains both.

**Added to** `tests/test_face.py`: 9 tests on a synthetic head projected with known rotations, 1.5 m from the camera.

- Yaw and pitch are recovered within 0.5° for five poses.
- Swapping the eye and mouth-corner labels (a mirrored frame) gives the same pose.
- Facing the camera, the 3D forehead and hat anchors land on the 2D ones (under 1% of a face width). `above_head` is within 6%: it is 1.5 face widths up and deeper in the head, so perspective puts it slightly lower.
- The reported bug as a test: chin up 25°, and the hat is closer to the eyes than the 2D anchor put it. Chin down 25°, and it is higher.
- The step 8 crop test was updated for the new filter argument.

**Measured** on the two test clips, `--set face.head_pose=true/false`:

| Clip | Player | pose % | hat jitter px (2D → 3D) | face ms (2D → 3D) |
| --- | --- | --- | --- | --- |
| `two_people_fitness_challenge` | P1 | 100 | 1.98 → 2.33 | 2.60 → 2.90 |
| `two_people_fitness_challenge` | P2 | 100 | 2.93 → 2.92 | |
| `two_people_jump` (4K) | P1 | 100 | 10.5 → 12.0 | 9.24 → 9.54 |
| `two_people_jump` (4K) | P2 | 100 | 9.42 → 10.2 | |

- The pose was used on every detect and flow frame.
- Yaw ranges: fitness clip P1 +16° to +54°, P2 −45° to +12°; pitch between −31° and +3°.
- Frame to frame, the smoothed yaw changes by a median of 0.4–0.8°.
- The hat moves a little more (up to +18%). Part of that is the hat now following real head movement, which these moving clips are full of. The cost is 0.3 ms.
- On a photo with the head turned (yaw +40°), the hat moves towards the middle of the head instead of staying on the face outline.

**Open:** check on the webcam (chin up, chin down, turning). `hat_depth_mm` is a first guess; a larger value moves the hat more with tilt. The mouth anchor is not affected by the pose.

**Checked:** 55 tests pass, `ruff` clean.

#### Step 10 · MediaPipe: head rotation, mouth opening and blinks

**Why.** Testing step 9 on the webcam, the hat still lifted off during a strong chin-up. On that frame, YuNet's 5 landmarks were almost frontal: nose 0.49 eye distances below the eyes (frontal 0.54), mouth 0.99 (frontal 0.99). `solvePnP` therefore found −3° of pitch. MediaPipe Face Landmarker, which fits 478 landmarks, found 33°, which matches the picture. Adding the pose model's ears to `solvePnP` gave 15–21°: better, but not enough. Mouth opening and blinking need MediaPipe anyway (step 10), so its head rotation is used as well.

**Contract (team heads-up):** `src/ipcvgame/core/types.py`: `Face.blendshapes: dict[str, float]`, default empty. These are MediaPipe's 52 scores from 0 to 1 (`jawOpen`, `eyeBlinkLeft`, `eyeBlinkRight`, `mouthSmileLeft`, `cheekPuff`, …), empty when not measured. MediaPipe names the left/right scores as if the image were not mirrored, so on our mirrored frame `eyeBlinkLeft` is the player's own right eye.

**Added:** `src/ipcvgame/face/landmarker.py`, `FaceLandmarker`. A thin wrapper around MediaPipe Face Landmarker in IMAGE mode: every call stands alone, because the crops come from different players in turn. It returns the blendshape scores and the head transformation matrix for one face crop, or `None`.

**Added:** `src/ipcvgame/face/head_pose.py`. The step 9 functions moved here from `tracker.py`, to keep that file readable. New in this step:

- `head_pose(..., ears, ears_mm)`: optional ear keypoints from the pose model as two more points.
- `mediapipe_angles(matrix)`: yaw and pitch from MediaPipe's matrix. Its face frame has y up and z towards the camera, so both axes are flipped (`C R C`, `C = diag(1, −1, −1)`).
- `rotation(yaw, pitch, roll)`: the inverse of `pose_angles()`, plus roll.
- `translation(R, model_mm, points, K)`: the head position for a known rotation, by linear least squares on the pinhole equations. The rotation comes from MediaPipe (which ran on a crop), and the position is fitted to the YuNet landmarks in the full frame.
- `sorted_landmarks()`.

**Changed:** `src/ipcvgame/face/tracker.py`

- `update()` now runs in three passes:
  1. find each face: detection, else optical flow (unchanged);
  2. MediaPipe for `landmarker_per_frame` players, the one that waited longest first (`_landmark()`), on a crop of `landmarker_crop` face widths around the face;
  3. smoothed geometry and the `Face` output, now with `blendshapes` while the last MediaPipe result is under `landmarker_max_age_s` old.
- `_smoothed_geometry()`: the rotation comes from MediaPipe while its last result is recent, else from `solvePnP` on YuNet's landmarks plus both ears (when the pose model trusts them).
  - Only yaw and pitch are taken from either source, and they go through the pose One Euro filter.
  - Roll comes from the eye line, as before.
  - The position comes from `translation()`.
  - The 3D anchors are then projected as in step 9.
  - Both pose sources go through the same steps, so switching between them does not make the hat jump.
- `geometry(pts, hat_rise, above_head_rise)`: the hat brim is now `hat_rise` above the forehead point (see the effects change below).
- `log_fields`:
  - `face_p<pid>_pose` is now the source: `mediapipe`, `pnp` or empty for 2D;
  - new `_jaw_open`, `_blink` (mean of both eyes), `_smile` (mean of both sides);
  - new `_mp` (1 = MediaPipe result this frame).
- `draw_debug`: the pose source in the label, and `jaw` and `blink` bars next to the face.

**Changed:** `src/ipcvgame/face/effects.py`. The hat now hangs from its visible brim: the lowest row of `hat.png` that is not transparent. `hat.png` has 34% transparent padding below the hat, so with the prototype's pivot at the bottom of the image, the visible brim floated half a face width above the anchor. Any pitch error made that look worse.

**Changed:** `configs/default.yaml`, `face:`

- `hat_drop: 0.40` became `hat_rise: 0.11`: the brim sits 0.11 face widths above the forehead point. That is −0.40 + 0.51 (34% padding × 1.5 face widths), so the hat looks the same as before when facing the camera.
- `above_head_rise: 1.5` became `1.0`, now measured from the brim; `above_head` stays 1.6 face widths above the eyes.
- `head_model_mm` gets `left_ear` and `right_ear` at (±72, 20, 85) mm: beside the head, at ear-canal height, behind the eyes.
- New: `landmarker` (true), `landmarker_model`, `landmarker_min_conf` (0.5), `landmarker_crop` (2.0), `landmarker_max_side` (256), `landmarker_per_frame` (1), `landmarker_max_age_s` (0.25).

**Changed:** `README.md`: download command and SHA-256 for `models/face_landmarker.task`. `tools/eval_face.py`: `pose %` now counts both pose sources, plus a new `mediapipe pose %`; `docs/evaluation.md` explains them.

**Tests:** 67 pass, `ruff` clean.

- `tests/test_face.py`:
  - MediaPipe's matrix gives our yaw and pitch for 5 poses, including the 33° chin-up;
  - `rotation()` inverts `pose_angles()`;
  - `translation()` recovers a known position;
  - with frontal-biased landmarks, the 5-point fit misses a 33° chin-up and the ears recover it;
  - with a stand-in landmarker and two players, MediaPipe runs once per frame, in turn (P1, P2, P1, P2), and its blendshapes and pitch reach `Face` and the pose.
- The frontal test now checks the exact perspective offset of the deeper hat points (`Z / (Z + depth)`) instead of a tolerance.
- `tests/test_effects.py`: the hat hangs from its brim on a padded sprite.
- Existing tests: imports moved to `head_pose.py`, `hat_rise`, and `landmarker: False` where YuNet is replaced.

**Measured**

On the chin-up screenshot:

| | Pose | Pitch | Brim above the eyes |
| --- | --- | --- | --- |
| Step 9 | YuNet `solvePnP` | −3° | 61 px (the floating hat) |
| Step 10 | MediaPipe | +33° | 8 px; the hat sits on the head |

On the test clips:

| Clip | Player | pose % | mediapipe pose % | hat jitter px | face ms (mean / p95) |
| --- | --- | --- | --- | --- | --- |
| `two_people_fitness_challenge` | P1 | 100 | 83.8 | 1.95 | 10.5 / 13.0 |
| `two_people_fitness_challenge` | P2 | 100 | 99.6 | 3.43 | |
| `two_people_jump` (4K) | P1 | 100 | 99.2 | 11.2 | 19.2 / 20.2 |
| `two_people_jump` (4K) | P2 | 100 | 99.7 | 10.0 | |

- MediaPipe costs ~8 ms per frame for one player (CPU), so the face stage went from ~3 to ~10.5 ms. With pose (~10 ms) and render that is ~23 ms per frame at 720p, enough for 30 FPS.
- Pitch now spans ±30° on the fitness clip; with YuNet `solvePnP` (step 9) it was −31° to +3°.
- Hat jitter: P1 is lower than with the step 9 pose (2.33). P2 is higher than in step 9 (2.92), probably from switching between the two pose sources on hard frames.
- Blendshapes are present on 84–100% of frames. `jawOpen` reaches 0.61 on the fitness clip.
- `blink` goes over 0.5 in up to 26% of the frames of the jump clip: eyelids also drop when people look down. A blink event (for the super charge) needs hysteresis, a per-player baseline, and probably a pitch check. This is noted for `docs/extra_features.md`.

**Open:**

- Check on the webcam: chin up and down, blinking, mouth opening (the `jaw` and `blink` bars in the debug view).
- MediaPipe's GPU delegate may cut its ~8 ms; not tried yet.
- With `landmarker_per_frame: 1`, two players are measured at ~15 Hz each. Set it to 2 if short blinks are missed.

#### Step 10 fix · Crash in `solvePnP` on degenerate points

**What happened.** The live game crashed with `cv2.error ... sqpnp.cpp:236: (-215:Assertion failed) point_coordinate_variance >= POINT_VARIANCE_THRESHOLD`, raised from the `solvePnP` fallback in `head_pose()`.

**Cause.** OpenCV's SQPnP solver asserts, instead of failing softly, when the image points (nearly) coincide or are not finite. Reproduced offline: it fails for a face a few pixels wide or a NaN coordinate, and works for normal and off-image points. YuNet never reports a face under `min_face` (20 px), so the likely route in a live run is optical flow: each step may shrink the face by up to `flow_scale_range[0]` (0.7×), and over `flow_max_s` that can compound to a few pixels. The exact frame of the crash was not recorded.

**Changed:**

- `src/ipcvgame/face/head_pose.py`, `head_pose()`: returns `None` (2D anchors for that frame) for non-finite points or eyes under 2 px apart, and for any `cv2.error` from the solver. A pose failure can no longer stop the game.
- `src/ipcvgame/face/tracker.py`, flow branch of `update()`: a flow step that would shrink the face below `min_face` is not applied. The face holds and fades instead, the same as when flow fails.

**Added to** `tests/test_face.py`:

- `head_pose()` returns `None` for NaN and for collapsed points (the crash case);
- flow that shrinks the face by 25% per frame stops (→ hold) before the face drops below `min_face`.

**Checked:** 70 tests pass, `ruff` clean.

#### After step 10 · Backlog: `docs/extra_features.md`

**Added:** `docs/extra_features.md`. The game features agreed during the merge, written up for after it, each with owners, inputs from the face stage, a starting design with thresholds, and how to check it:

1. team colour;
2. hit reaction;
3. status above the head;
4. super charge that a blink resets;
5. taunt;
6. Smash-Bros-style consumables (eat falling items with a chomp, or grab them from the floor);
7. cheek bulge.

It also lists the merge steps still needed first (gestures, game logic, renderer, identity), the technical follow-ups from Task 1, and ideas considered but not planned. No code changed.

## Task 4: Gestures

### Plan

**Source.** `Project/bodypose.py`: `KeypointTracker`, `Debounce`, the geometry helpers and the gesture functions, and how `main.py` turns their output into game actions. Its identity code (`PlayerIdentityTracker`) and its own detection loop belong to the identity merge.

**What the code does.**

- **Smoothing:** `KeypointTracker` treats keypoints under confidence 0.35 (`CONF_T`) as missing. A missing point keeps its last position for 10 frames (`MISSING_FRAMES`), and the rest are smoothed with an EMA (α = 0.5). A point that comes back after being lost jumps to the detection instead of sliding in from a stale spot.
- **Gestures,** all in shoulder widths and only when the shoulders are more than 10 px apart:
  - `hands_up`: both wrists above the nose, or above a point 0.3 shoulder widths above the shoulders when the nose isn't seen. Must hold for 2 frames (`CONFIRM_FRAMES`).
  - `crossed`: wrists in the opposite left/right order to the shoulders (independent of mirroring), both between chest and hips, and closer than 1.5 shoulder widths. Must hold for 2 frames.
  - `punch`: one arm straight (elbow > 150°), reaching more than 1 shoulder width, at chest height, with the other arm bent (< 120°). Stays on for 0.5 s after the last detection (`PUNCH_HOLD_S`).
- **Use in `main.py`:** shield and beam fire on the frame `crossed` / `hands_up` turns on. `game.attack()` is called on every frame the punch is on; the game's 1.8 s cooldown limits how often that fires. `punch_side`, `hand_x` and `hand_y` are computed but not used by the game or renderer.

**Decisions (Task 5 owner, 2026-10-25).**

- **Smoothing goes to `pose/motion.py` (Task 2),** where the architecture puts it. Gestures, the face stage and the game then all read the same held and smoothed keypoints.
- **Events on the rising edge** for all three gestures. Shield and beam behave as before. A punch thrown during the 1.8 s cooldown is lost, instead of firing when the cooldown ends.
- **Punch side dropped.** Nothing uses it yet, and with mirroring the model's `L` is the player's own right; the event is just `punch`.

**Where each part goes.**

| In the code | Goes to | Change |
| --- | --- | --- |
| `KeypointTracker` | `pose/motion.py`, `MotionFilter` | Frames to seconds; EMA from `core.filters` |
| `angle_at()` | `core/geometry.py` | Shared helper |
| `lower_body_y()`, `detect_hands_up()`, `detect_crossed()`, `arm_extended()`, `arm_bent()`, `detect_punch()` | `interaction/gestures.py`, module level | Same rules; numbers from config |
| `Debounce`, `punch_until`, `BodyPoseProcessor.process()` (gesture part) | `interaction/gestures.py`, `GestureRecognizer` | Seconds instead of frames; rising-edge `GestureEvent`s |
| `BodyPoseProcessor.reset_player()` | `MotionFilter.reset(pid)`, `GestureRecognizer.reset(pid)` | Also called when a player re-enters |
| `hands_were_up`, `arms_were_crossed` in `main.py` | `GestureRecognizer` | The rising edge is detected in the gesture stage |
| `draw_pose()`, gesture text under the box | `MotionFilter.draw_debug()`, `GestureRecognizer.draw_debug()` | |
| Per-crop pose model call | Already in `pose/estimator.py` | The pipeline runs the pose model once on the full frame |
| `center_x`, `torso_y`, `head_x/y`, `hands_center_x/y`, `hand_x/y` | Game merge | Positions for the game, computed from `state.motion` and `state.faces` |

**Steps.** Each step gets an entry under Changes below.

1. Motion smoothing: `KeypointTracker` into `MotionFilter`; the pass-through stays as `motion.backend: stub`.
2. Gesture rules: the six functions, `angle_at()` in `core/geometry.py`, thresholds in the `gestures:` config, tests on synthetic skeletons.
3. `GestureRecognizer`: timers in seconds, rising-edge events `punch`, `hands_up`, `arms_crossed`, reset on loss and re-entry, `draw_debug`, `log_fields`.
4. Check and record: event counts on the two test clips, Change Overviews, `docs/phase-1/status.md`.

**Verification.** Unit tests for every step; event counts on `data/test_clips/`. The Task 5 owner checks the gestures live on the webcam.

### Changes

#### Step 1 · Motion smoothing: `KeypointTracker` into `MotionFilter`

**Changed:** `src/ipcvgame/pose/motion.py`

- `MotionFilter` now does the prototype's `KeypointTracker.update()` per player (`_smooth()`, state in `_PlayerState`), with the same five rules:
  - keypoints under the confidence threshold count as missing;
  - a missing point keeps its last position and stays valid for `hold_s`;
  - an EMA smooths the positions;
  - a point that comes back after more than `hold_s` jumps to the detection;
  - a frame without a detection resets the player (the prototype reset when the pose model found nobody in the crop).
- The missing-frame counter became the time each keypoint was last seen (`last_good`, from `Frame.t`), so the hold is the same at any frame rate.
- Velocities are still the finite difference on `Frame.t`, now of the smoothed points. A point that jumps back gets velocity 0 on that frame, so the jump doesn't count as motion.
- The phase-1 pass-through stays as `backend: stub`, unchanged except for the new confidence threshold.

**Changed:** `src/ipcvgame/core/filters.py`: `EMA.__call__` takes an optional `snap` mask; masked rows jump to the new value instead of being smoothed. Without the mask it behaves as before, so the face stage is unaffected.

**Changed:** `configs/default.yaml`, `motion:` section

| Key | Value | From the prototype |
| --- | --- | --- |
| `backend` | `ema` | (new; `stub` = pass-through) |
| `kpt_conf_threshold` | 0.35 (was 0.5) | `CONF_T`; the gesture thresholds were tuned with it |
| `tau` | 0.048 s | `ALPHA = 0.5` per frame at 30 FPS: τ = −dt / ln(1 − α) |
| `hold_s` | 0.33 s | `MISSING_FRAMES = 10` at 30 FPS |

**Added:** `tests/test_motion.py` (7 tests: first detection snaps, EMA step after one time constant, hold → invalid → snap back, reset on no detection and on re-entry, stub pass-through, unknown backend). `tests/test_filters.py`: one test for the snap mask.

**Changed:** `tests/test_pipeline.py`, `test_velocity_uses_frame_time`: runs with `motion.tau=0` (no smoothing), so it still checks the plain finite difference.

**Checked:** 78 tests pass, `ruff` clean. On the first 300 frames of each test clip (jitter = median second difference of valid keypoints, which includes real motion):

| Clip | Backend | Jitter | Valid keypoints | Motion stage |
| --- | --- | --- | --- | --- |
| `two_people_fitness_challenge.webm` | stub | 3.25 px | 95.9% | 0.04 ms |
| | ema | 0.89 px | 98.5% | 0.08 ms |
| `two_people_jump.mp4` (4K) | stub | 12.34 px | 99.3% | 0.06 ms |
| | ema | 8.27 px | 100.0% | 0.11 ms |

#### Step 2 · Gesture rules

**Changed:** `src/ipcvgame/interaction/gestures.py`

- The prototype's `lower_body_y()`, `detect_hands_up()`, `detect_crossed()`, `arm_extended()`, `arm_bent()` and `detect_punch()` are module-level functions here, with the same names, logic and comments.
- They take `(kpts, valid, sw, cfg)`: the keypoints and validity from `Motion`, the shoulder width in pixels, and the `gestures:` config. Keypoints are indexed by name through `core/skeleton.py` instead of the file's own index constants.
- `detect_punch()` still returns `"L"`, `"R"` or `None`; its docstring notes that the model's `L` is the player's own right arm on our mirrored frame. The event will drop the side (step 3).
- `GestureRecognizer` is still the stub; step 3 connects it.

**Changed:** `src/ipcvgame/core/geometry.py`: new `angle_at(a, b, c)`, the prototype's helper, so other tasks can use it too.

**Changed:** `configs/default.yaml`, `gestures:` section, each with a comment:

| Key | Value | In the prototype |
| --- | --- | --- |
| `head_above_shoulders` | 0.3 | `0.3 * sw` in `detect_hands_up()` |
| `hip_below_shoulders` | 2.0 | `2.0 * sw` in `lower_body_y()` |
| `chest_above_shoulders` | 0.5 | `0.5 * sw` in `detect_crossed()` |
| `cross_max_wrist_dist` | 1.5 | `CROSS_MAX_WRIST_DIST` |
| `punch_elbow_angle` | 150° | `PUNCH_ELBOW_ANGLE` |
| `punch_min_reach` | 1.0 | `PUNCH_MIN_REACH` |
| `punch_above_shoulders` | 1.0 | `1.0 * sw` in `arm_extended()` |
| `punch_bent_angle` | 120° | `PUNCH_BENT_ANGLE` |

Values are from `Project/bodypose.py`, which `main.py` runs. `garb/rest/pose_detector.py` is an older single-player copy of the same code (`CROSS_MAX_WRIST_DIST` 1.2, `CONFIRM_FRAMES` 3); it is not merged.

**Added:** `tests/test_gestures.py` (15 tests on synthetic skeletons): `angle_at`, the hip fallback, standing (no gesture), hands up with and without the nose, crossed (also mirrored, too far apart, overhead), punch left and right, T-pose, unknown guard arm, short reach, raised arm, hanging arm with a guard.

**Checked:** 93 tests pass, `ruff` clean. How often each rule is true per player-frame on the two test clips (smoothed keypoints from step 1, no hold time or events yet). Neither clip has real punches, crossed arms or hands up as gestures, so every hit is a false positive:

| Clip | Player-frames | Hands up | Crossed | Punch |
| --- | --- | --- | --- | --- |
| `two_people_fitness_challenge.webm` | 495 | 3.6% | 0.0% | 4.8% |
| `two_people_jump.mp4` | 730 | 0.0% | 0.0% | 33.4% |

**Found: a hanging arm reads as a punch.** In the jump clip, the runners' arms hang straight (elbow 159–177°), so they pass the straightness check. The wrist is 1.0–1.4 sw from the shoulder, so it passes the reach check. The wrist is just above the hip keypoint, so it passes the height check. Meanwhile the other arm swings bent. `arm_extended()` checks the distance from the shoulder but not the direction, and the lowest allowed wrist position is the hip line. On all 268 false punch frames across both clips, the wrist is at most 0.96 sw to the side of its shoulder (median 0.19–0.23); a sideways punch at the other player reaches at least `punch_min_reach` (1 sw) sideways. Not changed in this step; the decision is for the Task 5 owner.
