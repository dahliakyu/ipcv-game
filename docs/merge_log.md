Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Merge log: Colour Clash prototype into the shared codebase

This log records every change made while merging the Colour Clash prototype (the `Colour Clash/` folder) into the shared codebase. The merge is done by the Task 5 owner on branch `face-tracking-eline` and goes to `main` in a single PR when it is complete.

Each entry names the files it touched, so owners can check what changed in their own code.

## Pre-merging Inits

### Clean-up of the `Colour Clash/` folder

The folder arrived as a Windows zip of 1.4 GB and ~33,800 files; after the clean-up it holds 24 MB and 15 files. No source code was changed in this step.

**Permissions.** Every directory came out of the zip read-only (`dr-xr-xr-x`), so nothing in it could be deleted. All directories were made writable for their owner (`chmod u+w`).

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
