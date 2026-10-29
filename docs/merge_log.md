# Merge log

This log records every change made while merging the teammates' Colour Clash code (the `Colour Clash/` folder, kept locally in `garb/rest/` during the merge) into the shared codebase. The merge is done by the Task 5 owner on branch `face-tracking-eline`o owners can check what changed in their own code.

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
- Every other import in the teammates' code (`ultralytics`, `torch`, `torchvision`, `cv2`, `numpy`) was already pinned.

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

**Action for anyone who installed it:** `torchreid`, `gdown` and `tensorboard` are no longer needed and can be removed with `pip uninstall torchreid gdown tensorboard tensorboard-data-server`.

### Open issues found before the merge

Found while preparing the merge, in the `Colour Clash/` folder (now `garb/rest/`, ignored by git, local only). Each was fixed in its task's merge, as noted.

- **Two detectors per frame (Task 3, Task 2).** Fixed in the identity merge (step 3): BoT-SORT runs on the pose model's boxes.
- **Stored appearance is never updated (Task 3).** Fixed in the identity merge (step 2). Matching is still greedy per track, not a joint assignment (Hungarian); with two players and linked tracks handled first, that rarely matters.
- **`Colour Clash/Project/test_osnet.py` fails as written.** Replaced by `tests/test_identity.py` (step 1).
- **Working directory.** Gone: the code now lives in `src/ipcvgame/`.

## Change Overviews

A quick reference for each teammate: where your code now lives, and what was changed beyond moving it. The task sections below have the step-by-step details and the reasons.

### Refactor: where to find your code

#### Task 1: `face_track/facial_tracking_yunet.py`

**Functions and classes**

| In your code | Now in | Note |
| --- | --- | --- |
| `measure()` | `face/tracker.py`, `measure()` | Same maths; constants come from config |
| `measure()`, the `lk` grid part | `face/tracker.py`, `flow_points()` | Built from the 4 key points |
| `geometry()` | `face/tracker.py`, `geometry()` | Also returns `forehead` and `above_head`; the fallback when the head pose is off or fails (Task 1 modifications) |
| (new, Task 1 modifications) head pose | `face/head_pose.py`: `camera_matrix()`, `head_pose()`, `pose_angles()`, `rotation()`, `mediapipe_angles()`, `translation()`, `anchor_points_mm()`; used in `face/tracker.py`, `FaceTracker._smoothed_geometry()` | New |
| (new, Task 1 modifications) mouth, blinks, MediaPipe pose | `face/landmarker.py`, `FaceLandmarker`; `face/tracker.py`, `FaceTracker._landmark()` | New |
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
| Keys `q` | `app/main.py` (`q`/`Esc`; also `p` pause, `r` new round, `R` full reset) | |
| `draw_status()` (name + HP bar) | `scene/renderer.py`, `_draw_player_health()` | Game UI: "PLAYER N" and the health bar, pinned to the bottom (Task 5 modifications) |
| `Track.predicted()`, `Track.vel` | Removed | Only used for matching |
| Keys `s`, `d`, `l`, `x`, `1`/`2` | Removed | Conditions come from clip names; HP demo is game logic |

**Constants**

All of these are in `configs/default.yaml`, `face:` section, unless another section is named.

| In your code | Now |
| --- | --- |
| `MODEL`, `MODEL_URL` | `model` (download command in `README.md`) |
| `SCORE_THRESH` | `score_threshold` |
| `MIN_FACE` | `min_face` |
| `MAX_SIZE_JUMP` | `max_size_jump` |
| `HOLD_FRAMES`, `FLOW_FRAMES`, `MAX_MISSED` | `hold_s`, `flow_max_s`, `lost_after_s` (seconds) |
| `LK_GRID` | `flow_grid_x`, `flow_grid_y`, `flow_grid_n` |
| `LK_PARAMS` | `flow_win`, `flow_levels`, `flow_criteria` |
| Numbers inside `Track.flow()` (8 points, 6 inliers, 3.0 px, 0.7–1.4) | `flow_min_points`, `flow_min_inliers`, `flow_ransac_px`, `flow_scale_range` |
| Numbers inside `measure()` / `geometry()` (0.20, 0.50, 0.35, 0.40) | `center_drop`, `forehead_rise`, `lip_to_nose`; the 0.40 became `hat_rise` (0.11), see Modifications |
| Numbers inside `Track.draw()` (1.5, 0.80) | `hat_width`, `mustache_width` |
| `OneEuro(1.5, 0.1)` | `filter_min_cutoff`, `filter_beta`, `filter_d_cutoff` |
| `HAT`, `MUSTACHE` (`hat.png`, `mustache.png`) | `hat_sprite`, `mustache_sprite` → `assets/face/` |
| `CAP_W`, `CAP_H`, `CAMERA_INDEX` | `source:` section (`width`, `height`, `uri`) |
| `MAX_PLAYERS` | `identity:` section (`max_players`), Task 3 |
| `COLORS` | `scene/draw.py`, `PLAYER_COLORS`, Task 5 |
| `MATCH_FACTOR`, `LIGHTS`, `EXPOSURE` | Removed (matching replaced; conditions from clip names; exposure not ported) |

#### Task 2 and Task 4: `Project/bodypose.py` (smoothing and gestures)

**Functions and classes**

| In your code | Now in | Note |
| --- | --- | --- |
| `KeypointTracker` | `pose/motion.py`, `MotionFilter._smooth()` + `_PlayerState` | Task 2; same rules, in seconds |
| `KeypointTracker.reset()` | `pose/motion.py`, `MotionFilter.reset(pid)` | Also on re-entry |
| `draw_pose()` | `pose/motion.py`, `MotionFilter.draw_debug()` | Shared `draw_skeleton()` |
| `angle_at()` | `core/geometry.py`, `angle_at()` | Shared helper |
| `Debounce`, `punch_until`, `punch_side` | `interaction/gestures.py`, `_PlayerState` + `GestureRecognizer.update()` | Seconds instead of frames; side dropped |
| `BodyPoseProcessor.process()`, gesture part (`sw > 10`, the three rules) | `interaction/gestures.py`, `shoulder_width()` + `detect_all()` | |
| `BodyPoseProcessor.reset_player()` | `MotionFilter.reset(pid)` + `GestureRecognizer.reset(pid)` | Also on re-entry |
| `hands_were_up`, `arms_were_crossed` in `main.py` | `GestureRecognizer.update()` | Rising-edge `GestureEvent`s |
| Gesture text under the box (`H:C:P`) | `GestureRecognizer.draw_debug()` | |
| `center_x`, `torso_y`, `hands_center_x/y` | `interaction/game.py`, `Game.body_position()` / `torso()` + `_act()` | Game positions (step 7) |
| `hand_x/y` | `interaction/game.py`, `punching_hand()` | Bolts now start at the punching hand (Task 4 modifications) |
| `head_x/y` | Not ported | `renderer.py`'s health bar above the head is pinned to the bottom now (Task 5 modifications); face effects use face anchors |
| `lower_body_y()`, `detect_hands_up()`, `detect_crossed()`, `arm_extended()`, `arm_bent()`, `detect_punch()` | `interaction/gestures.py`, same names | Task 4; numbers from config, keypoints by name |

**Constants** (`configs/default.yaml`)

| In your code | Now |
| --- | --- |
| `CONF_T` (keypoints) | `motion.kpt_conf_threshold` |
| `ALPHA` | `motion.tau` (seconds) |
| `MISSING_FRAMES` | `motion.hold_s` (seconds) |
| `CROSS_MAX_WRIST_DIST`, `PUNCH_ELBOW_ANGLE`, `PUNCH_MIN_REACH`, `PUNCH_BENT_ANGLE` | `gestures.cross_max_wrist_dist`, `punch_elbow_angle`, `punch_min_reach`, `punch_bent_angle` |
| Numbers inside the functions (0.3, 2.0, 0.5, 1.0 sw) | `gestures.head_above_shoulders`, `hip_below_shoulders`, `chest_above_shoulders`, `punch_above_shoulders` |
| `NOSE`, `L_SHOULDER`, …, `SKELETON` | `core/skeleton.py` (names and `EDGES`) |
| `CONFIRM_FRAMES`, `PUNCH_HOLD_S`, `sw > 10` | `gestures.confirm_s`, `punch_hold_s`, `min_shoulder_px` |
| `CONF_T` (detection), `IMGSZ`, `MODEL_PATH` | `pose.det_conf`, `pose.imgsz`, `pose.weights` (already there; the pipeline runs the pose model once on the full frame) |

#### Task 4: `game.py` (game logic)

**Functions and classes**

| In your code | Now in | Note |
| --- | --- | --- |
| `PaintSplat`, `Projectile`, `PaintBeam`, `Shield` | `interaction/game.py`, same names | Pixels; a splat stores `owner_id` instead of a colour |
| `Player` | `interaction/game.py`, `GamePlayer` | Renamed (`Player` is the identity type) |
| `Game.__init__()`, `get_player()`, `reset()` | `interaction/game.py`, same names | `reset(pid)` added |
| `attack()`, `beam_attack()`, `activate_shield()`, `_find_projectile_target()`, `_check_beam_hits()`, `_apply_damage()` | `interaction/game.py`, same names | Pixels; shield check added |
| `Game.update(dt, visible)` | `interaction/game.py`, `Game.advance(dt, visible)` | Renamed; `update(state)` is the module interface |
| (new) shield check | `interaction/game.py`, `Game.shielded(pid)` | |
| `main.py`: game loop part (bounds, positions, gesture edges → `attack()` / `beam_attack()` / `activate_shield()`, `dt`, `game.update()`) | `interaction/game.py`, `Game.update(state)` + `_act()` | Events instead of per-frame flags; `dt` from `Frame.t` |
| `bodypose.py`: `center_x`, `torso_y`, `hands_center_x/y` | `interaction/game.py`, `Game.body_position()` + `_act()` | Pixels, from the smoothed keypoints |
| `test_game.py` | `tests/test_game.py` | Same tests in px, plus new ones |
| (new) debug view, CSV columns | `interaction/game.py`, `Game.draw_debug()` + `log_fields()` | `--until game`, plain shapes; the game's look is drawn by `scene/renderer.py` |
| (new) magic ball and beam | `scene/magic_beam.py`, `MagicBeam`, drawn by `scene/renderer.py`; `scene/draw.py`, `blend_rgba()`, `main_hue()`, `shift_hue()` | Task 4 modifications; one hue-turned copy per team |
| (new) fire blast for the bolt | `scene/fire_bolt.py`, `FireBolt`, drawn by `scene/renderer.py`; `Projectile.age`, `BoltEnd` | Task 4 modifications; one hue-turned copy per team |
| (new) ring of fire for the shield | `scene/fire_ring.py`, `FireRing`, drawn by `scene/renderer.py` (`ring_geometry()`) | Task 4 modifications; one hue-turned copy per team |
| `Game.player`, `Game.shield` | Removed | Single-player shortcuts, unused |
| `PLAYER_COLORS` | `scene/draw.py`, `PLAYER_COLORS` / `player_color()` | The one colour table |

**Constants:** all in `configs/default.yaml`, `game:` section; see step 5 for the table. The sprite and health-bar keys added in the Task 4 modifications moved to `render:` in the renderer merge (step 1).

#### Task 3: `Project/` (identity)

**Functions and classes**

| In your code | Now in | Note |
| --- | --- | --- |
| `osnet.py` | `identity/osnet.py` | Unchanged (pre-merge copy of torchreid's file) |
| `OSNetEmbeddingExtractor` (`osnet_embedding.py`) | `identity/embedding.py`, same name | Weights path and device from config |
| `PlayerIdentityTracker` | `identity/tracker.py`, `IdentityTracker` (`backend: reid`) | Same attribute names: `embeddings` (your `players`), `track_to_player`, `pending_tracks` |
| `PlayerIdentityTracker.update()` | `IdentityTracker._identify()` | Seconds; faces only; slow retry when slots are full |
| `_find_player()` | `IdentityTracker._find_player()` | |
| `forget_missing_tracks()` | `IdentityTracker.forget_missing_tracks()` | Drops a link only when BoT-SORT drops the track |
| `main.py`: `person_detector.track(..., botsort.yaml)` with `yolo11n.pt` | `identity/tracks.py`, `BotSortTracks` | On the pose model's boxes; `yolo11n.pt` removed |
| `main.py`: loop over tracks, crop size check, `excluded_player_ids` | `IdentityTracker.update(dets, frame)`, `_crop()` | Returns `Player`s with status and `reentered` |
| `main.py`: `pose_processor.reset_player()` for lost players | `Player.reentered` → pipeline resets face, motion, gestures | |
| "Identifying..." label | `IdentityTracker.draw_debug()` | `--until identity` |
| `test_osnet.py` | `tests/test_identity.py` | No webcam; plus tests of the tracker |
| (new) link check and vector update | `IdentityTracker._check_and_update()`, `_relink()` | |
| (new) CSV columns | `IdentityTracker.log_fields()` | |
| `identitytracker.py` | Not merged | Only started `bodypose.py`'s demo |
| `models/osnet_x1_0_msmt17.pth` | `models/osnet_x1_0_msmt17_combineall.pth` | Same network and data, from Hugging Face (README) |

**Constants:** `REID_THRESHOLD`, `NEW_PLAYER_CONFIRMATION`, `MAX_PLAYERS`, `MIN_PERSON_HEIGHT`, `MIN_PERSON_WIDTH` and `botsort.yaml` are in `configs/default.yaml`, `identity:` section; see steps 2 and 3 for the tables.

#### Renderer: `renderer.py` (game view)

| In your code | Now in | Note |
| --- | --- | --- |
| `Renderer.render(camera_frame, game, pose)` | `scene/renderer.py`, `Renderer.draw(state, game)` | Pixels; positions from the frame state |
| Pixelated camera image (half size, nearest pixel) | `scene/renderer.py`, `pixelate()` | `render.pixel_size` |
| `_draw_player_paint()` | `scene/renderer.py`, `_draw_player_paint()` + `paint_area()` | On the torso, not the body box |
| `_draw_player_health()` | `scene/renderer.py`, `_draw_player_health()` | Pinned to the bottom: P1 left, P2 right (Task 5 modifications) |
| Top bar, controls, "STEP INTO VIEW", "Q: QUIT" | `scene/renderer.py`, `_draw_hud()` | Our three gestures; text fitted to the frame |
| `_draw_game_over()` | `scene/renderer.py`, `_draw_game_over()` | Dimmed frame; objects in flight hidden |
| Bolt rectangles, `_draw_paint_beam()`, `_draw_shield()`, `_draw_shield_splash()`, palette `.jpg` | Not merged | Replaced by the fire blast, magic beam and fire ring (Task 4 modifications), now drawn by `Renderer.draw_game()` |
| `main.py`: key `r` after a win → `game.reset()` | `app/main.py` key `r` → `Pipeline.new_round()` | Any time; `R` = full reset |

**Constants:** sizes and switches in `configs/default.yaml`, `render:` section; see steps 2–4 of the renderer merge.

### Modifications: changes beyond moving code

#### Task 1

- **Head crops instead of full-frame detection.** YuNet runs on a crop around each player's head, taken from the pose keypoints, and the player ID comes from Task 3. `update_tracks()`, greedy matching and `MAX_PLAYERS` are gone, so faces cannot swap between players. It is also faster: 1.2 ms per small crop against 24.5 ms for a full 720p frame.
- **Which detection is used (new).** If a crop holds more than one face, the one closest to the player's head keypoints is used.
- **Seconds instead of frames.** Hold, flow and lost limits are in seconds (`Frame.t`), so they behave the same at any frame rate.
- **Fade-out moved into the contract.** The opacity is now `Face.alpha`, a new `core/types.py` field, so the renderer and the game can see it.
- **More anchors.** `forehead` and `above_head` are added next to `hat` and `mouth`.
- **Head pose and 3D anchors (new).** `solvePnP` fits a canonical 3D head to the 5 landmarks. The forehead, hat and `above_head` anchors are then 3D points on that head, projected into the image. The script's fixed 2D offsets made the hat rise when the chin went up, and sink when it went down. `Face.yaw` is filled from the pose.
- **Head rotation from MediaPipe (new).** YuNet's 5 landmarks stay near-frontal at strong up/down tilts (−3° measured on a 33° chin-up), so the rotation now comes from MediaPipe Face Landmarker, which runs on one player per frame, in turn. `solvePnP` on YuNet's landmarks plus the pose model's ears is the fallback.
- **Mouth and blinks (new).** MediaPipe's blendshape scores go into `Face.blendshapes`, a new `core/types.py` field.
- **Hat hangs from its brim.** The script's `overlay()` pinned the hat at the bottom of the PNG, which has 34% transparent padding below the hat, so the visible brim floated half a face width above the anchor. The hat now hangs from its visible brim, and `hat_drop` (0.40 below the forehead point) became `hat_rise` (0.11 above it), which keeps the frontal look the same.
- **Smoothing order.** The One Euro filter now runs on the face sides and the anchors together, after the 3D projection, instead of on the 4 key points before `geometry()`.
- **Flow also moves the 5 landmarks (new),** not just the key points and flow points.
- **Large head crops are shrunk (new).** Crops over `crop_max_px` (192) are shrunk before YuNet, and the detection is scaled back to frame pixels: 23.4 → 9.2 ms on a 4K clip with nearly the same detections.
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
- **Punch reach is sideways (step 2b):** `arm_extended()` measures the wrist's horizontal distance from the shoulder, not the straight-line distance, so a straight hanging arm is no longer a punch.
- **Punch wrist at least at shoulder level (Task 4 modifications):** the wrist may be at most `punch_below_shoulder` (0.2 sw) below its own shoulder; `bodypose.py` allowed it down to the hips.
- **Events instead of per-frame flags (step 3):** one `GestureEvent` when a gesture turns on; `main.py`'s `hands_were_up` / `arms_were_crossed` bookkeeping moved into the gesture stage. Punches less than 0.5 s apart give one event.
- **No usable shoulder width restarts the timers (step 3);** `bodypose.py` kept counting across such frames.

#### Task 4: game logic

- **One colour table (step 5):** `game.py`'s P1 red and P2 blue replace the debug colours in `scene/draw.py`; splats store the attacker's pid.
- **Pixels instead of 0–1 coordinates (step 5);** speeds and sizes are fractions of the frame width or height in the config.
- **No bolt cooldown (step 5):** `bolt_cooldown_s` is 0, so every punch fires a bolt.
- **One action at a time (Task 4 modifications, new):** after any action the player must return to a normal position (no gesture on) before the next one; gestures starting together send only the highest priority (hands up, arms crossed, punch); no bolt while the player's own beam is alive.
- **One beam per raise (Task 4 modifications):** hands up and arms crossed stay on through dropouts shorter than `release_s` (0.3 s), and a player cannot start a beam while their previous one is alive; `game.py` had no limit.
- **Attacks fly towards the other player (Task 4 modifications):** sideways towards the nearest other player in view, not towards the middle of the frame as in `main.py`, so switching sides no longer sends attacks the wrong way; the middle rule is the fallback.
- **Slower bolt (Task 4 modifications):** `bolt_speed_w` 1.0 frame widths/s instead of `game.py`'s 1.4, so the bolt is easier to see coming and shield.
- **Bolts start at the punching hand (Task 4 modifications)** instead of the body box edge at hip height; the box edge is the fallback without a visible wrist.
- **The shield is drawn as a ring of fire (Task 4 modifications, new):** sprite frames per team instead of `renderer.py`'s shield picture, on the player's chest and sized to their shoulders.
- **The bolt is drawn as a fire blast (Task 4 modifications, new):** sprite frames per team instead of `renderer.py`'s bolt, with a break-up where it ends; `Projectile` has an `age` field, and `Game.bolt_ends` keeps where bolts stopped, for drawing.
- **The beam is drawn as a magic ball and beam (Task 4 modifications, new):** sprite frames instead of `renderer.py`'s paint beam, in each player's team colour.
- **The beam lasts its full duration (Task 4 modifications):** its tip stops at the frame edge instead of the beam ending there, so a target stepping into it later is also hit (once).
- **The shield blocks (step 6, new):** while it is up (not during the fade), bolts are absorbed and beams pass that player without damage; in `game.py` the shield was only drawn.
- **Drawing moved to the renderer (renderer merge):** the fire blast, magic ball and beam, ring of fire and health bar above are drawn by `scene/renderer.py`, with their config keys in `render:`; `game.py`'s rules are all that is left in `interaction/game.py`.

#### Task 3

- **`torchreid` replaced by a copied `osnet.py` (pre-merge).** `osnet_embedding.py` imports it instead of `torchreid`. Embeddings are identical (max difference 0.0).
- **`load_weights()` raises an error (pre-merge)** when no layer of the weight file matches, instead of only warning and running with random weights.
- **One model per frame (step 3).** BoT-SORT runs on the pose model's boxes; `yolo11n.pt` is gone. A new track needs score 0.5 (your detector's limit); weaker boxes from 0.35 still continue a track. Camera-motion compensation is off (4.4 → 0.4 ms).
- **Weights from Hugging Face.** Same network and training data as your file, with a public download and checksum.
- **Occluded, then lost (step 2).** A player keeps their link while BoT-SORT still holds the track (`"occluded"`, predicted box); only when the track is dropped is the player `"lost"`, and `reentered` is set when they come back. Your code dropped the link and reset the pose state on every frame without the track.
- **Two players** instead of 3; a new player takes the lowest free id.
- **Seconds instead of frames:** the 5-frame confirmation is `new_player_s` (0.13 s).
- **Linked tracks first (step 2):** a new track cannot take a player whose own track is visible.
- **Slow retry (step 2):** a track that finds no player while both slots are taken tries again every 0.5 s, not every frame.
- **Stored vectors are updated (step 2, new):** one player per frame, in turn, every 0.5 s, when their box overlaps no one, by 0.1 towards the fresh vector if it is still similar (≥ 0.5).
- **Linked tracks are checked (step 5, new):** the same OSNet run moves or swaps a link when the crop looks more like the other player. On a copy of the session clip with the frame halves swapped, the players were wrong for 6 of 120 frames instead of 60.
- **Faces only (step 5, new):** a track is identified only when the pose model sees a nose or an eye (`min_face_conf` 0.5), so a coat never becomes a player.
- **Debug view and CSV columns (step 4):** track id and similarity per player; OSNet runs per frame; OSNet warm-up at start-up.

#### Renderer

- **The renderer draws the game (step 1):** the sprites (fire blast, magic ball and beam, ring of fire) moved from `Game` to `scene/renderer.py`; the game logic has no drawing left except its debug view (plain shapes).
- **Paint on the torso (step 4, new):** splats are drawn between the shoulders and from the shoulder line to the hips, so they stay on the body when an arm goes out; `renderer.py` drew them in the body box, which widens with the arms.
- **Health bars pinned to the bottom (Task 5 modifications):** P1 bottom left, P2 bottom right, like in a fighting game, instead of above the head (`renderer.py`) or below the body (Task 4 modifications, then smoothed in step 4).
- **No frozen end (step 3, new):** after a win, the frame is dimmed behind the panel, and bolts and beams in flight are not drawn (the game stops then, as in `game.py`).
- **Key `r` any time (step 3):** a new round with the same players; `R` resets everything, including identity.
- **Face effects** are drawn on top of the game objects (new; `renderer.py` had none).
- **Sizes scale with the frame** instead of fixed pixels at 640 × 480; text shrinks to fit a narrow frame (step 5).

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

- `Face` has a new last field, `alpha: float = 1.0`: the effect's opacity, 1 while the face is tracked, fading to 0 once it is lost. It keeps the script's fade-out.
- It has a default, so existing code that builds a `Face` without it keeps working.

**Changed:** `configs/default.yaml`, `face:` section

- `backend`: `yunet` (real detector) or `stub` (the old nose-keypoint anchor, no model file needed; used by the tests).
- `model`, `score_threshold` (0.6, the script's `SCORE_THRESH`), `nms_threshold` (0.3) and `top_k` (50) for YuNet.
- The stub keys `above_head_factor` and `default_scale` stay, marked "stub backend only".

**Changed:** `README.md`: download command and SHA-256 for `models/face_detection_yunet_2023mar.onnx`. The script downloaded the model at runtime; the pipeline doesn't download it, and the model stays out of git (`models/` is ignored).

#### Step 2 · Effects: `load_sprite` and `paste`

**Changed:** `src/ipcvgame/face/effects.py` (replaces the stub's dot)

- `load_sprite(path)`: loads an RGBA PNG (a relative path is taken from the repo root) and fails with a clear error if it is missing or has no transparency.
- `paste(img, sprite, at, pivot, angle, width, alpha)`: the version from `docs/phase-1/facetracking-update-guide.md` (step 1), plus the script's `alpha` fade. It replaces the script's `overlay()`. The geometry is the same: the pivot lands on the anchor, the sprite is rotated by the head roll and scaled to a multiple of the face width. The difference is that only the sprite's box is warped and blended, and a sprite partly off-screen is clipped instead of crashing.
- `FaceEffects.draw`: hat at `anchors["hat"]` (pivot bottom-middle), mustache at `anchors["mouth"]` (pivot centre), both faded by `Face.alpha`. Faces with `valid=False` or `alpha=0` are skipped. The stub tracker has neither anchor, so until step 4 nothing is drawn on faces in the game view.
- The script's `draw_status()` (name + HP bar) is not ported: it is game UI.

**Added:** `assets/face/hat.png`, `assets/face/mustache.png` (copied from `face_track/`). The copies in the repo root stay until step 7, so `hat_tracker_v2.py` keeps working.

**Changed:** `configs/default.yaml`, `face:`: `hat_sprite`, `mustache_sprite`, `hat_width` (1.5) and `mustache_width` (0.8), from the script's `overlay()` calls.

**Added:** `tests/test_effects.py`: 7 tests covering pivot placement, scaling, rotation, alpha, off-screen clipping, sprite errors, and skipping hidden faces.

#### Step 3 · Geometry: `measure()` and `geometry()`

**Changed:** `src/ipcvgame/face/tracker.py`: two module-level functions above the (still stub) `FaceTracker`; the class itself is unchanged until step 4.

- `measure(row, center_drop, forehead_rise, lip_to_nose)`: the script's `measure()`. It turns one YuNet detection into the 4 key points (left side, right side, forehead, upper lip), the face centre at cheek level, and the 5 raw landmarks (which become `Face.landmarks`). Same maths; the constants are passed in from config instead of hard-coded.
- `geometry(pts, hat_drop, above_head_rise)`: the script's `geometry()`. It gives face width (`Face.scale`), head roll (`Face.roll`) and the anchors effects draw at, offset along the face's own "down" so they follow a tilted head. The script returned only the hat and lip points; it now returns all of `forehead`, `hat`, `mouth` and `above_head`.
- New anchor `above_head`: above the hat, where the script drew the player name. The game UI (HP bar, name) should draw there.
- Not ported: the flow-point grid (`lk`) the script built in `measure()`. It belongs to the optical-flow fallback and comes in step 5.

**Changed:** `configs/default.yaml`, `face:`: `center_drop` (0.20), `forehead_rise` (0.50), `lip_to_nose` (0.35) and `hat_drop` (0.40), all from the script, plus `above_head_rise` (1.5, the hat's height in face widths).

**Added:** `tests/test_face.py`, 4 tests on synthetic YuNet rows:

- an upright face gives roll 0, the right width, and anchors in top-to-bottom order;
- a face tilted by +10° and −25° gives that roll, the same width, and anchors rotated with the head;
- swapping the two eyes in the detection changes nothing.

#### Step 4 · Detection on per-player head crops

**Changed:** `src/ipcvgame/face/tracker.py`: `FaceTracker` now runs YuNet; the old stub is kept as backend `stub`.

- `head_box(player, kpt_conf_threshold, crop_size)`: a square crop around each player's head from that player's pose keypoints. The centre is the mean of the visible nose, eye and ear keypoints, or else the top of the person box. The side is `crop_size` shoulder widths, or person-box widths when a shoulder is not visible.
- `_detect()`: YuNet on the crop. Detections are shifted back to full-frame pixels, and then filtered in this order:
  - narrower than `min_face` (the script's `MIN_FACE`): dropped;
  - width changed by more than `max_size_jump` since the last frame (the script's `MAX_SIZE_JUMP`): dropped;
  - of the rest, the one closest to the crop centre (the head keypoints) is used, so another player's face that falls into the same crop is skipped.
- `update()`: per player, `measure()`, then the One Euro filter on the 4 key points (`core.filters.OneEuro`, same maths and values as the script's own filter), then `geometry()`, then `Face`:
  - `landmarks` = the 5 raw YuNet points;
  - `center`, `scale` (face width) and `roll` from the smoothed key points;
  - `yaw = 0.0`, since 5 landmarks give no yaw;
  - `anchors` = `forehead`, `hat`, `mouth`, `above_head`;
  - `alpha = 1.0`.
- Per-player state (`_FaceState`: filter, last width, last detection time) is kept between frames. It is dropped by `reset(pid)` (called when a player re-enters), when the player disappears from `players`, or after `lost_after_s` without a detection (the script's `MAX_MISSED`, now in seconds). A returning face then starts fresh instead of sliding in.
- `draw_debug`: per player, the head crop (dashed) labelled with the YuNet score or "no face", plus the 5 raw landmarks, face centre and scale circle, and every anchor with its name.
- Replaced from the script: full-frame detection, de-duplication, `update_tracks()` (greedy matching), `Track.predicted()`, the velocity and `MAX_PLAYERS`. Identity now comes from Task 3 through the head crop.
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

#### Step 5 · Optical-flow fallback, hold time and fade-out

**Changed:** `src/ipcvgame/face/tracker.py`

- `flow_points(pts, cfg)`: the script's grid of points on the face (6×6, eyes to mouth), which `measure()` used to build. It is now built from the 4 key points.
- `estimate_motion(prev_gray, gray, pts, cfg)`: the core of the script's `Track.flow()`. Lucas-Kanade optical flow on the grid, then a similarity transform (shift + rotation + scale) fitted with RANSAC. It returns `None`, so the face is not moved, in any of these cases:
  - fewer than `flow_min_points` points tracked;
  - fewer than `flow_min_inliers` inliers;
  - a scale outside `flow_scale_range`.
- `fade(blind_s, hold_s, lost_after_s)`: the script's `Track.alpha()` in seconds. Fully visible for `hold_s` after the last update, then a linear fade to 0 at `lost_after_s`.
- `FaceTracker.update()`, per player, each frame:
  - **detect:** YuNet found the face. The key points, landmarks and flow points are reset from the detection.
  - **flow:** no detection, but the last detection is under `flow_max_s` old and the fit is good. The key points, flow points and landmarks are moved by the fitted transform. The script moved only the key points and flow points.
  - **hold:** neither. The last smoothed position is shown and fades out.
- After a detect or flow update, the key points go through the One Euro filter and `geometry()`, as in the script's `_apply()`.
- `Face.alpha` comes from `fade()`; `Face.valid` is `alpha > 0`. A held face is still valid, so the game can keep drawing at its anchors while it fades.
- State is now dropped after `lost_after_s` without any update (detection or flow), not just without a detection. The size-jump check uses the current width, which flow also updates, as the script did.
- The frame is converted to grayscale once per frame (the raw frame, before effects are drawn) and kept for the next frame's flow.
- `draw_debug`: the crop label shows the source (`detect 0.91`, `flow`, `hold`) and `a=` alpha. The flow points are drawn while flowing.
- Dropped from the script: `Track.vel` and `predicted()`, only used for matching (see step 4).

**Changed:** `configs/default.yaml`, `face:`

- `hold_s` (0.1 s), from `HOLD_FRAMES` = 3 frames.
- `flow_max_s` (1.3 s), from `FLOW_FRAMES` = 40 frames.
- The flow grid (`flow_grid_x`, `flow_grid_y`, `flow_grid_n`) and the Lucas-Kanade settings (`flow_win` 31, `flow_levels` 4, `flow_criteria` [20, 0.01]), as in the script.
- RANSAC limits: `flow_min_points` 8, `flow_min_inliers` 6, `flow_ransac_px` 3.0, `flow_scale_range` [0.7, 1.4], as in the script.
- `lost_after_s` comment updated.

**Added to** `tests/test_face.py`: 6 tests.

- `fade()` holds, fades and reaches 0.
- The flow grid covers the face and turns with it.
- `estimate_motion()` recovers a known shift on a textured image, and rejects a zoom outside the allowed range.
- Through `FaceTracker` with a scripted detector (no model file): flow follows a shifted face when detection misses; then hold (anchors frozen, alpha 1), fade (alpha 0.5 halfway) and gone after `lost_after_s`.

#### Step 6 · Evaluation: `log_fields` and `tools/eval_face.py`

The script's `Metrics` class (jitter, alignment error, detection rate per lighting, latency) moves out of the tracker. The tracker now writes per-frame CSV columns, and a separate script computes the metrics from them.

**Changed:** `src/ipcvgame/face/tracker.py`

- `log_fields(state)`: per player, the columns `face_p<pid>_source` (detect/flow/hold, empty while no face), `_score` (YuNet score on detect frames), `_alpha`, `_scale`, `_roll_deg`, `_mouth_x/_y` (smoothed anchor), `_mouth_raw_x/_y` (unsmoothed upper-lip point, on detect frames only) and `_crop_px`. The script logged the same raw-versus-smoothed lip pair.
- The raw detection and crop box are now recorded in `update()` instead of `_detect()`, so they are also there when the detector is replaced in tests.

**Added:** `tools/eval_face.py`

- Takes clips, which it runs headless up to the face stage, writing `logs/eval/face-<clip>[-<tag>].csv`; or it takes existing logs.
- Prints one Markdown row per clip and player:
  - coverage: detect / flow / hold / no face %;
  - jitter: the script's second-difference RMS, raw versus smoothed, in px and % of face width;
  - lag: the script's alignment error (mean, p95, % face width);
  - crop size and face-stage time.
- `--set` and `--tag` run parameter experiments; `--markdown` saves the table.
- Differences from the script: conditions come from clip names instead of `s`/`d`/`l` key presses. Latency is the face-stage time from the profiler (`ms_face`). Identity switches are no longer counted here: faces cannot swap with head crops, and switches are Task 3's metric.

**Added:** `docs/evaluation.md`: how to evaluate on the clips for the report. The workflow and run-log columns are for all tasks; the face section has commands, column definitions, how to read the numbers, and an example. It ends with how to add an `eval_<stage>.py`.

**Added:** `tests/test_eval_face.py` (3 tests: the metrics on a synthetic log with known jitter, lag and coverage; missing frames count as "no face"; the Markdown table). Also added to `tests/test_face.py`: a test that raw mouth columns are only written on detect frames.

#### Step 7 · Cleanup

**Deleted** (superseded by `src/ipcvgame/face/`; still in git history):

- `hat_tracker.py`, `hat_tracker_v2.py`: the earlier standalone MediaPipe versions of the face script.
- `hat.png`, `mustache.png` in the repo root: byte-identical to `assets/face/`.

**Changed:**

- `docs/phase-1/facetracking-update-guide.md`: a "History" note at the top. It says the guide was for the standalone script, where its `paste()` now lives, and links here.
- `docs/phase-1/status.md`:
  - the module table now shows the face stage as done;
  - Task 1's "Stub today" became a status paragraph and a checklist: done items ticked, the next steps listed;
  - the old cross-section read note is replaced (the face stage has its own `kpt_conf_threshold`);
  - a face row in the timing table;
  - the anchor-name decision ticked (`forehead`, `mouth`, `hat`);
  - the stub-limitation line about the face dot removed;
  - test count updated to 44.
- `README.md`: `assets/` in the layout; YuNet (MIT, OpenCV model zoo) in the licence section.

**Not changed:** `.gitignore` keeps `garb/` until the whole Colour Clash merge is done, not just Task 1.

## Task 1: Modifications

Changes beyond the Task 1 script, decided by the Task 5 owner after step 6 (2026-10-25). They only touch `face/`, apart from one new `Face` field (`blendshapes`; team heads-up, approved):

- **Speed:** shrink each head crop to a maximum size before YuNet; the 4K test clip cost 24 ms in the face stage.
- **Head pose:** fills `Face.yaw` and places the forehead, hat and `above_head` anchors in 3D.
- **Mouth opening and blinking** with MediaPipe Face Landmarker, for game actions:
  - eating consumables with the mouth;
  - a blink resets the super-charge move;
  - taunt (smile) and cheek bulge effects.

### Shrink large head crops before detection

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

### Head pose and 3D anchors

**Why the scope grew.** The plan was to fill `Face.yaw`. Testing the shrunk crops on the webcam showed the hat lifting off the head when the chin goes up. The cause is the script's 2D geometry: the forehead and hat sit at fixed offsets above the eyes, whatever way the head points. When the chin goes up, the top of the head turns away from the camera and moves *down* in the image, but the hat stays put. When the chin goes down, the hat sinks onto the forehead. So the full head pose is estimated, and those anchors are placed in 3D.

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
- The crop-shrinking test was updated for the new filter argument.

**Open:** check on the webcam (chin up, chin down, turning). `hat_depth_mm` is a first guess; a larger value moves the hat more with tilt. The mouth anchor is not affected by the pose.

### MediaPipe: head rotation, mouth opening and blinks

**Why.** Testing the head pose on the webcam, the hat still lifted off during a strong chin-up. On that frame, YuNet's 5 landmarks were almost frontal: nose 0.49 eye distances below the eyes (frontal 0.54), mouth 0.99 (frontal 0.99). `solvePnP` therefore found −3° of pitch. MediaPipe Face Landmarker, which fits 478 landmarks, found 33°, which matches the picture. Adding the pose model's ears to `solvePnP` gave 15–21°: better, but not enough. Mouth opening and blinking need MediaPipe anyway, so its head rotation is used as well.

**Contract (team heads-up):** `src/ipcvgame/core/types.py`: `Face.blendshapes: dict[str, float]`, default empty. These are MediaPipe's 52 scores from 0 to 1 (`jawOpen`, `eyeBlinkLeft`, `eyeBlinkRight`, `mouthSmileLeft`, `cheekPuff`, …), empty when not measured. MediaPipe names the left/right scores as if the image were not mirrored, so on our mirrored frame `eyeBlinkLeft` is the player's own right eye.

**Added:** `src/ipcvgame/face/landmarker.py`, `FaceLandmarker`. A thin wrapper around MediaPipe Face Landmarker in IMAGE mode: every call stands alone, because the crops come from different players in turn. It returns the blendshape scores and the head transformation matrix for one face crop, or `None`.

**Added:** `src/ipcvgame/face/head_pose.py`. The head-pose functions moved here from `tracker.py`, to keep that file readable. New in this change:

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
  - The 3D anchors are then projected as with the YuNet pose.
  - Both pose sources go through the same steps, so switching between them does not make the hat jump.
- `geometry(pts, hat_rise, above_head_rise)`: the hat brim is now `hat_rise` above the forehead point (see the effects change below).
- `log_fields`:
  - `face_p<pid>_pose` is now the source: `mediapipe`, `pnp` or empty for 2D;
  - new `_jaw_open`, `_blink` (mean of both eyes), `_smile` (mean of both sides);
  - new `_mp` (1 = MediaPipe result this frame).
- `draw_debug`: the pose source in the label, and `jaw` and `blink` bars next to the face.

**Changed:** `src/ipcvgame/face/effects.py`. The hat now hangs from its visible brim: the lowest row of `hat.png` that is not transparent. `hat.png` has 34% transparent padding below the hat, so with the script's pivot at the bottom of the image, the visible brim floated half a face width above the anchor. Any pitch error made that look worse.

**Changed:** `configs/default.yaml`, `face:`

- `hat_drop: 0.40` became `hat_rise: 0.11`: the brim sits 0.11 face widths above the forehead point. That is −0.40 + 0.51 (34% padding × 1.5 face widths), so the hat looks the same as before when facing the camera.
- `above_head_rise: 1.5` became `1.0`, now measured from the brim; `above_head` stays 1.6 face widths above the eyes.
- `head_model_mm` gets `left_ear` and `right_ear` at (±72, 20, 85) mm: beside the head, at ear-canal height, behind the eyes.
- New: `landmarker` (true), `landmarker_model`, `landmarker_min_conf` (0.5), `landmarker_crop` (2.0), `landmarker_max_side` (256), `landmarker_per_frame` (1), `landmarker_max_age_s` (0.25).

**Changed:** `README.md`: download command and SHA-256 for `models/face_landmarker.task`. `tools/eval_face.py`: `pose %` now counts both pose sources, plus a new `mediapipe pose %`; `docs/evaluation.md` explains them.

**Tests:**

- `tests/test_face.py`:
  - MediaPipe's matrix gives our yaw and pitch for 5 poses, including the 33° chin-up;
  - `rotation()` inverts `pose_angles()`;
  - `translation()` recovers a known position;
  - with frontal-biased landmarks, the 5-point fit misses a 33° chin-up and the ears recover it;
  - with a stand-in landmarker and two players, MediaPipe runs once per frame, in turn (P1, P2, P1, P2), and its blendshapes and pitch reach `Face` and the pose.
- The frontal test now checks the exact perspective offset of the deeper hat points (`Z / (Z + depth)`) instead of a tolerance.
- `tests/test_effects.py`: the hat hangs from its brim on a padded sprite.
- Existing tests: imports moved to `head_pose.py`, `hat_rise`, and `landmarker: False` where YuNet is replaced.

**Open:**

- Check on the webcam: chin up and down, blinking, mouth opening (the `jaw` and `blink` bars in the debug view).
- MediaPipe's GPU delegate may cut its ~8 ms; not tried yet.
- With `landmarker_per_frame: 1`, two players are measured at ~15 Hz each. Set it to 2 if short blinks are missed.

### Fix: crash in `solvePnP` on degenerate points

**What happened.** The live game crashed with `cv2.error ... sqpnp.cpp:236: (-215:Assertion failed) point_coordinate_variance >= POINT_VARIANCE_THRESHOLD`, raised from the `solvePnP` fallback in `head_pose()`.

**Cause.** OpenCV's SQPnP solver asserts, instead of failing softly, when the image points (nearly) coincide or are not finite. Reproduced offline: it fails for a face a few pixels wide or a NaN coordinate, and works for normal and off-image points. YuNet never reports a face under `min_face` (20 px), so the likely route in a live run is optical flow: each step may shrink the face by up to `flow_scale_range[0]` (0.7×), and over `flow_max_s` that can compound to a few pixels. The exact frame of the crash was not recorded.

**Changed:**

- `src/ipcvgame/face/head_pose.py`, `head_pose()`: returns `None` (2D anchors for that frame) for non-finite points or eyes under 2 px apart, and for any `cv2.error` from the solver. A pose failure can no longer stop the game.
- `src/ipcvgame/face/tracker.py`, flow branch of `update()`: a flow step that would shrink the face below `min_face` is not applied. The face holds and fades instead, the same as when flow fails.

**Added to** `tests/test_face.py`:

- `head_pose()` returns `None` for NaN and for collapsed points (the crash case);
- flow that shrinks the face by 25% per frame stops (→ hold) before the face drops below `min_face`.

### Backlog: `docs/extra_features.md`

**Added:** `docs/extra_features.md`. The game features agreed during the merge, written up for after it, each with owners, inputs from the face stage, a starting design with thresholds, and how to check it:

1. team colour;
2. hit reaction;
3. status above the head;
4. super charge that a blink resets;
5. taunt;
6. Smash-Bros-style consumables (eat falling items with a chomp, or grab them from the floor);
7. cheek bulge.

It also lists the merge steps still needed first (gestures, game logic, renderer, identity), the technical follow-ups from Task 1, and ideas considered but not planned. No code changed.

## Task 4: Gestures and game logic

### Plan

#### Gestures

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
- **Punch reach measured sideways (after step 2).** Step 2 showed that the punch rule from `bodypose.py` fires on a straight hanging arm; step 2b fixes it.

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
2b. Punch rule fix: sideways reach in `arm_extended()`.
3. `GestureRecognizer`: timers in seconds, rising-edge events `punch`, `hands_up`, `arms_crossed`, reset on loss and re-entry, `draw_debug`, `log_fields`.
4. Check and record: event counts on the two test clips, Change Overviews, `docs/phase-1/status.md`.

**Verification.** Unit tests for every step; event counts on `data/test_clips/`. The Task 5 owner checks the gestures live on the webcam.



#### Game logic

**Source.** `game.py` with its tests (`test_game.py`), and the part of `main.py` that turns gestures into game actions.

**What the code does.**

- **Coordinates:** 0–1 fractions of the camera frame.
- **Per player (`Player`):**
  - health (100) and a bolt cooldown;
  - a position: `x` is the shoulder midpoint and `y` the hip level, from `bodypose.py`'s `center_x` and `torso_y`;
  - a body box (`bounds`, the BoT-SORT person box);
  - paint splats.
- **Actions**, called by `main.py` when a gesture turns on. Bolts and beams go right for a player in the left half of the screen, left otherwise.
  - **punch → `attack()`:** a bolt from the edge of the body box facing the other player, at hip height. It moves at 1.4 frame widths/s and does 10 damage; then a 1.8 s cooldown.
  - **hands up → `beam_attack()`:** a beam from the midpoint of the hands. It charges for 0.35 s, then grows at 2.4 frame widths/s for 0.8 s. It does 12 damage, hits each player once, and hits a box within ±0.03 frame heights of the beam's line.
  - **arms crossed → `activate_shield()`:** a shield beside the player for 1 s, fading out over 0.24 s. It blocks nothing.
- **`update(dt, visible)`:**
  - cooldowns and shields count down, and bolts and beams move;
  - a bolt hits the first other visible player whose box it crosses since the last frame, so a fast bolt cannot skip a box;
  - a hit lowers the target's health and adds 4 paint splats: one at the impact point and 3 random ones near it. They are seeded, so the same hit always gives the same splats, and kept in the middle of the box (0.38–0.62);
  - health 0: the attacker wins and the game freezes.

  `dt` comes from `time.perf_counter()`, capped at 0.1 s.
- **`reset()`:** a new round, keeping positions; `main.py` calls it with key `r` after a win.
- **What the renderer reads:** `players`, `projectiles`, `beams`, `shields`, `winner_id`, `MAX_HEALTH`, `BEAM_CHARGE_DURATION`, `SHIELD_FADE_DURATION`, `PLAYER_COLORS`.

**Decisions (Task 5 owner, 2026-10-25).**

- **One sideways punch fires one bolt.** The 1.8 s cooldown goes (`game.bolt_cooldown_s: 0`, the key stays). The gesture stage already sends one event per punch, at most one every 0.5 s. What the bolt looks like was decided later (Task 4 modifications: a fire blast).
- **The shield blocks.** While it is up (1 s, not counting the fade), bolts and beams that reach the player do no damage and leave no paint. A blocked bolt disappears; a blocked beam does not try that player again.
- **One colour table:** the colours from `game.py` (P1 red, P2 blue) replace `PLAYER_COLORS` in `scene/draw.py`, and the game, renderer and debug views all use it. Splats store the attacker's pid; the colour is looked up when drawing.
- **Pixels.** Positions are pixels in the mirrored frame, like in every other module. Speeds and sizes are in the config as fractions of the frame width or height, so the game plays the same at any resolution.

**Where each part goes.**

| In the code | Goes to | Change |
| --- | --- | --- |
| Constants (`MAX_HEALTH`, `ATTACK_COOLDOWN`, …) | `game:` config | Each with a reason; cooldown 0 |
| `PLAYER_COLORS` | `scene/draw.py`, `PLAYER_COLORS` | Replaces the current colours |
| `PaintSplat`, `Projectile`, `PaintBeam`, `Shield` | `interaction/game.py`, same names | Pixels; a splat stores the attacker's pid instead of a colour |
| `Player` | `interaction/game.py`, `GamePlayer` | Renamed: `Player` is the identity type in `core/types.py` |
| `attack()`, `beam_attack()`, `activate_shield()`, `_find_projectile_target()`, `_check_beam_hits()`, `_apply_damage()` | `interaction/game.py`, same names | No cooldown; the shield check is new |
| `update(dt, visible)` | `Game.advance(dt, visible)` | Renamed: `update(state)` is the module interface |
| `main.py`: actions on gesture edges, `bounds`, positions, `dt` | `Game.update(state)` | From `state.events`, `state.players` (boxes), `state.motion` (keypoints) and `Frame.t` |
| `reset()` | `Game.reset(pid=None)` | `None` = new round; a pid = that player starts fresh |
| `player`, `shield` (single-player shortcuts) | Removed | Not used anywhere |
| `test_game.py` (7 tests) | `tests/test_game.py` | Same tests, pytest, pixels |
| Key `r` after a win | `app/main.py`, key `r` | Already resets the whole pipeline, at any time (since the renderer merge: a new round, and `R` resets everything) |

**Steps.** Numbered after the gesture steps; each gets an entry under Changes below.

5. Colours, config and state: `PLAYER_COLORS` in `scene/draw.py`, the `game:` config, the dataclasses, `Game.__init__`, `get_player()`, `reset()`.
6. Combat: `attack()`, `beam_attack()`, `activate_shield()`, `advance()`, hit tests, damage, splats, winner, shield blocking. The tests from `test_game.py`, plus tests for blocking.
7. Pipeline: `Game.update(state)`, with `dt` from `Frame.t` (capped), positions and boxes from the state, events to actions, and only active players can be hit. Tests with fake states.
8. Debug view and CSV: `--until game` draws bolts, beams, shields, splats, health and the winner as simple shapes until the renderer merge; `log_fields`.
9. Check and record: run on the clips, `docs/phase-1/status.md`, Change Overviews.

**Verification.** Unit tests for every step. The two clips only produce two `hands_up` events, so they check beams but not bolts or shields. The Task 5 owner checks the full game on the webcam after step 8.

### Changes

#### Step 1 · Motion smoothing: `KeypointTracker` into `MotionFilter`

**Changed:** `src/ipcvgame/pose/motion.py`

- `MotionFilter` now does `KeypointTracker.update()` from `bodypose.py` per player (`_smooth()`, state in `_PlayerState`), with the same five rules:
  - keypoints under the confidence threshold count as missing;
  - a missing point keeps its last position and stays valid for `hold_s`;
  - an EMA smooths the positions;
  - a point that comes back after more than `hold_s` jumps to the detection;
  - a frame without a detection resets the player (`bodypose.py` reset when the pose model found nobody in the crop).
- The missing-frame counter became the time each keypoint was last seen (`last_good`, from `Frame.t`), so the hold is the same at any frame rate.
- Velocities are still the finite difference on `Frame.t`, now of the smoothed points. A point that jumps back gets velocity 0 on that frame, so the jump doesn't count as motion.
- The phase-1 pass-through stays as `backend: stub`, unchanged except for the new confidence threshold.

**Changed:** `src/ipcvgame/core/filters.py`: `EMA.__call__` takes an optional `snap` mask; masked rows jump to the new value instead of being smoothed. Without the mask it behaves as before, so the face stage is unaffected.

**Changed:** `configs/default.yaml`, `motion:` section

| Key | Value | In `bodypose.py` |
| --- | --- | --- |
| `backend` | `ema` | (new; `stub` = pass-through) |
| `kpt_conf_threshold` | 0.35 (was 0.5) | `CONF_T`; the gesture thresholds were tuned with it |
| `tau` | 0.048 s | `ALPHA = 0.5` per frame at 30 FPS: τ = −dt / ln(1 − α) |
| `hold_s` | 0.33 s | `MISSING_FRAMES = 10` at 30 FPS |

**Added:** `tests/test_motion.py` (7 tests: first detection snaps, EMA step after one time constant, hold → invalid → snap back, reset on no detection and on re-entry, stub pass-through, unknown backend). `tests/test_filters.py`: one test for the snap mask.

**Changed:** `tests/test_pipeline.py`, `test_velocity_uses_frame_time`: runs with `motion.tau=0` (no smoothing), so it still checks the plain finite difference.

#### Step 2 · Gesture rules

**Changed:** `src/ipcvgame/interaction/gestures.py`

- `lower_body_y()`, `detect_hands_up()`, `detect_crossed()`, `arm_extended()`, `arm_bent()` and `detect_punch()` from `bodypose.py` are module-level functions here, with the same names, logic and comments.
- They take `(kpts, valid, sw, cfg)`: the keypoints and validity from `Motion`, the shoulder width in pixels, and the `gestures:` config. Keypoints are indexed by name through `core/skeleton.py` instead of the file's own index constants.
- `detect_punch()` still returns `"L"`, `"R"` or `None`; its docstring notes that the model's `L` is the player's own right arm on our mirrored frame. The event will drop the side (step 3).
- `GestureRecognizer` is still the stub; step 3 connects it.

**Changed:** `src/ipcvgame/core/geometry.py`: new `angle_at(a, b, c)`, the helper from `bodypose.py`, so other tasks can use it too.

**Changed:** `configs/default.yaml`, `gestures:` section, each with a comment:

| Key | Value | In `bodypose.py` |
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

**Found: a hanging arm reads as a punch.** In the jump clip, the runners' arms hang straight (elbow 159–177°), so they pass the straightness check. The wrist is 1.0–1.4 sw from the shoulder, so it passes the reach check. The wrist is just above the hip keypoint, so it passes the height check. Meanwhile the other arm swings bent. `arm_extended()` checks the distance from the shoulder but not the direction, and the lowest allowed wrist position is the hip line. On all 268 false punch frames across both clips, the wrist is at most 0.96 sw to the side of its shoulder (median 0.19–0.23); a sideways punch at the other player reaches at least `punch_min_reach` (1 sw) sideways. Not changed in this step; the decision is for the Task 5 owner.

#### Step 2b · Punch reach measured sideways

**Changed:** `src/ipcvgame/interaction/gestures.py`, `arm_extended()`: the reach check uses the wrist's horizontal distance from its shoulder, `abs(wrist.x - shoulder.x) / sw`, instead of the straight-line distance. Everything else in the rule is unchanged, and `punch_min_reach` stays 1.0.

**Why:** a straight arm hanging down is also 1 sw or more from its shoulder, and with the wrist just above the hip keypoint it passed every check (step 2). On the 268 false punch frames the wrist was at most 0.96 sw to the side; a sideways punch at the other player reaches out at least 1 sw.

**Changed:** `configs/default.yaml`: the `punch_min_reach` comment says it is a sideways distance, and why.

**Added:** `tests/test_gestures.py`, `test_hanging_arm_above_hip_line_is_no_punch`: the jump-clip pose (1.9 sw from the shoulder, 0.2 sw to the side), which the old check accepted.

#### Step 3 · `GestureRecognizer`: events with hold times

**Changed:** `src/ipcvgame/interaction/gestures.py`

- `shoulder_width()` and `detect_all()` (new helpers): the shoulder-width check from `BodyPoseProcessor.process()` (`sw > 10`), then all three rules on one frame. Without a usable shoulder width every rule is false.
- `GestureRecognizer.update()` keeps per-player timers in `_PlayerState`:
  - **hands up and arms crossed** must hold for `confirm_s`. This is `Debounce` from `bodypose.py`, in seconds: the start time of the current run of frames instead of a frame count.
  - **punch** stays on for `punch_hold_s` after its last detected frame (`punch_until` in `bodypose.py`, now from `Frame.t` instead of `time.perf_counter()`).
  - **one `GestureEvent` when a gesture turns on**, named `hands_up`, `arms_crossed` or `punch`, with `conf` 1.0. This replaces `hands_were_up` / `arms_were_crossed` in `main.py`, and gives one `punch` event per punch.
  - a player missing from `Motion` starts over, as `bodypose.py` did when the pose model found nobody.
- `reset(pid)` clears one player (`reset_player()`).
- `draw_debug()`: the skeleton, the `H:C:P` line from `bodypose.py` under each body box (1 = on), and this frame's events.
- `log_fields()`: CSV columns `gest_p<pid>_hands_up`, `_arms_crossed`, `_punch` (1 while on) and `gest_p<pid>_events`.

**Changed:** `src/ipcvgame/app/pipeline.py`: a re-entering player also resets the gesture stage (`gestures.reset(pid)`).

**Changed:** `configs/default.yaml`, `gestures:` section, each with a comment:

| Key | Value | In `bodypose.py` |
| --- | --- | --- |
| `min_shoulder_px` | 10 | `sw > 10` in `process()` |
| `confirm_s` | 0.03 s | `CONFIRM_FRAMES = 2`: at 30 FPS the 2nd frame comes 0.033 s after the 1st |
| `punch_hold_s` | 0.5 s | `PUNCH_HOLD_S` |

**Behaviour differences from `bodypose.py`:**

- A frame without a usable shoulder width restarts the hands-up and crossed timers. `bodypose.py` skipped the frame and kept counting.
- Punches less than 0.5 s apart give one event; a fast double punch counts once.

**Added:** 8 tests in `tests/test_gestures.py`:
- hands up fires once after the confirm time, and again after a release;
- a one-frame glitch is no gesture;
- the crossed event name;
- one punch gives one event, a new punch after the hold time gives another;
- a missing player starts over;
- `reset()`;
- shoulders under 10 px;
- `log_fields()`.

#### Step 4 · Check and record

**Changed:** `docs/phase-1/status.md`

- "Where we are": `pose/motion.py` marked as smoothing done; `interaction/gestures.py` done, with `interaction/game.py` on its own row as the remaining stub. Test count 102.
- Task 2 and Task 4 sections: "Stub today" replaced by the merged status and a checklist of what is done and still open (model choice, `signals`, `tools/eval_motion.py`, `tools/eval_gestures.py`, the game logic, the punch classifier for later).
- Conventions: re-entry also calls `gestures.reset(pid)`; `EMA` has a `snap` mask; `core/geometry.py` has `angle_at`.
- Naming: gesture event names are `punch`, `hands_up`, `arms_crossed`.
- Performance: motion and gestures are ~0.1 ms each (steps 1 and 3).
- Known limitations: no more "skeletons jitter" or "no gestures fire"; gestures fire, but the game only prints them.

**Change Overviews** for Task 2 and Task 4 were filled in step by step (steps 1–3).

#### Step 5 · Colours, config and game state

**Changed:** `src/ipcvgame/scene/draw.py`: `PLAYER_COLORS` now has the team colours from `game.py`, P1 red `(55, 55, 235)` and P2 blue `(235, 110, 40)` (BGR). It was P1 blue and P2 orange, so blue meant P1 in the debug views but P2 in the game. Every debug view uses `player_color()`, so they all change with it.

**Changed:** `src/ipcvgame/interaction/game.py`

- The dataclasses `PaintSplat`, `Projectile`, `PaintBeam` and `Shield` from `game.py` keep their names and fields, with positions in pixels.
- `PaintSplat` stores the attacker's pid (`owner_id`) instead of a colour; the colour comes from `player_color()` when drawing.
- `Player` from `game.py` is now `GamePlayer`, because `Player` is already the identity type in `core/types.py`. `bounds` is a pixel box (`np.ndarray`), `x`/`y` are pixels.
- `Game.__init__(cfg)` starts with no players, instead of `{1: Player()}` for the single-player shortcuts `player` and `shield`, which are dropped (nothing used them).
- `get_player()` is the same.
- `reset()` is the same new round as in `game.py` (positions kept). `reset(pid)` (new, for the module interface) gives one player full health, no paint and no shield.
- `update()` still prints events (stub) until step 7.

**Changed:** `configs/default.yaml`, `game:` section, each with a comment:

| Key | Value | In `game.py` |
| --- | --- | --- |
| `max_health` | 100 | `MAX_HEALTH` |
| `bolt_cooldown_s` | 0.0 | `ATTACK_COOLDOWN` (1.8); 0 by decision |
| `bolt_speed_w`, `bolt_damage` | 1.4 frame widths/s, 10 | `PROJECTILE_SPEED`, `PROJECTILE_DAMAGE` |
| `beam_charge_s`, `beam_speed_w`, `beam_duration_s`, `beam_damage` | 0.35 s, 2.4 frame widths/s, 0.8 s, 12 | `BEAM_CHARGE_DURATION`, `BEAM_SPEED`, `BEAM_DURATION`, `BEAM_DAMAGE` |
| `beam_hit_radius_h` | 0.03 frame heights | `BEAM_HIT_RADIUS_Y` |
| `shield_s`, `shield_fade_s` | 1.0 s, 0.24 s | `SHIELD_DURATION`, `SHIELD_FADE_DURATION` |
| `splat_extra`, `splat_spread`, `splat_band` | 3, 0.08, [0.38, 0.62] | Numbers inside `_apply_damage()` |
| `max_dt_s` | 0.1 s | `min(now - previous_time, 0.1)` in `main.py` |

**Changed:** `docs/phase-1/status.md` (colour row: P1 red, P2 blue; it said "P1 orange, P2 blue", which was also wrong before). `docs/extra_features.md`: the "one colour table" item of feature 1 is marked done, and its follow-up row is removed.

**Added:** `tests/test_game.py` (4 tests): team colours, a new player at full health, `reset()` keeps positions and clears the round (`test_reset_starts_a_fresh_round` from `test_game.py`), `reset(pid)`.

#### Step 6 · Combat

**Changed:** `src/ipcvgame/interaction/game.py`

- `attack()`, `beam_attack()`, `activate_shield()`, `_find_projectile_target()`, `_check_beam_hits()` and `_apply_damage()` keep their names, logic and order from `game.py`, in pixels. `update(dt, visible)` is now `advance(dt, visible)`, because `update(state)` is the module interface (step 7).
- Speeds and the beam's hit radius are config fractions times the frame size (`frame_w`, `frame_h`, set from each frame in step 7). Bolts and beams leave the game at the frame edge (`0 … frame_w`), as at 0 … 1 before.
- `player_id` has no default any more (`game.py` defaulted to 1 for its single-player version).
- `attack()` sets the cooldown from `bolt_cooldown_s` (0, so every punch fires).
- A splat stores `owner_id`. The minimum box size in `_apply_damage()` is 1 px (was 0.001 of the frame).
- **Shield blocking (new):**
  - `shielded(pid)` is true while that player's shield is up, not during its fade-out;
  - a bolt that reaches a shielded player disappears without damage or paint;
  - a beam that reaches a shielded player adds them to `hit_player_ids` without damage, so it does not hit them later in its life, after the shield is gone.

**Added:** 13 tests in `tests/test_game.py`.
- The 6 combat tests from `test_game.py`, same names and numbers, in a 1000 × 1000 px frame so 0–1 coordinates become pixels:
  - red bolt hits blue;
  - blue bolt hits red;
  - a beam hits a target only once;
  - the cooldown, with `game.py`'s 1.8 s set in the config;
  - a fully painted player loses;
  - a bolt never hurts its owner.
- New: every punch fires with cooldown 0, a shield blocks a bolt, a shield blocks a beam for the rest of the beam, a fading shield does not block and then disappears, an invisible player is not hit, a bolt leaves the frame, and the same hit gives the same splats.

#### Step 7 · Pipeline: `Game.update(state)`

**Changed:** `src/ipcvgame/interaction/game.py`

- `update(state)` replaces the stub that printed events. It does the game part of the `main.py` loop, in the same order:
  1. **Frame size and time:** from `state.frame`. `dt` is the time since the last frame from `Frame.t`, capped at `max_dt_s` (`main.py` used `time.perf_counter()`, also capped at 0.1 s); 0 on the first frame.
  2. **Players:** every `active` player gets their body box from `Player.bbox` (was the BoT-SORT box) and their position from `body_position()`. Only `active` players can be hit (`main.py`: players identified this frame).
  3. **Actions:** each gesture event becomes an action through `_act()`:
     - **`punch` → `attack()`:** from the edge of the body box that faces the other player, at hip height;
     - **`hands_up` → `beam_attack()`:** from the midpoint of the wrists;
     - **`arms_crossed` → `activate_shield()`:** at the body.

     Direction: right for a player in the left half of the frame, as in `main.py`. An event is skipped if the player has no position this frame, as `main.py` skipped actions when `center_x` was `None`.
  4. **`advance(dt, visible)`.**
- `body_position(motion)` is `center_x` / `torso_y` from `bodypose.py`: the shoulder midpoint and the hip level, from the smoothed keypoints. It reuses `shoulder_width()` and `lower_body_y()` from `interaction/gestures.py`, and returns `None` without a usable shoulder width.
- `actions` lists this frame's actions as `(pid, "bolt" | "beam" | "shield")`, for the CSV in step 8.

**Changed:** `configs/default.yaml`, `game:` section: `min_shoulder_px` (10) and `hip_below_shoulders` (2.0), the same values as in `gestures:`. `bodypose.py` computed the position with those numbers; the game has its own copy because each module reads only its own config section.

**Added:** 8 tests in `tests/test_game.py`, with fake frame states:
- boxes and positions are read from the state;
- a punch fires from the box edge facing the other player, for both players;
- a beam starts between the wrists;
- a shield is raised at the body;
- time comes from `Frame.t`, with dt 0 on the first frame and capped after a stall;
- a bolt hits through `update()`;
- a lost player is not hit and their events are ignored;
- an event without shoulders is ignored.

#### Step 8 · Debug view and CSV columns

**Changed:** `src/ipcvgame/interaction/game.py`

- `draw_debug()` (`--until game`) draws the game state as plain shapes, until the renderer merge draws it properly. Everything is in the owner's or attacker's colour, and sizes scale with the frame width.
  - the skeleton (thin);
  - on each body box: the paint splats, and a health bar with `P<pid> <health>` (plus `shield` while it blocks), kept inside the frame;
  - bolts: a dot with a short trail;
  - beams: a growing circle while charging, then a line as thick as the hit band (2 × `beam_hit_radius_h`);
  - shields: an ellipse beside the player, thin while fading;
  - this frame's actions, bottom left, and `P<pid> wins - press r for a new round`.
- `log_fields()`: CSV columns
  - `game_p<pid>_health`;
  - `game_p<pid>_shield` (1 while it blocks);
  - `game_p<pid>_actions` (this frame's actions joined by `+`);
  - `game_bolts` and `game_beams` (in flight);
  - `game_winner`.

  `game.py` and `main.py` had no logging.

**Added:** 2 tests in `tests/test_game.py`: the debug view draws a bolt, a beam, a shield, splats and the winner without errors; the CSV columns.

#### Step 9 · Check and record

Done after the identity and renderer merges, so the status describes the game as it is now.

**Changed:** `docs/phase-1/status.md`
- "Where we are": `interaction/game.py` marked as merged (it still said "Stub, prints events").
- Task 4 section: the game (actions, damage, round end, keys), with a checklist of what is done and still open (balance on the webcam, simultaneous attacks and the other backlog features). "`Game.update` still prints events" removed.
- Known limitations: "the game only prints them" removed.

**Change Overviews** for `game.py` were filled in step by step (steps 5–8) and updated after the renderer merge.

## Task 4: Modifications

Game-design changes by the Task 5 owner after the merge steps above, starting from the webcam tests (2026-10-26). Each entry names the files it touched. The drawing described here (sprites, `_bolt_effect()` and the other loaders, `ring_geometry()`, `bar_geometry()`) and its config keys moved from `Game` and `game:` to `scene/renderer.py` and `render:` in the renderer merge (Task 5: Renderer, step 1).

### Bolt from the hand, health bar below the body

Two changes after the first webcam test (single player).

**Changed:** `src/ipcvgame/interaction/game.py`

- **Bolt from the hand.**
  - A punch now fires the bolt from the punching wrist, at its height, instead of from the body box edge at hip height.
  - `punching_hand(motion)` (new) picks the wrist that reaches furthest to the side of its shoulder. The punch event has no side, so the arm is found again from the keypoints.
  - This is `hand_x` / `hand_y` from `bodypose.py`, which computed the point but never used it.
  - Without a valid wrist, the bolt starts at the box edge at hip height, as in `main.py`.
  - The direction is unchanged: towards the middle of the frame.
- **Health bar in the debug view.** It was the width of the body box, along its top edge, so it grew when an arm was extended and jumped up with raised hands.
  - **Length:** depends only on the player's distance to the camera: `debug_bar_face_widths` (2.5) × `Face.scale`. Without a face, `debug_bar_shoulder_widths` (1.3) × the shoulder width; without either, the box width as before.
  - **Position:** at the bottom of the body box, centred under the shoulders, kept inside the frame. With a webcam the box usually ends at the bottom of the frame, so the bars sit along the bottom edge. Above the head (tried first) it overlapped the beams, which start between the raised hands.
  - The new `bar_geometry()` computes both.

**Changed:** `configs/default.yaml`, `game:` section: `debug_bar_face_widths` 2.5 and `debug_bar_shoulder_widths` 1.3. The two give the same length: on the test clips the shoulder keypoints are 1.8–2.0 face widths apart (median per clip).

**Changed:** `tests/test_game.py`: the box-edge punch test became "a punch fires from the punching hand" and "without wrists, from the box edge". The time test starts from the hand. 2 new tests: the bar's length and position ignore a wider box (arm out), and the shoulder-width fallback.

### Punch: wrist at least level with the shoulder

The webcam test still fired punches with a straight arm held low, at chest or belly height, reaching sideways. `arm_extended()` from `bodypose.py` accepted any wrist height between 1 sw above the shoulder line and the hips.

**Changed:** `src/ipcvgame/interaction/gestures.py`, `arm_extended()`:
- **Lower limit:** the wrist may be at most `punch_below_shoulder` × sw below that arm's own shoulder, instead of above the hip level. The arm's own shoulder is used because the shoulders tilt during a punch.
- **Upper limit:** unchanged; it still excludes hands up.
- `lower_body_y()` is still used by `detect_crossed()` and the game.

**Changed:** `configs/default.yaml`, `gestures:` section:
- new `punch_below_shoulder: 0.2`, with a comment. The request was "at least level with the shoulder". A level arm puts the wrist at shoulder height give or take keypoint noise, so 0.2 sw keeps those punches; 0 makes the rule strict.
- The `punch_above_shoulders` comment no longer mentions the hips.

**Changed:** `docs/phase-1/status.md`: the punch description says the wrist must be at least level with its shoulder.

**Added:** 2 tests in `tests/test_gestures.py`: a straight sideways arm at chest height (0.75 sw below the shoulder) is no punch; a punch 0.125 sw below the shoulder still counts.

### Power attack drawn as a magic ball and beam (magic_effect frames)

The beam (hands up) is drawn with the `assets/magic_effect/` frames ("Magic Attack" by NYKNCK, itch.io, credited in the README) instead of a plain line, as in the pack's preview (`magic.gif`, kept out of the repo): a magic ball between the player's hands, and the beam coming out of it. Each team has its own colour.

**Added:** `src/ipcvgame/scene/magic_beam.py`, `MagicBeam` (Task 5, so the renderer can use it later).
- **Ball (`M5025-1` … `-26`, 512 × 512 px):** drawn on the beam's start point, the wrist midpoint when the hands went up. Its 26 frames play once over the beam's whole life (charge plus firing, 1.15 s), so it grows while charging, swirls while the beam fires, and shrinks to a dot as the beam ends.
- **Beam (`L5020-1` … `-3`, 256 × 256 px):** the beam's cross-section, thickest first: horizontal stripes, identical in every column. One column is kept and stretched to the beam's length, so any length is exact. After the charge the beam starts at the thinnest band, thickens one step every `beam_ramp_s`, and thins again in its last steps, as in the preview. The full band is as thick as the hit band (2 × `beam_hit_radius_h` × frame height), so what you see is what hits. The beam is drawn under the ball.
- **Team colour:** the pack's colours (red) belong to `beam_effect_owner` (P1, the red team). For any other player, every pixel's hue is turned by the same angle, so the pack's main hue becomes that player's team colour from `scene/draw.py`. Brightness and saturation stay. P2's copy is blue.

**Changed:** `src/ipcvgame/scene/draw.py`:
- `blend_rgba()` (new): alpha-blends a BGRA sprite onto the frame at a top-left corner, clipped to the frame.
- `hue_of()`, `main_hue()`, `shift_hue()` (new): hue of a colour; main hue of sprites; a hue-turned copy of a sprite. `main_hue()` is a circular mean, because hue is an angle: the pack's reds lie just above 0 and just below 180, and a plain average or median of those would give cyan.

**Added:** `src/ipcvgame/scene/animation.py`, `load_frames()`: reads the frames of a folder or a glob pattern, in natural order, and checks they are RGBA and the same size. The pack numbers its frames without leading zeros, so name order would put `M5025-10` before `M5025-2`.

**Changed:** `src/ipcvgame/interaction/game.py`
- `draw_debug()`: each beam is drawn with its owner's `MagicBeam` (`_beam_effect()`, loaded on first draw so the game logic runs without the files). With `beam_ball: null` the old line and charge circle are drawn.
- `advance()`: **the beam's tip now stops at the frame edge, and the beam lasts its full duration** (`beam_duration_s`, 0.8 s). In `game.py` a beam ended as soon as its tip left the frame, about 0.3 s after firing, which would cut off the ball animation and the thinning at the end. Gameplay effect: each target is still hit at most once per beam, but stepping into a beam that is still active now also hits.

**Changed:** `configs/default.yaml`, `game:` section, each with a comment:
- `beam_ball` and `beam_band` (glob patterns for the two frame sets);
- `beam_ramp_s` (0.1 s, the preview's frame time);
- `beam_ball_h` (0.36 of the frame height: the ball fills about half its frame, so ~130 px at 720p, three times the beam's thickness, as in the preview);
- `beam_effect_owner` (1).

**Added:** `tests/test_magic_beam.py` (9 tests, on a small made-up pack):
- frames load in natural order;
- the ball plays once over the beam's life;
- the band ramps up and down;
- the beam reaches the tip and no further, with the ball on top;
- left beams, and only the ball while charging;
- `main_hue()` is circular for red;
- `shift_hue()` keeps white and transparency;
- `blend_rgba()` clips;
- each team gets its own beam colour from the real pack.

`tests/test_animation.py` (3 tests: a folder and a glob in natural order, missing or mixed frames raise). `tests/test_game.py`: the beam stops at the frame edge and lasts its duration; the debug view without the effect.

### Normal attack drawn as a fire blast (assets/beam frames)

The bolt (punch) is drawn with the 6 fire-blast frames in `assets/beam/` ("Fire Attack FX - Pixel Art Effect - FX061" by NYKNCK, itch.io, credited in the README) instead of a dot. Each team has its own colour.

**Added:** `src/ipcvgame/scene/fire_bolt.py`, `FireBolt` (Task 5, so the renderer can use it later).
- **The frames:** 131 × 32 px pixel art, a one-shot fire blast drawn to the right: a launch flash (frame 1), the blast in flight (2–3), its break-up into sparks (4–6). The blast moves within the frames.
- **Split by phase** (`bolt_sprite_launch`, `bolt_sprite_flight`; the rest is the end):
  - launch: shown for one frame time after the punch;
  - flight: looped while the bolt flies;
  - end: played once over `bolt_end_s` where the bolt ended: a hit, a shield or the frame edge.
- **Anchor:** each frame is drawn with its own head (its rightmost opaque column) on the bolt's hit point, so the front of the fire is always on the bolt and the flames trail behind it. A single anchor for all frames put the launch flash ~130 px behind the hand, because the blast starts at the left of the pack's frames. The height is anchored on the middle row of the flight frames, shared, so the blast doesn't jump up and down. Bolts going left are mirrored.
- **Team colour:** the frames' colours (orange fire) belong to `bolt_effect_owner` (P1, the red team). Other players get a copy with every hue turned by the same angle (P2: blue), as for the beam.

**Changed:** `src/ipcvgame/interaction/game.py`
- `Projectile` has a new field `age` (s since it was fired), counted up in `advance()`. It drives the animation.
- `BoltEnd` (new) and `Game.bolt_ends`: when a bolt stops (a hit, a shield or the frame edge), `advance()` keeps its position, direction and owner for `bolt_end_s` so the break-up can play there. It is for drawing only; `reset()` clears it.
- `draw_debug()` draws the bolt ends and the bolts with their owner's `FireBolt` (`_bolt_effect()`, loaded on first draw). With `bolt_sprite: null` the old dot is drawn.

**Changed:** `configs/default.yaml`, `game:` section, each with a comment:
- `bolt_sprite` (`assets/beam`), `bolt_sprite_launch` (1), `bolt_sprite_flight` (2);
- `bolt_sprite_fps` (12, a common pixel-art rate; the two flight frames flicker like fire);
- `bolt_end_s` (0.25: the 3 break-up frames at about the same rate);
- `bolt_sprite_h` (0.08 of the frame height: the blast ~27 px thick and ~240 px long at 720p);
- `bolt_effect_owner` (1);
- `bolt_speed_w` 1.4 → 1.0: the frame is crossed in 1 s instead of 0.7 s, so the bolt is easier to see coming and shield.

**Added:** `tests/test_fire_bolt.py` (7 tests, on made-up frames):
- the frame for each phase;
- each frame's head on the bolt, flames behind;
- left bolts mirrored;
- the blast centred on the bolt's height;
- the hue shift gives the team colour;
- too few frames raise;
- each team gets its own colour from the real frames.

`tests/test_game.py`:
- a bolt's age advances;
- a bolt end is kept where the bolt stopped (shield, frame edge) for `bolt_end_s` and cleared by `reset()`;
- no frames gives a dot.

The tests from `test_game.py` (the `arena` fixture) keep `game.py`'s bolt speed 1.4, so their numbers still hold; the time test reads the speed from the config.

### Beam: one per raise of the hands

The webcam test showed the power attack sometimes firing from several places at once. Reproduced with synthetic poses through the gesture stage and the game. Three causes:

- **Short dropouts.** Hands up (and arms crossed) turned off on the first frame its rule failed, for example when a wrist keypoint dipped for a frame or two. Raising for 0.03 s more then started a new beam where the hands were now, while the first beam (alive for 1.15 s) was still firing.
- **Wrists above the frame.** The motion stage keeps a lost wrist for 0.33 s (`motion.hold_s`); after that, hands up turned off, and coming back fired again.
- **No limit in the game.** `beam_attack()` from `game.py` started a new beam for every hands-up event, even while that player's previous beam was alive.

A third effect is left as is: raising one arm before the other passes through the punch pose, so it can fire a bolt as well.

**Changed:** `src/ipcvgame/interaction/gestures.py`, `GestureRecognizer.update()`:
- Once on, hands up and arms crossed stay on until their rule has failed for `release_s`. `_PlayerState.last_seen` keeps the last time each rule held.
- Turning on is unchanged (`confirm_s`). It is the same idea as `punch_hold_s` for the punch.

**Changed:** `src/ipcvgame/interaction/game.py`, `beam_attack()`: returns whether it fired, and does nothing while that player's previous beam is still alive (charging, firing, or not yet past the frame edge). `_act()` records the `beam` action only when it fired.

**Changed:** `configs/default.yaml`, `gestures:` section: `release_s: 0.3` (~9 frames at 30 FPS), with a comment. With the motion stage's 0.33 s hold, wrists can be lost for ~0.6 s without ending the gesture.

**Changed:** `docs/phase-1/status.md`: the Task 4 gesture description mentions the release time.

**Changed:** `tests/test_gestures.py`: the repeat test lowers the hands for longer than `release_s`. **Added:**
- a 2-frame dropout does not restart hands up or arms crossed;
- a gesture ends after `release_s`;
- `tests/test_game.py`: one beam per player at a time, and the other player can still fire.

### One action at a time; reset to a normal position first

The webcam test sometimes fired the major attack and a normal attack together. Raising the arms one after the other passes through the punch pose, so a punch fired and then the beam. A punch shortly after a beam also flew while the beam was still firing.

**Changed:** `src/ipcvgame/interaction/gestures.py`, `GestureRecognizer.update()`:
- **Reset first:** after any event (punch, hands up, arms crossed) the player is not ready (`_PlayerState.ready`) until they are back in a normal position, with none of the three gestures on. A gesture that turns on before that sends no event. When the gestures count as off comes from their existing hold times: 0.5 s after the last punch frame (`punch_hold_s`), 0.3 s after lowering the hands or uncrossing (`release_s`). So no new threshold is needed.
- **Same frame:** if several gestures turn on in the same frame, only the first in `GESTURES` sends an event. That is the priority order: hands up (the major attack), arms crossed, punch.
- `draw_debug()` adds "reset first" to the `H:C:P` line while the player is not ready. `log_fields()` adds `gest_p<pid>_ready`.

**Changed:** `src/ipcvgame/interaction/game.py`, `attack()`: no bolt while the player's own beam is still alive (1.15 s), so the two attacks never overlap even after a quick reset. The other player is not affected.

**Known limit:** when the arms go up one after the other, the punch pose usually comes first. The punch then fires and the beam is blocked: never both, but the punch wins. Telling a raise from a punch needs a better punch detector (`docs/limitations.md`, "Punch").

**Changed:** `docs/phase-1/status.md`: the gesture description mentions the reset rule.

**Added:**
- `tests/test_gestures.py`:
  - a new action needs a normal position first (punch, then hands up gives nothing; after a reset, arms crossed fires);
  - gestures starting together send only one event, the major attack first;
  - the `ready` CSV column.
- `tests/test_game.py`: no bolt while the player's own beam is alive. `test_log_fields` now fires its bolt from P2, since P1's beam is still alive there.

### Shield drawn as a ring of fire (FX019 frames)

The shield (arms crossed) is drawn with the frames in `assets/shield/FX019/` instead of an ellipse. Each team has its own colour.

**Added:** `src/ipcvgame/scene/fire_ring.py`, `FireRing` (Task 5, so the renderer can use it later).
- **The frames:** 45 frames of 759 × 512 px, a ring of fire at the pack's 10 FPS (`PFX019.gif`).
  - Frames 1–13: whirling arcs that close into the ring.
  - Frames 14–29: the burning ring, looping.
  - Frames 30–45: exact copies of 14–29, not used.
- **Forming** (`shield_form_frames`) plays once over `shield_form_s` (0.3 s) instead of the pack's 1.3 s. The shield blocks from its first frame and lasts only `shield_s` (1 s), so the ring has to be up quickly.
- **The ring** (`shield_loop_frames`) then loops at `shield_sprite_fps`.
- **Anchor and size,** both measured from the loop frames:
  - the centre is the middle of the frames' combined outline, fixed, so it doesn't jitter;
  - the size is the median height of the ring, because flames flick outside it now and then.
- **Fade:** the ring fades out with the shield's fade (`shield_fade_s`) through the blend's opacity.
- **Team colour:** orange fire belongs to `shield_effect_owner` (P1); other players get a hue-turned copy (P2: blue), as for the bolt and the beam.

**Changed:** `src/ipcvgame/interaction/game.py`
- `draw_debug()` draws each shield with its owner's `FireRing` (`_shield_effect()`, loaded on first draw). With `shield_sprite: null` the old ellipse is drawn.
- `ring_geometry()` (new): the ring follows the player. It is centred on the chest (halfway between shoulders and hips) and is `shield_ring_sw` shoulder widths across, so its size follows the player's distance to the camera. Without a shoulder width it goes where the shield went up, as wide as the body box. Blocking is unchanged and does not depend on where the ring is drawn.

**Changed:** `configs/default.yaml`, `game:` section, each with a comment:
- `shield_sprite` (`assets/shield/FX019`);
- `shield_form_frames` (13), `shield_loop_frames` (16), `shield_form_s` (0.3), `shield_sprite_fps` (10);
- `shield_ring_sw` (2.6: frames the upper body with the arms crossed in front);
- `shield_effect_owner` (1).

**Changed:** `.gitignore`: preview GIFs of the sprite packs (`assets/**/*.gif`) stay out of the repo; they are reference only.

**Added:**
- `tests/test_fire_ring.py` (6 tests, on made-up frames):
  - forming, then the loop, with the copies unused;
  - the ring's centre and size are measured;
  - the ring is centred and sized when drawn;
  - the fade lowers opacity;
  - the hue shift, and too few frames raise;
  - each team gets its own ring from the real frames.
- `tests/test_game.py`: the ring sits on the chest at 2.6 shoulder widths, and the fallback without shoulders.


### Attacks fly towards the other player

On the webcam, after the players had switched sides, a beam sometimes flew away from the opponent. `_act()` used `main.py`'s rule: fire right from the left half of the frame, left from the right half. That is only right while each player keeps to their own half.

**Changed:** `src/ipcvgame/interaction/game.py`: new `Game.facing(pid, positions)`. Bolts, beams and the shield face the nearest other player in view (their shoulder midpoint). The old rule, towards the middle, is the fallback when nobody else is in view or both stand at the same x. Attacks still fly sideways only; aiming in 2D stays as feature 11.

**Added:** `tests/test_game.py`: beams fly at the other player in four layouts (normal, switched, both in the right half, both in the left half), and towards the middle with nobody else in view.

## Task 3: Identity tracking

### Plan

**Source.** `PlayerIdentityTracker` in `Project/bodypose.py`, the tracking loop in `main.py`, `Project/botsort.yaml`, `Project/osnet.py` and `Project/osnet_embedding.py` (after the pre-merge OSNet change above). `Project/identitytracker.py` only starts `bodypose.py`'s demo and is not merged.

**What the code does.**

- **Tracking:** `main.py` runs a second model, `yolo11n.pt`, as a person detector (score ≥ 0.5, `IDENTITY_CONFIDENCE`) with Ultralytics' BoT-SORT (`botsort.yaml`). BoT-SORT gives each person a track id that lasts while it can follow them from frame to frame.
- **Identity:** `PlayerIdentityTracker` turns track ids into player ids.
  - A track already linked to a player keeps that player; OSNet does not run again.
  - A new track's person crop (at least 40 × 20 px, `MIN_PERSON_HEIGHT` / `MIN_PERSON_WIDTH`) goes through OSNet. Its 512-number vector is compared with each stored player's vector by cosine similarity. The best player at ≥ 0.6 (`REID_THRESHOLD`) becomes this track's player. Players already used this frame are skipped (`excluded_player_ids`).
  - No match: the track must be seen 5 times (`NEW_PLAYER_CONFIRMATION`) before it becomes a new player with the next free id, up to 3 (`MAX_PLAYERS`). Its last vector is stored and never updated.
  - `forget_missing_tracks()`: a track that is not in this frame's output loses its link, and its player's pose state is reset (`reset_player()`). When the track comes back, OSNet runs again.
- **Per frame:** BoT-SORT plus pose processing on each linked track's box; the pose model runs once per player crop.

**Decisions (Task 5 owner, 2026-10-26).**

- **One model per frame.** BoT-SORT runs on the YOLO-pose boxes that the pose stage already returns, through Ultralytics' `BOTSORT` class; `yolo11n.pt` goes. Measured on `two_people_fitness_challenge.webm`: 0.4 ms per frame.
- **Occluded, then lost.** A player whose track BoT-SORT still holds (not seen for at most `track_buffer` frames) is `"occluded"`: the link stays, the box is BoT-SORT's prediction, and no state is reset when the same track comes back. Only when BoT-SORT drops the track is the player `"lost"`; when OSNet links a new track to them, `reentered` is True for that frame. `bodypose.py` dropped the link and reset the pose state on every frame without the track.
- **Two players** (`max_players: 2`): the game is 1v1, so a third person never becomes a player. `bodypose.py` allowed 3.
- **BoT-SORT settings,** from `botsort.yaml` with two changes:
  - motion compensation off (`gmc_method: none`): the webcam does not move. Measured: same track ids, 4.4 → 0.4 ms per frame;
  - a new track needs score 0.5 (`new_track_thresh`), like `main.py`'s detector, while detections from 0.35 (`pose.det_conf`) still continue an existing track. `main.py` dropped every box under 0.5 before BoT-SORT.
- **Stored vectors are updated (new).** While a player is active and their box overlaps no other detection, their vector is blended with a fresh OSNet vector at intervals, if the two are still similar enough to be the same person. One player per frame, in turn, so OSNet adds at most one crop (~10 ms on the GPU) to a frame.
- **Linked tracks are checked (new, after the clip test in step 5).** The same OSNet run also checks the link: if the crop looks more like the other player, the links are corrected. `bodypose.py` never checked a linked track, so when BoT-SORT carried a track onto the other player, the wrong player stayed linked.
- **Faces only (new, after a webcam test in step 5).** A track is identified only when the pose model sees a face (nose or eyes), so a coat the pose model takes for a person never becomes a player.
- **OSNet weights from Hugging Face.** `osnet_x1_0_msmt17_combineall` from the authors' repo ([kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet)), saved as `models/osnet_x1_0_msmt17_combineall.pth`. It is the same network trained on the same data (all 4,101 MSMT17 identities) as `Project/models/osnet_x1_0_msmt17.pth`, but it has a public download with a checksum. The old file matches no file in that repo, probably because it came from torchreid's Google Drive zoo, which needs `gdown`. Compared on the two test clips (stored vector from the first frame, as in `bodypose.py`; similarity to the same player vs to the other player):

  | Weights | Fitness clip: same / other | Jump clip: same / other | ms per crop |
  | --- | --- | --- | --- |
  | `Project/models/osnet_x1_0_msmt17.pth` | 0.66 / 0.59 | 0.90 / 0.55 | 9.3 |
  | `osnet_x1_0_msmt17_combineall` (chosen) | 0.55 / 0.47 | 0.90 / 0.54 | 9.5 |
  | `osnet_ain_x1_0_msmt17` | 0.63 / 0.56 | 0.90 / 0.61 | 10.8 |

  The two x1.0 files separate the players equally well (gap 0.07–0.08 on the fitness clip, where both wear black shirts and jeans and change posture, and 0.35–0.36 on the jump clip). OSNet-AIN, which the model zoo reports as better on unseen datasets, separates them worse on both clips (0.07 and 0.29) and needs a second network file, so it is not used. The 0.6 threshold is kept for now. With the chosen weights, only 36% of the fitness clip's same-player similarities reach it (the jump clip: 100%), so it is checked again on re-entry clips.
- **Kept from `bodypose.py`:** greedy matching per track with excluded ids, the 0.6 threshold, the minimum crop size, the OSNet x1.0 network and its preprocessing.
- **Small changes found while merging (step 2):**
  - tracks already linked to a player are handled before new tracks, so a new track cannot take a player whose own track is visible this frame (`main.py` handled tracks in detection order and threw such a clash away afterwards);
  - a track that matched no player while both player slots are taken (a bystander) tries OSNet again only every 0.5 s; `bodypose.py` tried every frame, ~10 ms each time;
  - a new player takes the lowest free id instead of a counter, so `reset(pid)` frees that id.
- **Not in this merge:** `Player.pos` stays `(0, 0)`.

**Where each part goes.**

| In the code | Goes to | Change |
| --- | --- | --- |
| `osnet.py` | `identity/osnet.py` | Moved |
| `OSNetEmbeddingExtractor` (`osnet_embedding.py`) | `identity/embedding.py`, same name | Device and weights path from config |
| `Project/models/osnet_x1_0_msmt17.pth` | `models/osnet_x1_0_msmt17_combineall.pth` | Same network and data, from Hugging Face; README download command |
| `Project/models/yolo11n.pt` | Removed | BoT-SORT runs on the pose boxes |
| `PlayerIdentityTracker` | `identity/tracker.py`, `IdentityTracker` | Same steps; occluded/lost and `reentered`; vector update |
| `_find_player()` | `IdentityTracker._find_player()` | Threshold from config |
| `update()` | `IdentityTracker._identify()` | Confirmation in seconds; slow retry when slots are full; lowest free id |
| `main.py` loop over tracks | `IdentityTracker.update(dets, frame)` | Linked tracks first; builds the `Player`s |
| `forget_missing_tracks()` | `IdentityTracker` | Links drop when BoT-SORT drops the track, not when it is missing for a frame |
| `person_detector.track(..., botsort.yaml)` in `main.py` | `IdentityTracker`, Ultralytics `BOTSORT` on the `PoseDetection` boxes | `yolo11n.pt` removed |
| `botsort.yaml` | `identity.botsort:` config | Each value with a reason; two changed (above) |
| `REID_THRESHOLD`, `NEW_PLAYER_CONFIRMATION`, `MAX_PLAYERS`, `MIN_PERSON_HEIGHT`, `MIN_PERSON_WIDTH` | `identity:` config | Seconds instead of frames; 2 players |
| `pose_processor.reset_player()` on lost players | `Player.reentered` | Already used by the pipeline to reset face, motion and gestures |
| "Identifying..." label, box colours | `IdentityTracker.draw_debug()` | |
| Left/right-half stub | `identity.backend: zones` | Kept for tests without weights |
| `test_osnet.py` | `tests/test_identity.py` | Loads the weights if present, else skipped; no webcam |

**Steps.** Each step gets an entry under Changes below.

1. OSNet into `identity/`: `osnet.py`, `embedding.py`, the Hugging Face weights in `models/` with a README download command, a load test.
2. Config and identity logic: the `identity:` section, `IdentityTracker` with confirmation, re-identification, max players, occluded/lost, `reentered` and the vector update. Tests with a fake OSNet and fake tracks.
3. BoT-SORT on the pose boxes: `BOTSORT` fed with `PoseDetection`s, predicted boxes for occluded players. Tests with synthetic moving boxes.
4. Debug view, `log_fields` and warm-up: boxes with status and track id, "Identifying..." for pending tracks; CSV columns; one OSNet call at start-up (the first takes 84 ms).
5. Fixes from the clip and webcam tests: link check and correction; faces only.
6. Check and record: Change Overviews, open issues, `docs/phase-1/status.md`.

**Verification.** Unit tests for every step and runs on `data/test_clips/` and the recorded `data/clips/session.mp4`. None of the clips has crossing, occlusion or re-entry, so step 5 uses edited copies of `session.mp4` (one player blacked out for 3.5 s; the two halves of the frame swapped for 3.5 s). ID switches on real crossings can only be counted once such clips are recorded. The Task 5 owner checks re-entry live on the webcam.

### Changes

#### Step 1 · OSNet into `identity/`

**Added:** `src/ipcvgame/identity/osnet.py`: `Project/osnet.py` moved unchanged (the network and `load_weights()`); only the header names the new place and weights file. Its upstream code style is kept, so ruff's `UP` and `B` rules are off for this one file (`pyproject.toml`).

**Added:** `src/ipcvgame/identity/embedding.py`: `OSNetEmbeddingExtractor` from `Project/osnet_embedding.py`, same preprocessing (BGR → RGB, 256 × 128, ImageNet mean and std) and the same length-1 vector.
- The weights path and device come from the config (`"auto"` = CUDA if available), and a relative path is taken from the repo root.
- A missing weights file raises an error that points to the README.
- Nothing is printed on load.
- Checked against `osnet_embedding.py` on a crop from `data/clips/session.mp4` with the same weights: identical vectors (max difference 0.0).

**Changed:** `configs/default.yaml`, `identity:` section: `reid_weights` (`models/osnet_x1_0_msmt17_combineall.pth`) and `reid_device` (`auto`).

**Changed:** `requirements.txt`: the identity comment names `identity/osnet.py` and the Hugging Face weights.

**Added:** `tests/test_identity.py`, replacing `Project/test_osnet.py` (4 tests, no webcam):
- the network gives 512 numbers;
- `load_weights()` raises when no layer matches;
- with the real weights: vectors have length 1 and a crop matches itself better than another crop (skipped when the weights are not downloaded);
- a missing weights file raises.

#### Step 2 · Config and identity logic

**Changed:** `src/ipcvgame/identity/tracker.py`: `IdentityTracker` now holds `PlayerIdentityTracker` from `bodypose.py` as `backend: reid`, with the same names: `embeddings` (your `players`), `track_to_player`, `pending_tracks`, `_find_player()`, `forget_missing_tracks()`. Your `update()` is `_identify()`. The tracker comes in step 3; until then the config keeps `backend: zones`.

Compared with `bodypose.py`:
- **Same rules:** a linked track keeps its player without OSNet; a new track's crop (≥ 40 × 20 px) is matched to the most similar stored player at ≥ 0.6, skipping players already used this frame; otherwise it becomes a new player after being seen for a while.
- **Seconds instead of frames:** `NEW_PLAYER_CONFIRMATION` (5 frames) became `new_player_s` (0.13 s). A pending track must be seen on every frame; a frame without it starts the wait again, as in your `forget_missing_tracks()`.
- **Two players** instead of 3 (`max_players`). A new player takes the lowest free id, not the next number, so `reset(pid)` frees the id.
- **Occluded, then lost:** `forget_missing_tracks()` drops a link only when the tracker no longer holds the track at all. While it holds it as lost, the player is `"occluded"` with the tracker's predicted box. Your code dropped the link on the first frame without the track and reset the player's pose state; now nothing is reset until the player is `"lost"`, and `reentered` is True on the frame OSNet links a new track to a lost player.
- **Link moves to a new track:** when OSNet finds a player on a new track while their old track is still held (occluded), the old link is dropped. That is not a re-entry.
- **Linked tracks first:** `main.py` went through the tracks in detection order, so a new track could take a player whose own track came later in the list; that track then lost its player for the frame. Now linked tracks are handled first and their players are excluded for new tracks.
- **Slow retry:** in your code a track that found no player while all slots were taken was dropped from `pending_tracks` and tried again on the next frame, so a bystander cost an OSNet run every frame. Now it retries after `retry_s` (0.5 s).
- **Stored vectors are updated (new):** `_update_embedding()`. One active player per frame, in turn, at most every 0.5 s, and only when their box overlaps no other detection. If the fresh vector's similarity to the stored one is at least 0.5, the stored vector moves 0.1 of the way towards it (and is scaled back to length 1). Your code stored the first vector and never changed it.
- **Players as `Player`s:** `update(dets, frame)` returns `dict[pid, Player]` from `core/types.py` (box, detection, status, `reentered`); `pos` stays `(0, 0)`.
- **The left/right stub** stays as `backend: zones`, unchanged except that it reads `max_players` (was `num_players`).
- **`draw_debug`:** player boxes with status (dashed when not active), green "Identifying..." boxes for tracks without a player (your label), and the stored player ids.

**Changed:** `configs/default.yaml`, `identity:` section

| Key | Value | In `bodypose.py` |
| --- | --- | --- |
| `backend` | `zones` until step 3 | (new; `reid` = your tracker) |
| `max_players` | 2 | `MAX_PLAYERS = 3` |
| `reid_threshold` | 0.6 | `REID_THRESHOLD` |
| `new_player_s` | 0.13 s | `NEW_PLAYER_CONFIRMATION = 5` frames at 30 FPS |
| `min_crop_h`, `min_crop_w` | 40, 20 px | `MIN_PERSON_HEIGHT`, `MIN_PERSON_WIDTH` |
| `retry_s` | 0.5 s | (new; every frame) |
| `update_every_s`, `update_weight`, `update_min_sim`, `update_max_iou` | 0.5 s, 0.1, 0.5, 0.0 | (new; no update) |

`num_players` became `max_players`; `docs/phase-1/status.md`'s `--set` example follows.

**Added:** `tests/test_identity.py`, 12 tests with a fake OSNet (the vector is the crop's mean colour) and a scripted tracker:
- a new player after `new_player_s`; a linked track skips OSNet;
- occluded → lost → re-entered with the same look; the same track back from occluded is not a re-entry; a new track takes over from a held one;
- two players and a bystander who never becomes P3 and is retried only after `retry_s`;
- a player in use is not given to another track; a too-small crop is not identified;
- the vector update blends by `update_weight` and players take turns; it is skipped when boxes overlap or the crop is too different;
- `reset()` and `reset(pid)`; an unknown backend raises.

#### Step 3 · BoT-SORT on the pose boxes

**Added:** `src/ipcvgame/identity/tracks.py`, `BotSortTracks`: replaces `person_detector.track(..., tracker="botsort.yaml")` in `main.py`.
- Ultralytics' `BOTSORT` class gets the boxes and scores of the pose stage's `PoseDetection`s, so `yolo11n.pt` no longer runs; one model per frame.
- It returns the tracks seen this frame (track id and which detection) and the tracks BoT-SORT still holds as lost, with its predicted box (Kalman, constant velocity) clipped to the frame. `IdentityTracker` uses the predicted box for occluded players.
- `reset()` clears BoT-SORT (key `r`).

**Changed:** `src/ipcvgame/identity/tracker.py`: `backend: reid` builds `BotSortTracks` from the config.

**Changed:** `configs/default.yaml`, `identity:` section
- `backend: reid` is now the default.
- New `botsort:` block with every value from `Project/botsort.yaml` and a reason. Two changed:

  | Key | Value | `botsort.yaml` | Why |
  | --- | --- | --- | --- |
  | `new_track_thresh` | 0.5 | 0.25 | A new track needs the score `main.py` asked of its detector (`IDENTITY_CONFIDENCE`); boxes from 0.35 still continue a track |
  | `gmc_method` | `none` | `sparseOptFlow` | The webcam does not move; 4.4 → 0.4 ms per frame |

- `track_buffer` (30) counts frames, as in Ultralytics: a held track, and its player's `"occluded"` status, lasts 1 s at 30 FPS and 1.7 s at the ~17 FPS of `data/clips/session.mp4`.
- BoT-SORT's second matching round (scores 0.1–0.25) gets no boxes, because the pose stage drops everything under 0.35 (`pose.det_conf`). In `main.py` the detector's 0.5 limit did the same.

**Changed:** `tests/test_pipeline.py` and `tests/test_recorder.py` use `identity.backend=zones`, like `face.backend=stub`, so they need no model files.

**Added:** `tests/test_identity.py`, 4 tests on made-up boxes:
- two moving boxes keep their track ids;
- a missing box is held as lost with its predicted box moving on, and removed after `track_buffer` frames;
- a box under `new_track_thresh` does not start a track but does continue one;
- `IdentityTracker` with the real BoT-SORT and a fake OSNet finds both players.

#### Step 4 · Debug view, CSV columns and warm-up

**Changed:** `src/ipcvgame/identity/tracker.py`
- **Debug view (`--until identity`):** each player box shows status, BoT-SORT track id and, on frames where OSNet ran for that player, the similarity. Tracks without a player show `main.py`'s "Identifying..." with their track id. A bottom line lists the stored players and this frame's OSNet runs.
- **`log_fields` (new):** `identity_p<pid>_track` (linked track id), `identity_p<pid>_sim` (similarity of a match or vector update this frame), `identity_new_best_sim` (best similarity of a track without a player to any stored player that is not in use; shows how close a returning player came to `reid_threshold`), `identity_pending` (tracks without a player), `identity_osnet_runs` (OSNet crops this frame). Status, box and `reentered` were already in the base CSV.
- **Warm-up (new):** one OSNet run on a blank crop at start-up, when the weights are loaded. On `data/clips/session.mp4` the first frame's identity stage went from 115 ms to 22 ms (two players identified in it).

**Added:** `tests/test_identity.py`: the CSV columns for a linked player, a pending track and a lost player, and the debug view draws.

#### Step 5 · Fixes from the clip and webcam tests

Two problems that `bodypose.py` has too, found by testing the merged tracker.

**Swapped players stayed swapped.** On a copy of `data/clips/session.mp4` with the two halves of the frame swapped for 60 frames, BoT-SORT carried P1's track onto the other player (their box landed where P1 had been). A linked track never runs OSNet again, in `bodypose.py` as in step 2, so P1 stayed on the wrong person for all 60 frames. The vector update's OSNet run had already seen it (similarity 0.45 to P1, under `update_min_sim`, so nothing was blended in).

**Changed:** `src/ipcvgame/identity/tracker.py`: `_update_embedding()` became `_check_and_update()`, and runs before the `Player`s are built:
- If the crop looks more like another stored player (at least `reid_threshold`, and more than like its own player):
  - that player is active too: their crop is checked as well, and the two links are swapped only if it looks more like this player. One doubtful crop cannot move two players.
  - that player is not active (occluded or lost): the link moves to them, and this player loses their link (lost). Tracks waiting for a free player try again right away instead of after `retry_s`.
- Players whose link changed get `reentered` on that frame, so the face, motion and gesture stages drop their state: the position jumped.
- Otherwise, the vector update as before.

On the swapped copy, P1 and P2 were on the wrong person for 6 of the 120 frames around the two jumps (was 60): one frame at the first jump, five at the second, until P2's next check.

**A coat became a player.** On the webcam, the pose model took a coat for a person. With one player in view, it took the free second slot after 0.13 s, and kept it. On `session.mp4`, such extra detections (a coat, a chair) score up to 0.56 with confident shoulders (median 0.92), but the best face keypoint (nose or eyes) has a median confidence of 0.15; the real players have 1.00 (5th percentile 0.97).

**Changed:** `src/ipcvgame/identity/tracker.py`: a track without a player is identified (OSNet, a new player) only when its best face keypoint reaches `min_face_conf`. OSNet does not run on the coat at all. A player who comes back facing away is found once they face the camera.

**Changed:** `configs/default.yaml`, `identity:` section: `min_face_conf: 0.5` (new); the `update_every_s` comment names the check.

**Added:** `tests/test_identity.py`, 4 tests:
- a detection without a face is never identified, and OSNet does not run;
- a track carried onto an occluded player: the link moves, and the first player is found again on a new track;
- both tracks carried onto the other player: the links are swapped and both are `reentered`;
- one doubtful crop: no swap.

#### Step 6 · Check and record

**Changed:** `docs/merge_log.md`: Change Overviews for Task 3 (where your code is now, and what changed), and the open issues found before the merge marked as fixed.

**Changed:** `docs/phase-1/status.md`
- "Where we are": `identity/tracker.py` marked as merged; test count 200.
- Task 3 section: "Stub today" replaced by what the tracker does now, with a checklist of what is done and still open.
- Performance: the identity stage.

**Changed:** `docs/extra_features.md`: identity marked as done in "finish the merge"; new feature 12 (game start: instructions and pose calibration); technical follow-ups from the identity merge.

## Task 5: Renderer

### Plan

**Source.** `renderer.py` (`Renderer.render()` and its helpers) and the palette picture it loads, plus the keys in `main.py`. The game logic it reads (`game.py`) is already merged into `interaction/game.py`.

**What the code does.**

- **Camera image:** shrunk to half size and scaled back up with the nearest pixel, for a blocky retro look; everything else is drawn sharp on top.
- **Game objects** (all in team colours): bolts as two rectangles, the paint beam as a pulsing square while charging and then a thick line, the shield as the palette picture with a team-colour outline, a white flash and a splash of white dots.
- **Paint on the players:** each splat is a small pixel cross on the body box; the crosses grow as the player's health drops (`coverage` = 1 − health / 100).
- **Health:** a bar above the head (`head_x`, `head_y` from `bodypose.py`) with "PLAYER N" and an outline.
- **HUD:** a dark bar at the top with "COLOUR CLASH" and the controls ("HANDS UP: PAINT BEAM   PUNCH: BOLT   Q: QUIT"), or "STEP INTO VIEW" when nobody is seen; "Q: QUIT" at the bottom.
- **End of the game:** a dark panel in the middle, "PLAYER N WINS!" in the winner's colour and "R: RESTART   Q: QUIT". In `main.py`, key `r` after a win calls `game.reset()`: a new round with the same players.

**What our code does now.**

- The default view (`Renderer.draw()`) shows player boxes, skeletons, face effects and timings, but no game objects.
- The game is only visible in `--until game` (`Game.draw_debug()`): the fire blast, magic ball and beam, and ring of fire sprites (Task 4 modifications), splats as filled circles, a health bar below the body and one line of text for the winner.
- **End of the game looks frozen.** `Game.advance()` stops once someone wins (as in `game.py`), so bolts, their break-up sparks and beams stay where they were, under the winner text, while the camera keeps running. Splats drawn as large circles pile up into blobs.
- Key `r` resets the whole pipeline, including identity, so P1 and P2 can swap colours in the next round.

**Decisions (Task 5 owner, 2026-10-27).**

- **The renderer draws the game.** The sprite effects, splats, health bars and HUD move to the renderer (Task 5). `interaction/game.py` keeps only the game logic, so the Task 4 owner explains rules, not drawing. `--until game` goes back to simple shapes (bolt dots, beam lines with their hit band, shield ellipses, health numbers) for debugging the logic; the default view is the game.
- **Health bar below the body** (Task 4 modifications), restyled with `renderer.py`'s outline and "PLAYER N" label.
- **Splats as `renderer.py`'s pixel crosses** that grow as health drops, instead of the debug circles.
- **Paint on the torso and a steady health bar (after the Task 5 owner's check in step 3).** The game keeps splats in a band of the body box, and the renderer draws that band on the torso (shoulders to hips), so the paint no longer slides off the body when an arm goes out. The health bar's length is smoothed. Recording where on the torso a hit really landed stays as feature 13 in `docs/extra_features.md`.
- **Pixel look on:** `render.pixel_size: 2` (`renderer.py`'s half size); 1 turns it off.
- **End of the game (Task 5 owner):** the panel from `renderer.py`; bolts, sparks and beams in flight are not drawn once there is a winner, and the frame behind the panel is dimmed, so nothing appears frozen. Key `r` starts a new round (`game.reset()`, as in `main.py`) at any time; `R` (shift) resets the whole pipeline, including identity. Restarting by a gesture is part of feature 12.
- **No skeletons or boxes in the game view** (`renderer.py` drew none): the player is named on their health bar. `render.show_skeleton` brings the skeletons back for testing.
- **Not merged:** `renderer.py`'s own bolt, beam and shield drawing and the palette picture; the sprites from the Task 4 modifications replace them.

**Where each part goes.**

| In the code | Goes to | Change |
| --- | --- | --- |
| `Renderer.render()` | `scene/renderer.py`, `Renderer.draw(state, game)` | Pixels; reads `Game` state and `state.faces` |
| Pixelated camera image | `Renderer.draw()` | `render.pixel_size` |
| `_draw_player_paint()` | `scene/renderer.py`, same name | Splat colour from the attacker's pid |
| `_draw_player_health()` | `scene/renderer.py`, same name | Below the body (Task 4 modification) |
| Top bar, controls, "STEP INTO VIEW" | `scene/renderer.py`, `_draw_hud()` | Our three gestures named |
| `_draw_game_over()` | `scene/renderer.py`, same name | Dimmed frame; objects in flight hidden |
| `_draw_shield()`, `_draw_shield_splash()`, `_draw_paint_beam()`, bolt rectangles, palette `.jpg` | Not merged | Replaced by the fire ring, magic beam and fire blast |
| (ours) sprite loading and drawing in `Game` (`_bolt_effect()`, `_beam_effect()`, `_shield_effect()`, `ring_geometry()`, `bar_geometry()`) | `scene/renderer.py` | Config keys move from `game:` to `render:` |
| `main.py`: key `r` after a win | `app/main.py`: `r` = new round, `R` = full reset | |

**Steps.** Each step gets an entry under Changes below.

1. Sprites to the renderer: loading and drawing of the fire blast, magic beam and fire ring, the ring and health-bar geometry, and their config keys move from `Game` to the renderer; `Game.draw_debug()` back to simple shapes.
2. Game view: pixel look, splats as pixel crosses, health bars, no skeletons or boxes, face effects on top.
3. HUD and end of the game: top bar with controls or "STEP INTO VIEW", the game-over panel, keys `r` and `R`.
4. Fixes from the check: paint on the torso; smoothed health-bar length.
5. Check and record: runs on the clips with `--record-view`, Change Overviews, `docs/phase-1/status.md`, `docs/extra_features.md`.

**Verification.** Unit tests for each step (drawing on made-up frames, keys). Views recorded with `--record-view` on `data/clips/session.mp4` and the test clips. The Task 5 owner checks the game, including the end of a round, on the webcam.

### Changes

#### Step 1 · Sprites to the renderer

**Changed:** `src/ipcvgame/scene/renderer.py`: the renderer now draws the game objects (`draw_game()`), moved from `Game.draw_debug()`:
- bolts as the fire blast and its break-up where they ended, beams as the magic ball and beam, shields as the ring of fire on the chest; plain shapes when a sprite is set to `null`;
- `bolt_effect()`, `beam_effect()`, `shield_effect()`: the per-team sprite loading from `Game` (`_bolt_effect()` and the others). The three near-identical loaders share one helper, `_team_copy()`: the frames as they are for the owner, a hue-turned copy for every other player;
- `ring_geometry()` and `bar_geometry()`, unchanged except that they read the torso from `Game.torso()`.

**Changed:** `src/ipcvgame/interaction/game.py`
- No drawing code or sprite imports left beyond `draw_debug()`. It shows the game state as plain shapes for checking the logic: dashed body boxes with health and shield, splats as dots, bolts as dots with a tail and an x where they ended, beams as a line with their hit band, shields as ellipses, the actions and the winner.
- New read-only values for drawing, so the renderer does not read the `game:` config: `beam_charge_s`, `beam_life_s`, `shield_fade_s`, `bolt_end_s`, `beam_band_px()`, `shield_opacity()`, and `torso()` (shoulder midpoint, shoulder and hip level, shoulder width), which `body_position()` now uses.

**Changed:** `configs/default.yaml`
- The sprite keys (`bolt_sprite*`, `bolt_effect_owner`, `beam_ball`, `beam_band`, `beam_ramp_s`, `beam_ball_h`, `beam_effect_owner`, `shield_sprite`, `shield_form_frames`, `shield_loop_frames`, `shield_form_s`, `shield_sprite_fps`, `shield_ring_sw`, `shield_effect_owner`) moved from `game:` to `render:`, with their comments.
- `debug_bar_face_widths` / `debug_bar_shoulder_widths` became `render.bar_face_widths` / `bar_shoulder_widths`.
- `bolt_end_s` stays in `game:`: the game keeps a bolt's end point that long.

**Changed:** tests
- `tests/test_game.py`: the health-bar, ring and sprite tests moved out; one test for `torso()` and the drawing values.
- `tests/test_renderer.py` (new): every object drawn, plain shapes without sprites, the health-bar and ring geometry tests from `test_game.py`.
- `tests/test_fire_bolt.py`, `test_fire_ring.py`, `test_magic_beam.py`: the per-team sprite tests use the renderer.

#### Step 2 · Game view

**Changed:** `src/ipcvgame/scene/renderer.py`, `Renderer.draw()`, in `renderer.py`'s order: camera image, paint, game objects, then (new) face effects, then health bars and timings.
- **Pixel look:** `pixelate()`, `renderer.py`'s shrink and nearest-pixel upscale, by `pixel_size` (2). Sprites, face effects and text are drawn sharp on top.
- **Paint:** `_draw_player_paint()` from `renderer.py`: a cross of 5 cells per splat, inside the body box. The cells grow with coverage (1 − health / max health), so a beaten player looks more painted. Changed: the colour comes from the attacker's pid (`player_color(splat.owner_id)`), not a colour stored in the splat; the sizes are in the config. They replace the debug circles that piled up into blobs.
- **Health bar:** `_draw_player_health()` from `renderer.py` (dark backing, "PLAYER N", off-white outline, fill in the team colour), but below the body at `bar_geometry()`'s place (Task 4 modifications) instead of above the head. Bar and label heights scale with the frame instead of `renderer.py`'s fixed 10 px.
- **Only visible players** get paint and a health bar (`renderer.py` drew only detected players).
- **No skeletons or boxes** unless `show_skeleton` is on.

**Changed:** `configs/default.yaml`, `render:` section

| Key | Value | In `renderer.py` |
| --- | --- | --- |
| `pixel_size` | 2 | Half size, `INTER_NEAREST` back up |
| `show_skeleton` | `false` | (none drawn) |
| `splat_cell_w`, `splat_cell_h` | [0.02, 0.025], [0.015, 0.025] | `0.02 + 0.025 * coverage`, `0.015 + 0.025 * coverage` |
| `bar_h`, `bar_font_h` | 0.021, 0.017 of the frame height | 10 px bar, label at scale 0.8, at 480p |

**Added:** `tests/test_renderer.py`: the pixel blocks; paint crosses in the attacker's colour, inside the box and growing with damage; health-bar fill, and no bar for a lost player.

#### Step 3 · HUD and end of the game

**Changed:** `src/ipcvgame/scene/renderer.py`
- **Top bar** (`_draw_hud()`) from `renderer.py`: a dark bar with "COLOUR CLASH" in P1's colour, and on the right the controls or "STEP INTO VIEW" while nobody is seen; the keys at the bottom. Changed: the controls name our three gestures ("PUNCH: FIRE BOLT   HANDS UP: MAGIC BEAM   CROSS ARMS: SHIELD"; `renderer.py` had no shield), and sizes scale with the frame. The timings moved below the bar.
- **End of the game** (`_draw_game_over()`) from `renderer.py`: the dark panel framed in the winner's colour with "PLAYER N WINS!" and the keys. New: the frame behind it is darkened (`game_over_dim`), and bolts, their break-ups, beams and shields are no longer drawn once there is a winner. The game stops then (`game.py`'s rule), so they stood still in mid-air, which looked like a frozen game.

**Changed:** `src/ipcvgame/app/pipeline.py` and `app/main.py`: key `r` calls `Pipeline.new_round()` (`game.reset()`: full health, no paint, same players), as `main.py` did after a win, but at any time. `R` (shift) resets every module, as `r` did before, so the players are identified again. `README.md` lists the keys.

**Changed:** `configs/default.yaml`, `render:` section: `hud_h` (0.07 of the frame height: `renderer.py`'s 34 px at 480p), `hud_title_scale` and `hud_text_scale` (1.4 and 1.0 at 480p, as in `renderer.py`), `game_over_dim` (0.5).

**Added:** `tests/test_renderer.py`: the top bar is drawn; after a win the panel is in the winner's colour and the bolt is not drawn; `new_round()` keeps the players and `reset()` clears them.

#### Step 4 · Fixes from the check: paint on the torso, steady health bar

Two problems the Task 5 owner saw in the game view after step 3.

**The paint moved off the body.** The game keeps splats as fractions of the body box (`game.py`), and the box widens when an arm goes out, so the splats slid with it onto the arm or the background.

**Changed:** `src/ipcvgame/scene/renderer.py`: `paint_area()` gives the torso (between the shoulders, from the shoulder line to the hips, from `Game.torso()`), or the box's splat band without a shoulder width. `_draw_player_paint()` stretches the game's splat band (`game.splat_band`, 0.38–0.62 of the box) over that area, and sizes the cells by it. On `data/clips/session.mp4` the paint area's centre moves half as much from frame to frame as the box band did (95th percentile 8 against 16 px for P1, 3.8 against 11.9 px for P2); what is left is the body moving.

**The health bar changed length.** Its length comes from the face width, which jitters by a few px, and falls back to 1.3 shoulder widths on frames without a face.

**Changed:** `src/ipcvgame/scene/renderer.py`: the length goes through the shared `EMA` filter per player (`bar_length_tau_s`, 1 s), reset when the player re-enters and on `R`. On the session clip the frame-to-frame change went from 10.5 to 3.4 px (95th percentile) for P1, and from 3.8 to 1.3 px for P2.

**Changed:** `src/ipcvgame/interaction/game.py`: `Game.splat_band` (read-only, for drawing). `configs/default.yaml`: `splat_cell_w` / `splat_cell_h` are fractions of the torso now ([0.04, 0.05] and [0.04, 0.06]: `renderer.py`'s sizes, as the box is ~2 shoulder widths wide and ~2.5 torso heights tall); `bar_length_tau_s: 1.0` (new).

**Added:** `tests/test_renderer.py`: the paint area stays the same when the box widens, with the box band as fallback; the bar length is smoothed when the face is missed.

#### Step 5 · Check and record

Runs with `--record-view` on `data/clips/session.mp4` (a full round, to the winner panel) and both test clips.

**Changed:** `src/ipcvgame/scene/renderer.py`: on the 428 px wide fitness clip, the controls line ran over the title, and the health bars covered the key line. `fit_scale()` shrinks a line to the space it has; the health bars stay above the key line.

**Added:** `tests/test_renderer.py`: text fitted to a narrow frame.

**Changed:** `docs/merge_log.md`: Change Overviews for `renderer.py`. `docs/phase-1/status.md`: renderer row, render timing, test count. `docs/extra_features.md`: rendering done; removing `garb/` is now possible.

## Task 5: Modifications

Design changes by the Task 5 owner after the renderer merge, from the webcam check (2026-10-27). Each entry names the files it touched.

### Health bars pinned to the bottom

The bar below each body (Task 4 modifications) moved with the player and changed length with the face width, even smoothed (renderer step 4): too much movement on screen. Now the bars stay in place, like in a fighting game.

**Changed:** `src/ipcvgame/scene/renderer.py`
- `_draw_player_health()`: P1's bar in the bottom-left corner, P2's in the bottom-right, each `bar_w` of the frame wide. P2's bar fills from its right edge, so both drain towards their corner; P2's label is right-aligned. Other pids have no slot.
- A bar is shown for every player the game knows, also while they are out of view, so their health stays visible.
- `bar_geometry()` and the length smoothing (`_bar_len`, `Renderer.reset()`) are removed: nothing depends on the player's position or face any more.
- The key line ("R: NEW ROUND   Q: QUIT") is centred between the bars, shrunk to fit the gap, on a dark backing so it reads on a bright floor.

**Changed:** `src/ipcvgame/app/pipeline.py`: `reset()` no longer resets the renderer (it has no state left).

**Changed:** `configs/default.yaml`, `render:` section: `bar_w` (0.3 of the frame width, leaving the middle for the key line) replaces `bar_face_widths`, `bar_shoulder_widths` and `bar_length_tau_s`.

**Changed:** `tests/test_renderer.py`: the three bar-geometry and smoothing tests are replaced by one: P1's bar bottom left and P2's bottom right wherever they stand, P2 filling from the right edge, and the bar still shown for a player out of view.

## Game start: instructions and pose calibration

The first feature built from the backlog in `docs/extra_features.md` (feature 12 there, the "Feature 12:" and "Feature 1:" commits), by the Task 5 owner on 2026-10-28. It touches Task 2 (`pose/measure.py`), Task 3 (`identity/tracker.py`), Task 4 (`interaction/game.py`) and Task 5 (`scene/`, `app/`); each step names the files it touched.

### Plan

**What.** A game UI around the round. Before, the game started the moment the players were seen. Now there is a start screen, a short phase in which each player learns the moves while the game measures them, a countdown, and a result screen.

**Decisions (Task 5 owner, 2026-10-28).**

- **Phases:** `waiting` → `calibration` → `countdown` → `playing` → `result`. Damage, bolts and beams only count in `playing`.
- **Play again:** from `result`, both players with hands up for ~2 s (or key `r`) go straight to `countdown`. The same players keep their calibration, so they do not see the instructions again. Key `R` (full reset) goes back to `waiting`, and identity forgets the players too.
- **Start places by position:** a start place is ticked in its team colour once someone stands in that half of the frame with a visible face, whoever it is. Identity numbers players in the order it first sees them, so ticking by player number would show the wrong colour.
- **One move at a time per player,** on a panel on the player's own half: the neutral pose, then punch, hands up and arms crossed. Each move shows a stick figure and its effect, and is ticked off when its gesture event arrives. A move not done within ~8 s flashes as a reminder. Nothing fires, so the phase doubles as a practice round.
- **Pose calibration** in the neutral pose (arms down, facing the camera, ~2 s): shoulder width, arm length and torso length per player, as medians over the frames. They let the game check gesture thresholds per player (a short arm), and they are the depth reference for `Player.pos`.
- **Identity at calibration:** P1 becomes the player in the left place, and the OSNet vector from the neutral pose replaces the first-sight vector.
- **Depth in metres** by the pinhole model, with the webcam's field of view and an average shoulder width as assumed values in config.
- **Testing without the intro:** `--no-intro` plays from the first frame, for clips recorded without the intro. `--solo` runs the intro with one player.
- **Logging:** the steps are recorded in this log; no entries in Change Overviews, since no teammate's code moved.

**Steps.** Each step gets an entry under Changes below.

1. Game phases (Task 4).
2. Waiting, countdown and result screens (Task 5).
3. Instruction panels (Task 5).
4. Pose calibration (Task 2 measurements, Task 4 rules).
5. Identity at calibration (Task 3).
6. Depth reference (Task 3, `Player.pos`).
7. Docs; webcam check by the Task 5 owner.

### Changes

#### Step 1 · Game phases

**Changed:** `src/ipcvgame/interaction/game.py`
- `Game.phase` with the five phases (`PHASES`), the time the phase started, `phase_elapsed()` and `countdown_left()`. `_next_phase()` moves on:
  - `waiting` → `calibration`: every start place taken for `ready_s` (`place_taken()`: someone active with a face, by shoulder midpoint, in that half of the frame);
  - `calibration` → `countdown`: every round player has done every move;
  - `countdown` → `playing`: after `countdown_s`;
  - `playing` → `result`: a winner;
  - `result` → `countdown`: every round player holds hands up for `replay_hold_s`.
- Holds (`_Hold`) survive dropouts shorter than `hold_gap_s`. Hands up for the replay uses the gesture rule `detect_hands_up()` per frame, because an event only marks its start.
- In `calibration`, gesture events tick off the player's current move (`_tick()`, logged as the action `learned_<move>`) instead of firing; `advance()` runs only in `playing`. `GamePlayer.move` / `move_t` hold each player's progress; `current_move(pid)` gives the next move.
- `new_round()` (key `r`) starts from the countdown with the same players; `reset()` (key `R`) clears the moves too and goes back to `waiting`. With `game.intro: false` both go straight to `playing`, as before.
- `game.round_players`: the players the intro waits for (2, or 1 for `--solo`).
- The phase is shown in `--until game` and logged as `game_phase`, the next move as `game_p<pid>_move`.

**Changed:** `src/ipcvgame/app/main.py`: `--no-intro` (`game.intro=false`) and `--solo` (`game.round_players=1`); the docstring's key list. `src/ipcvgame/app/pipeline.py`: `new_round()` calls `Game.new_round()`.

**Changed:** `configs/default.yaml`, `game:` section: `intro`, `round_players`, `ready_s`, `countdown_s`, `replay_hold_s`, `hold_gap_s`, `head_above_shoulders`.

**Changed:** `tests/test_game.py`: the existing tests run with `intro: false`; new tests for every transition, hold gaps, new round and reset, the solo intro, logging and the debug view.

#### Step 2 · Waiting, countdown and result screens

**Changed:** `src/ipcvgame/scene/renderer.py`
- **Waiting:** the start places (`start_place()`, `_draw_start_place()`), dashed with "STAND HERE" until taken, then solid with a tick and "READY". With one player, a single place in the middle.
- **Countdown:** 3-2-1 in the middle, each number settling over its second (`_draw_big_text()`); "FIGHT!" for `fight_s` once the round starts.
- **Result:** `_draw_game_over()` shows each player's final health in their colour, and "HANDS UP TO PLAY AGAIN" with a bar that fills during the hold (`Game.hold_fraction()`).
- The top bar says what the phase expects (`status_text()`); health bars are shown from the countdown on.

**Changed:** `configs/default.yaml`, `render:` section: `start_place_w`, `start_place_y`, `start_text_scale`, `countdown_h`, `fight_s`.

**Changed:** `tests/test_renderer.py`: status texts, start places and their tick, the countdown and "FIGHT!", the result's hold bar; the game-over test sets the `result` phase.

#### Step 3 · Instruction panels

**Added:** `src/ipcvgame/scene/instructions.py`: a stick figure per move as fixed joint positions in shoulder widths (`ARMS`, `figure_points()`), drawn with OpenCV lines (`draw_figure()`) and mirrored for P2 so the punch points at the opponent; the move texts (`MOVE_TEXT`: name, how to do it, effect); `draw_tick()`.

**Changed:** `src/ipcvgame/scene/renderer.py`, `_draw_instructions()`: per player at the bottom of their half (`instruction_box()`), over the legs so the face, arms and hat stay visible; one panel in the middle with `--solo`. The move's name in the team colour, the figure, how to do it and its effect, and one dot per move (done, now, to do). "NICE!" with a tick for `nice_s` after a move, "DONE!" when all are done; after `remind_s` without the move, the frame flashes and "TRY IT NOW!" appears.

**Changed:** `configs/default.yaml`, `render:` section: `instruction_w`, `instruction_y`, `instruction_alpha`, `instruction_title_scale`, `instruction_text_scale`, `remind_s`, `remind_hz`, `nice_s`.

**Changed:** `tests/test_renderer.py`: each figure, as keypoints, passes the gesture rule of its own move and no other; every move has a figure and texts; the panels and the reminder flash.

#### Step 4 · Pose calibration

**Added:** `src/ipcvgame/pose/measure.py` (Task 2): `shoulder_px()` and `arm_px()` (shoulder to elbow plus elbow to wrist, so a bent arm measures the same as a straight one) from the smoothed keypoints.

**Changed:** `src/ipcvgame/interaction/game.py`
- The neutral pose is the first calibration move (`CALIBRATION_MOVES`), checked by the game every frame (`_neutral_pose()`): the player in view with the head turned at most `neutral_max_yaw_deg`; shoulders, elbows, wrists and hips seen; both wrists at least `neutral_wrist_below_sw` below the shoulder line and slower than `neutral_max_speed_sw`.
- Held for `neutral_s` (gaps up to `hold_gap_s`), `_measure()` stores the medians of one sample per frame in `GamePlayer.shoulder_px` / `arm_px`. A hold that starts over drops its samples. Kept on a new round, cleared by `reset()`.
- `arm_sw()` gives the arm in shoulder widths; `short_arm()` is true under `min_arm_sw`. `neutral_fraction()` for the panel's bar.
- Logged as `game_p<pid>_shoulder_px`, `_arm_px` and `_arm_sw`, and shown in `--until game`.

**Changed:** `src/ipcvgame/scene/renderer.py`: the neutral panel fills a bar while the pose is held, then shows "NICE! MEASURED"; the punch panel adds "STRETCH YOUR ARM ALL THE WAY" for a short arm.

**Changed:** `src/ipcvgame/scene/instructions.py`: the cross-arms panel teaches wrists crossed just past the middle of the chest ("CROSS WRISTS IN FRONT OF CHEST"). With the hands on the opposite shoulders, the pose model puts each wrist back on its own side and the rule misses the cross; details in `docs/limitations.md`.

**Changed:** `configs/default.yaml`, `game:` section: `neutral_s`, `neutral_wrist_below_sw`, `neutral_max_speed_sw`, `neutral_max_yaw_deg`, `min_arm_sw`.

**Changed:** `tests/test_game.py`: the calibration helpers stand in a real neutral pose; medians with noisy frames, each reason a pose is not neutral, a hold that restarts, a short arm.

**Added:** `docs/limitations.md`: known limits and improvements for the report, starting with the missed high arm cross.

#### Step 5 · Identity at calibration

**Changed:** `src/ipcvgame/identity/tracker.py` (Task 3)
- `order_left_to_right(xs)`: renumbers the given players by x, lowest number leftmost. Stored vectors, track links and pending refreshes move with them, and both come out as `reentered` on the next update, so motion, face and gestures drop their per-player state. The zones backend ignores it: its P1 is always the left half.
- `refresh_vector(pid, frame)`: replaces the stored vector with one from this frame's crop. If another person overlaps the crop, it is taken on the first update where the crop is clear.
- `calibrated`: the players whose vector is from the neutral pose; logged as `identity_p<pid>_calibrated`, and "cal" on the box in `--until identity`.

**Changed:** `src/ipcvgame/app/pipeline.py`, `_calibrate_identity()`: when the phase goes from `waiting` to `calibration`, it passes each round player's position to `order_left_to_right()`; on a `learned_neutral` action it calls `refresh_vector()`.

**Changed:** `src/ipcvgame/interaction/game.py`: `player_x()` (shoulder midpoint, or the box centre), shared by `place_taken()` and the ordering.

**Changed:** `tests/test_identity.py`: renumbering, players already in order, a refreshed vector, a refresh that waits for a clear crop, reset, the zones backend. `tests/test_pipeline.py`: the pipeline calls identity at the right moments.

#### Step 6 · Depth reference

**Changed:** `src/ipcvgame/identity/tracker.py`, `_position()`: `Player.pos` is now (x, depth) in metres instead of zeros, by the pinhole model, depth = f · S / s:
- f, the focal length in px, from the horizontal field of view: f = (width / 2) / tan(hfov / 2);
- S, the real shoulder width, assumed the same for everyone; s, the shoulder width in px, from the pose keypoints;
- after calibration (`set_body_reference()`), the depth at calibration is the reference and is scaled by the torso length: depth = depth_cal · torso_cal / torso. The torso does not shrink when the player turns; the shoulder width does. Shoulder width remains the fallback when the hips are not seen;
- x = the shoulder midpoint's offset from the image centre · depth / f;
- smoothed with the shared `EMA` filter, restarted on a re-entry; kept when no keypoints are usable.

**Changed:** `src/ipcvgame/pose/measure.py`: `torso_px()`, shoulder midpoint to hip midpoint. `src/ipcvgame/interaction/game.py`: the calibration also stores the median torso length (`GamePlayer.torso_px`, logged as `game_p<pid>_torso_px`). `src/ipcvgame/app/pipeline.py`: on `learned_neutral` it also passes the shoulder width and torso length to `set_body_reference()`.

**Changed:** `configs/default.yaml`, `identity:` section: `camera_hfov_deg` (70°, typical for a 720p/1080p webcam), `shoulder_m` (0.39 m, between the adult averages for women and men), `pos_kpt_conf`, `pos_tau_s`.

**Changed:** `tests/test_identity.py`: position from the shoulder width, the torso after calibration (a turn does not change the depth), kept without usable keypoints.

#### Step 7 · Docs

**Changed:** `README.md`: the intro, `--no-intro`, `--solo` and the keys. `docs/extra_features.md`: the code map rows for the game phases, intro screens, identity and keys; the feature moved here; follow-up rows for the webcam checks. Code comments that said "feature 12" now name this section.

## Game start: Modifications

Changes to the game start after it was finished. Each entry names the files it touched.

### Players line up in the calibration

The neutral pose now also needs the two players to stand level: side by side, at the same distance from the camera. Then each player's calibrated depth lies on one line, which a later dodge by stepping forward needs (`docs/extra_features.md`, feature 7: a dodge is a change in depth from that line).

**Changed:** `src/ipcvgame/interaction/game.py`
- `_check_line_up()`, every calibration frame, by depth (`Player.pos`, from the shoulder width; the players face the camera in the neutral pose): level when the two depths are less than `line_up_depth_m` apart. The smaller depth is the nearer player. The feet would be a cue that does not depend on body size, but they are usually out of frame (they are on `session.mp4`). Without a depth for both players there is nothing to check; someone not in view means not level.
- Hysteresis: once not level, the gap must shrink under `line_up_hysteresis` of the tolerance before the players are level again.
- `_measure()`: the neutral pose only counts while the players are level, so a player is not measured while they stand off the line.
- `line_up_hint(pid)`: "back" for the nearer player, "wait" for a player whose neutral pose waits; None when level, or when nobody needs the neutral pose any more. An already measured player is still told to step back while the other is measured.
- State in `level`, `nearer` and `line_gap`, reset when a phase starts. Logged as `game_level` and `game_line_gap`, and shown in `--until game`.
- With one round player (`--solo`) there is no one to line up with.

**Changed:** `src/ipcvgame/scene/renderer.py`, `_draw_instructions()`: with two players, the neutral pose panel says "STAND LEVEL WITH THE OTHER PLAYER" in place of "= MEASURING YOU" (the bar shows that), so a reminder still fits. While the players are not level, it shows "STEP BACK TO LINE UP" (nearer player) or "WAIT FOR THE OTHER PLAYER" in the team colour instead (`LINE_UP_TEXT`).

**Changed:** `configs/default.yaml`, `game:` section: `line_up_depth_m` (0.4 m: wide, because depth assumes the same shoulder width for everyone) and `line_up_hysteresis` (0.75).

**Why the hysteresis:** on `session.mp4`, where the players were not asked to line up, they stood a median 0.35 m apart by depth, so the gap crossed 0.4 m back and forth within 0.1 s, and the hint flickered.

**Changed:** tests
- `tests/test_game.py`: level by depth; the nearer player sent back, then measured once level; a measured player still told to step back; no line-up with one player; hysteresis. In `test_not_neutral`, a player out of view now also holds up the other player's measurement.
- `tests/test_renderer.py`: the line-up hint replaces the reminder; the stand-level line with two players only.

### Gestures count only while facing the camera

A gesture now counts, as an action in the round and as a move in the calibration, only while the player faces the camera with head and body. A gesture made turned away is ignored, and the player is told to face the camera.

**Changed:** `src/ipcvgame/interaction/game.py`
- `facing_camera(state, pid)`: a tracked face turned at most `act_max_head_yaw_deg`, and the body turned at most `act_max_body_yaw_deg`. `act_facing_camera: false` turns the rule off.
- `body_yaw_deg()`: turning shrinks the shoulder width in the image by cos(angle) but not the torso length, so the shoulder-to-torso ratio against the one from the neutral pose (facing the camera) gives the body's turn, at any distance. It needs the calibration, so with `--no-intro` only the head is checked.
- `_track_body_yaw()`: the body turn is taken only from frames with the arms down, and the last value is used for a gesture. The gestures themselves move the shoulders: crossing the arms rolls them in and raising both hands lifts them, so on `session.mp4` a player crossing her arms while facing the camera read 22–38°. The gesture rules make the player return to a normal position between two actions, so the value is from just before the gesture.
- `_facing_or_blocked()`: a gesture that does not count is logged as the action `not_facing_<gesture>`, and `not_facing_t` keeps its time for the hint. In the calibration, only the event of the player's current move is checked.
- `body_yaw` is kept for a new round and cleared by `reset()`; logged as `game_p<pid>_body_yaw`.

**Changed:** `src/ipcvgame/scene/renderer.py`, `_draw_not_facing()`: "FACE THE CAMERA" above the player's head (the face's `above_head` anchor, or above the body box) in their team colour for `face_camera_s` after a gesture did not count.

**Changed:** `configs/default.yaml`: `game:` `act_facing_camera`, `act_max_head_yaw_deg` (30°), `act_max_body_yaw_deg` (30°: the shoulders look 13% narrower); `render:` `face_camera_s`, `face_camera_text_scale`.

**Changed:** tests
- `tests/test_game.py`: the made-up frames have no faces, so the fixtures turn the rule off (`NO_FACING`). New tests: no face or a turned head blocks a gesture; the body turn from the shoulders against the torso, only with the arms down; no body check before the calibration; a calibration move counts only facing the camera.
- `tests/test_renderer.py`: the game fixtures turn the rule off; "FACE THE CAMERA" shows after a blocked gesture and goes away.

### Small pose model instead of nano

The arms-crossed rule missed most crosses with the hands on the opposite shoulders, mostly with baggy or dark clothes, because the nano pose model put the wrists in the wrong place (`docs/limitations.md`, "Arms crossed"). With the same rule, the small model found 9 of 10 attempts on `data/clips/shield.mp4` (nano: 3) and 4 real crosses with no false ones on `session.mp4` (nano: 5 real, 2 false). Punches and hands up were the same, and the pose stage still takes ~10 ms per frame on the GPU. Finding the hands with MediaPipe Hand Landmarker was tried first and did not find the hands in most attempts (details in `docs/limitations.md`).

**Changed:** `configs/default.yaml`, `pose.weights`: `models/yolo26s-pose.pt`, with the comparison in its comment. `README.md`: the download command. `docs/phase-1/status.md`: the pose rows and the pose model decision.

## Face effects: team colour, hit reaction, status above the head, cheek bulge

The second feature built from the backlog in `docs/extra_features.md` (feature 1 there after the renumbering of 2026-10-28; before that features 1, 2, 3 and 7; the "face effect extra feature" and "Face effect extra features" commits), by the Task 5 owner on 2026-10-28. It touches Task 1 (`face/`), Task 4 (`interaction/game.py`) and Task 5 (`scene/renderer.py`, `app/pipeline.py`); each step names the files it touched.

### Plan

**What.** Effects on and above each player's face that show who is who and what happens to them: the team colour on the face, a reaction to every hit, a name tag and status icons above the head, and puffed cheeks after eating.

**Decisions (Task 5 owner, 2026-10-28).**

- **Team colour on the face:** the hat's band in the team colour and two stripes of war paint on each cheek. The cheeks are new 3D anchor points on the head model, so they follow tilt and turn like the hat.
- **Hit reaction:** every hit that does damage knocks the hat off, the way the attack went, and the hat pops back after a moment; the face flashes white, then in the attacker's colour. The game reports hits (`Game.hits`); the face effects keep one animation record per player, timed with `Frame.t`.
- **Status above the head:** a name tag "P1"/"P2" in the team colour, a crown for the player with more HP and dizzy stars below 25% HP. The health bars stay pinned to the bottom corners.
- **Dizzy stars:** on an ellipse with a fixed flattening from the config, turned by the head's roll. Following the chin up and down would need the head pitch in `Face`, and `core/types.py` stays as it is.
- **Cheek bulge:** a real image warp (a radial magnifier with `cv2.remap`) for about a second, not a sticker. Its only trigger is eating a consumable: the consumables feature calls `FaceEffects.bulge(pid, t)`. Puffing the cheeks (MediaPipe's `cheekPuff`) does not start it. Until the consumables are built, the bulge does not appear in the game.
- **Logging:** the steps are recorded in this log; no entries in Change Overviews, since no teammate's code moved. The face effects and the hit event are in Task 1 and Task 4 packages; their owners should check these steps.

**Steps.** Each step gets an entry under Changes below.

1. Team colour (Task 1).
2. Hit event (Task 4).
3. Hit reaction (Task 1, wired by Task 5).
4. Status above the head (Task 5).
5. Cheek bulge (Task 1).
6. Docs; webcam check by the Task 5 owner.

### Changes

#### Step 1 · Team colour

**Changed:** `src/ipcvgame/face/head_pose.py` (Task 1), `anchor_points_mm()`: `left_cheek` and `right_cheek` as 3D points on the head, `cheek_out` to each side, `cheek_drop` below the eyes and `cheek_depth_mm` behind them. "Left" and "right" are the image sides, like `head_model_mm`'s eyes.

**Changed:** `src/ipcvgame/face/tracker.py` (Task 1)
- `geometry()` takes the face config and adds the cheeks as 2D offsets, for frames without a head pose.
- `_smoothed_geometry()` smooths every anchor it is given, in one One Euro filter, so a new anchor needs no change there.

**Changed:** `src/ipcvgame/face/effects.py` (Task 1)
- `band_mask()`: the band pixels of `hat.png`, by hue (measured around the hue circle, so 178 counts as red), saturation and brightness. Without the brightness limit, dark reddish pixels along the black outline were taken too.
- `recolor()` turns those pixels to the team colour's hue and keeps their shading. `FaceEffects.team_hat(pid)` makes each team's hat on first use.
- `war_paint()`: two stripes on each cheek anchor, along the eye line, shortened by cos(yaw) on the cheek turned away. `fill_polys()` fills them anti-aliased, blended only in their box.

**Changed:** `configs/default.yaml`, `face:` section: `cheek_out`, `cheek_drop`, `cheek_depth_mm`, `hat_band_hue`, `hat_band_hue_tol`, `hat_band_min_sat`, `hat_band_min_val`, `warpaint_len`, `warpaint_thick`, `warpaint_gap`, `warpaint_opacity`.

**Changed:** `tests/test_face.py`: the cheeks below the eyes on both sides, the 3D cheeks matching the 2D ones when facing the camera, a turned-away cheek moving towards the nose. `tests/test_effects.py`: only the band changes colour, red across 0/180, the stripes' position, roll, squash and colour.

#### Step 2 · Hit event

**Changed:** `src/ipcvgame/interaction/game.py` (Task 4)
- `Hit` dataclass: `target_id`, `owner_id`, `direction` (the way the bolt or beam travelled, +1 to the right, −1 to the left) and `damage`.
- `_apply_damage()` takes the direction from both callers and adds a `Hit` to `Game.hits`, and the action `hit` for the target (`game_p<pid>_actions`, `--until game`). Only hits that do damage count: a shield block, a hit on a player already at 0 or after the win adds nothing.
- `Game.hits` is cleared at the start of `update()` and by `_clear_round()`.

**Changed:** `tests/test_game.py`: bolt hits from both sides with their direction, one beam hit, no hit through a shield, a hit lasting one frame.

#### Step 3 · Hit reaction

**Changed:** `src/ipcvgame/face/effects.py` (Task 1)
- `_Anim`: one record per player with the hat knock-off (time, brim point, face width, roll, direction) and the flash (time, attacker).
- `FaceEffects.update(hits, faces, t)` starts the reaction for players with a visible face. Every hit restarts the flash; the hat is knocked off only while it is on the head, so hits in a row never throw two hats.
- `flying_hat()`: the hat stays where it was thrown in the frame, not on the moving head. In face widths: x = direction × vx × t, y = −vy × t + g t² / 2, turning by `hat_spin_deg` per second the same way. It is fully visible for the first half of `hat_fly_s`, then fades. The design's throw of 1.5 face widths/s moved the hat only ~0.3 face widths before it faded on `session.mp4` and looked like a wobble, so it is 3 face widths/s sideways and up, with gravity 14.
- `hat_size()`: the hat back on the head grows with an ease-out over `hat_pop_s`.
- `flash()`: white for the first half of `flash_s`, then the attacker's colour, fading. Drawn as an ellipse polygon (`ellipse()`) at the face centre, over the war paint and mustache and under the hat.
- `draw(canvas, faces, t)`: without `t` (the `--until` debug views, which have no game) it draws the faces as before. `reset(pid)` forgets the reaction.

**Changed:** `src/ipcvgame/scene/renderer.py`: `draw()` passes `game.hits` and `Frame.t` to the face effects. `src/ipcvgame/app/pipeline.py`: the face effects are reset on `Player.reentered` and with key `R`.

**Changed:** `configs/default.yaml`, `face:` section: `hat_fly_s`, `hat_pop_s`, `hat_fly_v`, `hat_gravity`, `hat_spin_deg`, `flash_s`, `flash_axes`, `flash_opacity`.

**Changed:** `tests/test_effects.py`: the throw direction, spin and fade, the hat popping back, no second hat, the flash colours, no reaction without a face, drawing, reset.

#### Step 4 · Status above the head

**Changed:** `src/ipcvgame/scene/renderer.py` (Task 5)
- `_draw_status()`, right after the face effects: the name tag in every phase; the crown and the dizzy stars from the countdown on, when health counts. While a face fades out, its status is drawn on a copy of the frame and blended in with `alpha`.
- `crowned()`: the round player with strictly the most HP, or None at a draw or with one player.
- `_draw_tag()`: "P<pid>" upright at `above_head`, sized by the face width with a minimum, and the crown (`_draw_crown()`) above it. "FACE THE CAMERA" takes the place of both while it is shown (`_hint_shown()`).
- With a head near the top of the frame, there is no room above it under the top bar; on `session.mp4` the tags and crown were hidden there. Then the tag goes beside the hat, on the side towards the middle of the frame.
- Steady tags (`_Tag` per player, `Renderer.reset(pid)`). The first version jittered: on `session.mp4` the tag jumped between above the head and beside the hat 26–45 times and changed size on 73–94% of frames.
  - The tag goes back above the head only with `tag_side_hyst` tag heights of room to spare. Beside the hat, it changes sides only `tag_side_flip` of the frame width past the middle.
  - The text size changes only past `tag_size_tol`.
  - The position has its own One Euro filter, smoother than the face's, because `above_head` is far from the eyes and shakes most.
  - On the clip, the jumps between above and beside went from 26/45 to 5/3, the size changes from 1789/1350 to 36/1, and the movement with the head still from 2.0 to 0.6 px per frame (P1).
- `_draw_stars()`: `stars_n` stars ring the hat (above `above_head` there would be the tag), turning `stars_hz` times a second; the ones behind the head are drawn first and smaller. `star()` gives the five-pointed polygon.

**Changed:** `src/ipcvgame/app/pipeline.py`: the renderer is reset with the face effects.

**Changed:** `configs/default.yaml`, `render:` section: `tag_face_h`, `tag_min_h`, `tag_side_fw`, `tag_side_hyst`, `tag_side_flip`, `tag_size_tol`, `tag_filter_min_cutoff`, `tag_filter_beta`, `crown_per_tag`, `dizzy_hp`, `stars_n`, `stars_hz`, `stars_rise`, `stars_orbit`, `stars_flat`, `star_size`.

**Changed:** `tests/test_renderer.py`: tags in every phase, the hint replacing the tag, the crown's rules, stars only at low HP and once health counts, stars moving, the fade, the star shape, the tag beside the hat and back, its side, its size tolerance, its smoothing and reset.

#### Step 5 · Cheek bulge

**Changed:** `src/ipcvgame/face/effects.py` (Task 1)
- `bulge_map()`: the `cv2.remap` maps. An output pixel at distance ρ < r from the cheek anchor takes the source pixel at ρ' = ρ (ρ/r)^e, with e = k (1 − u²) and u = ρ/r. Since ρ' < ρ, the middle is magnified (about 2× within a quarter of the radius at k = 0.5). The design's ρ (ρ/r)^k left a seam at the rim; the (1 − u²) factor makes the warp meet the unwarped image with slope 1.
- `warp_bulge()` remaps only the circle's box, in place.
- `bulge(pid, t)` starts a bulge (the only trigger, for eating); `bulge_strength()` follows half a sine over `bulge_s`. The cheek turned away gets a smaller circle, like the war paint. The warp runs after the war paint, so the paint swells with the cheek, and before the mustache.

**Changed:** `configs/default.yaml`, `face:` section: `bulge_radius`, `bulge_k`, `bulge_s`.

**Changed:** `tests/test_effects.py`: the maps (centre, magnification, rim, k = 0), a warped spot, no bulge from puffing the cheeks, the rise and fall, drawing only while bulging.

#### Step 6 · Docs

**Changed:** `docs/extra_features.md`: the code map rows for the face effects and the renderer, the cheek anchors in the face table; the feature moved here and the backlog renumbered (final smash 1, consumables 2, …), with the consumables' eating rule calling `FaceEffects.bulge()`. Code comments, config and tests that said "feature 1" now name this section; references to later features use the new numbers. `docs/phase-1/status.md`: the face effects in Task 1's list and the new feature numbers.

## Consumables: drinks caught with the mouth

The third feature built from the backlog in `docs/extra_features.md` (feature 2 there after the renumbering of 2026-10-28, feature 3 before that and feature 6 originally; the "extra features consumables", "Extra feature consumable", "Extra features consumables:", "Consumable extra_features", "consumables:" and "Consumables" commits), by the Task 5 owner on 2026-10-28. It touches Task 4 (`interaction/game.py`), Task 1 (`face/effects.py`) and Task 5 (`scene/renderer.py`, `scene/draw.py`, `README.md`); each step names the files it touched.

### Plan

**What.** Drinks fall from the sky during the round. A player drinks one by catching it with their open mouth and closing it. Each drink gives one of three buffs: HP, attack or defense. Drinks fall in random sizes, and a bigger drink gives a bigger buff.

**Decisions (Task 5 owner, 2026-10-28).**

- **Only with the mouth:** a drink that touches an open mouth sticks to it, and closing the mouth drinks it. There is no grabbing with the hands and no drinks on the floor. The first design allowed the mouth or a hand; it was changed before step 1.
- **Categories:** each buff has its own three drinks from `assets/Fancy Drinks/` in a colour away from the team colours (HP green, attack orange, defense purple), with an icon on a badge (a plus, a lightning bolt, a shield).
- **Size and value:** each drink gets a random size factor (0.7–1.4), and its value is linear in it, from the category's smallest to its largest value.
- **Buffs:** HP is added at once, up to the maximum. Attack and defense are timed multipliers (10 s): damage = base × the attacker's attack × (1 − the target's defense), rounded. A drink of a running buff's category replaces its value and restarts its time.
- **Cheeks:** drinking starts the cheek bulge (`FaceEffects.bulge()`, from "Face effects"), its only trigger. While a drink is at the mouth, the cheeks already bulge gently (added after step 4).
- **One player:** everything can be tried alone (`--solo`, or `--no-intro` with one player in view): drinks fall, can be drunk, and their buffs show by the health bar and in the log. Only the buffs' effect on damage needs an opponent; tests cover it. Every step has a test with one player.
- **Logging:** the steps are recorded in this log; no entries in Change Overviews, since no teammate's code moved. The drink rules are in Task 4's package and the held-drink bulge in Task 1's; their owners should check these steps.

**Steps.** Each step gets an entry under Changes below.

1. Falling drinks (Task 4).
2. Drinking with the mouth (Task 4, from Task 1's face data).
3. Buffs (Task 4).
4. Look (Task 5).
5. Fixes from the webcam check: cheeks while holding, the hold time.
6. Docs; webcam check by the Task 5 owner.

### Changes

#### Step 1 · Falling drinks

**Changed:** `src/ipcvgame/interaction/game.py` (Task 4)
- `Drink`: `x`, `y` (centre, px), `kind`, `size`, `value` and `variant` (0–1). With `variant`, the renderer picks one of the category's sprites, so the game needs no sprite list. `DRINK_KINDS`: `hp`, `attack`, `defense`.
- `_advance_drinks()` runs inside `advance()`, so only while playing and not after a win. It counts `_drink_wait` down and spawns a drink when it is due and fewer than `drink_max` are in the air; when it is full, the next drink comes as soon as there is room. Drinks fall at a constant speed (one frame height in `drink_fall_s`) and are gone below the frame.
- `_spawn_drink()`: a random category and size; the value is linear in the size; a random x between the `drink_margin_w` margins; the drink starts just above the frame. `drink_height()` gives its height on screen.
- The random generator is seeded from `drink_seed` (null: different every game). `_clear_round()` clears the drinks and draws a new wait.
- Logged as `game_drinks`. In `--until game` (`draw_debug()`): a circle of the drink's size in the category colour, with its letter and value.

**Changed:** `configs/default.yaml`, `game:` section: `drink_every_s`, `drink_max`, `drink_fall_s`, `drink_h`, `drink_size`, `drink_margin_w`, `drink_hp`, `drink_attack`, `drink_defense`, `drink_buff_s`, `drink_seed`. The buff values and time are used from step 3 on.

**Added:** `assets/Fancy Drinks/` (the sprite pack, with its Read Me) and `assets/Jester/`.

**Changed:** `tests/test_game.py`: drinks one by one up to the maximum, the start above the frame and the constant speed, gone below the frame, bigger drinks give bigger values, drinks only while playing and cleared by a new round, the same seed gives the same drinks, drinks for a single player (the solo intro to playing, and `--no-intro` with only P1).

#### Step 2 · Drinking with the mouth

**Changed:** `src/ipcvgame/interaction/game.py` (Task 4)
- `_track_mouth()` runs every frame for every player. The mouth opens when `jawOpen` is above `drink_mouth_open` (0.5) and closes below `drink_mouth_closed` (0.25). Talking on `session.mp4` and `shield.mp4` stayed under 0.5 (at most 0.44). A frame without a score (MediaPipe runs for each player in turn) keeps the state; a new face starts closed, and no visible face means closed.
- `_drink_with_mouths()` runs only while playing, before `advance()`.
  - A drink touches the open mouth when its centre is within `drink_catch_fw` face widths of the mouth anchor plus half its height. Then it is held (`Drink.held_by`, `held_t`, action `catch`), it does not fall, and its rim (`drink_hold_rim` of its height above the centre) sits at the mouth.
  - Each player holds one drink at a time, and the nearer mouth wins.
  - Closing the mouth drinks it: `Drunk(pid, kind, value)` in `Game.drunk` for this frame, and the action `drink_<kind>`.
  - Held longer than `drink_hold_s`, or with the face lost, the drink falls on, and that player cannot catch it again (`Drink.dropped_by`), so holding the mouth open does not keep a drink.
- `reset(pid)` (re-entry) forgets the mouth and lets a held drink fall.
- In `--until game`: each mouth's catch circle, filled while open; a held drink has a thick outline. Logged as `game_p<pid>_mouth_open` and `_holding` (the held drink's kind).

**Changed:** `src/ipcvgame/scene/renderer.py` (Task 5): `draw()` calls `FaceEffects.bulge()` for each drink in `game.drunk`.

**Changed:** `configs/default.yaml`, `game:` section: `drink_mouth_open`, `drink_mouth_closed`, `drink_catch_fw`, `drink_hold_rim`, `drink_hold_s` (1 s at first; see step 5).

**Changed:** `tests/test_game.py`: the mouth's hysteresis; a drink caught by the open mouth and drunk on closing; talking away from a drink drinks nothing; a drink held too long falls on and cannot be caught again; the nearer mouth wins and one drink per mouth; a lost face drops the drink; one player can drink; no drinking outside playing. `tests/test_renderer.py`: drinking puffs the cheeks.

#### Step 3 · Buffs

**Changed:** `src/ipcvgame/interaction/game.py` (Task 4)
- `GamePlayer.attack` / `attack_left` and `defense` / `defense_left`: 1 and 0 without a buff.
- `_apply_drink()` runs as a drink is drunk. HP adds the value, rounded, up to `max_health`; attack and defense take the value and `drink_buff_s`, replacing a running buff of the same kind.
- `_tick_buffs()` counts the buffs down in `advance()`, so only while playing; an ended buff goes back to no effect.
- `_apply_damage()`: damage = base × the attacker's `attack` × (1 − the target's `defense`), rounded, for bolts and beams alike. `Hit.damage` is the damage after the buffs.
- `_refresh()` clears the buffs (new round, reset).
- Logged as `game_p<pid>_attack`, `_attack_left`, `_defense`, `_defense_left`; `--until game` adds "atk x1.42 7s" and "def 35% 7s" to the player's label.

**Changed:** `configs/default.yaml`: the reasons for `drink_hp`, `drink_attack`, `drink_defense` and `drink_buff_s`, now in use.

**Changed:** `tests/test_game.py`: an HP drink heals up to the maximum; an attack drink multiplies the damage until it ends; a defense drink cuts the damage taken; attack and defense together; the same kind replaces the value and restarts the time; the buffs end with the round; one player drinks a buff and sees it in the log.

#### Step 4 · Look

**Changed:** `src/ipcvgame/scene/draw.py` (Task 5): `DRINK_COLORS`, the one table of buff colours (green, orange, purple) for the badges, icons, text and `--until game`.

**Changed:** `src/ipcvgame/scene/renderer.py` (Task 5)
- `drink_sprites(kind)` loads the config's sprites on first use, cropped to their visible part.
- `_draw_drinks()`: each drink as one of its kind's sprites (`Drink.variant` picks it), scaled to its height with the nearest pixel (pixel art). Without sprites (`drink_sprites: null`) it is a circle. `draw_badge()` puts the buff's icon (`ICONS`: a plus, a lightning bolt, a shield) on a light disc at the drink's upper right.
- Drinks are drawn after the face effects, so a drink held at the mouth is in front of the mustache. They are not drawn on the result screen.
- `_draw_popups()`: "+18 HP", "ATTACK x1.4" or "DEFENSE 35%" in the buff's colour, starting at the mouth and offset `popup_side_fw` face widths towards the middle of the frame, so it does not cover the face. It rises `popup_rise_h` over `popup_s`, fades over the last third and stays under the top bar.
- `_draw_buffs()`: above each health bar (P1 from the left, P2 from the right), each running attack or defense buff as its badge, its value and a time bar, fading over the last `buff_fade_s`. `Game.drink_buff_s` gives the full time.
- `faded()` blends a box drawn in a `with` block at an opacity.

**Changed:** `src/ipcvgame/interaction/game.py` (Task 4): `Game.drink_buff_s` for the renderer; `--until game` takes its colours from `DRINK_COLORS`.

**Changed:** `configs/default.yaml`, `render:` section: `drink_sprites` (three drinks per kind: HP Lime Lemonade, Cucumber Water, Power Up; attack Orange Juice, Tropic Cider, Poolside; defense Midnight, Pomegranate, Neon Night), `drink_badge`, `drink_badge_min_px`, `popup_s`, `popup_rise_h`, `popup_side_fw`, `popup_text_scale`, `buff_icon_h`, `buff_fade_s`.

**Changed:** `README.md`: a credit row for the drinks in "Third-party licences" ("Fancy Drinks || Pixel Asset Pack" by Pop Shop Packs; use in game projects allowed, no redistribution as assets, credit optional).

**Changed:** `tests/test_renderer.py`: the sprites per kind, cropped and optional; a falling drink with its badge; the pick-up text rising and ending; the buff icons above the health bars while running, and their fade; one player sees the buff icon; `faded()`.

#### Step 5 · Fixes from the webcam check: cheeks while holding, the hold time

**Changed:** `src/ipcvgame/face/effects.py` (Task 1)
- `FaceEffects.update(hits, faces, t, holding)`: the players holding a drink. Their cheeks bulge gently (`_Anim.hold_level` towards `bulge_hold_k`, half the drinking puff), reached and let go over `bulge_hold_ramp_s`.
- `bulge()` starts the puff from the current strength (`_Anim.bulge_k0`): it rises from there to `bulge_k` over the first half of `bulge_s` and falls as a half sine over the second half, so after a held drink it does not drop to 0 first. `bulge_strength()` is the larger of the hold level and the puff.

**Changed:** `src/ipcvgame/scene/renderer.py` (Task 5)
- `draw()` passes the players holding a drink to `FaceEffects.update()`.
- A drink at a mouth is drawn at most `drink_held_fw` face widths tall, its top `drink_held_drop_fw` below the mouth anchor (the upper lip), without its badge. At ~2.3 m a falling drink is taller than a face, and at full size it covered the cheeks.

**Changed:** `src/ipcvgame/interaction/game.py` (Task 4), `_drink_with_mouths()`: a held drink checks the face, then the closed mouth, then the time limit, so a mouth closing on the frame the time runs out still drinks.

**Changed:** `configs/default.yaml`: `face:` `bulge_hold_k`, `bulge_hold_ramp_s`; `render:` `drink_held_fw`, `drink_held_drop_fw`; `game:` `drink_hold_s` from 1 to 2.5 s. In the first webcam run (`logs/run-20261028-231534.csv`), 3 of 4 caught drinks stuck to the mouth and then fell on: the mouth counted as closed 0.87, 1.00, 1.14 and 2.07 s after the catches, and one closed on the very frame the time ran out.

**Changed:** `tests/test_effects.py`: the cheeks bulge gently while a drink is held; the drinking puff rises from the held bulge without a dip. `tests/test_renderer.py`: a held drink bulges the cheeks; the held drink at the lower lip, face-sized, without a badge. `tests/test_game.py`: a mouth closing as the time runs out still drinks.

#### Step 6 · Docs

**Changed:** `docs/extra_features.md`: the code map rows for the game and the renderer; the feature moved here and the backlog renumbered (final smash 1, aiming 2, paint splash 3, both attacking 4, dodge 5); follow-ups for the webcam check of the drinks, the shield's render time and the asset pack in the repo. Code comments, config and tests that said "feature 2" now name this section; references to later features use the new numbers. `docs/phase-1/status.md`: the drinks in the overview table and Task 4's list, the held-drink bulge in Task 1's, and the new feature numbers.
