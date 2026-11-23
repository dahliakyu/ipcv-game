# IPCV webcam multiplayer game (University of Twente, IPCV 2026-1A)

Group 4: Colour Clash, a two-player paint fight on one webcam. Five tasks: face
tracking and AR, pose and motion, multiplayer identity, interaction and game
control, scene processing and real-time integration. The game is final; the
current work is evaluation, the technical report and the poster.

- Project brief: `docs/IPCV_2026-1A_Project_Description_v1-1.pdf`
- How to play, setup, file structure: `README.md`
- Every change and its evidence: `docs/merge_log.md`; limitations: `docs/project_report.md`, section 7
- Evaluation: `README.md`, "Evaluation"; scripts `tools/eval_*.py`, results `results/`,
  report `docs/project_report.md`

## Ownership

Each task has one owner, and each owner must be able to explain their own code.

| Task | Packages | Owner |
| --- | --- | --- |
| 1 Face + AR | `face/` | Eline Kanters |
| 2 Pose + motion | `pose/` | Riyasdeen Samsudeen |
| 3 Identity | `identity/` | Lizzy-Milou de Bruijn |
| 4 Interaction | `interaction/` | Pjotr van der Made |
| 5 Integration | `core/`, `io/`, `scene/`, `app/`, `tools/` (shared infrastructure) | Ellen Zheng |

- Work only in the packages of the task the current user owns. If it is not
  clear which task that is, ask before editing.
- In another task's packages, only build stubs or interfaces unless that
  task's owner asks for more.
- Each owner delivers their module **and** its evaluation: the script, the
  clips, the report section and the poster box. The integration owner keeps
  the contracts, the pipeline and the overall frame rate.
- Owners must understand every line in their module: explain design choices
  and trade-offs, and prefer explainable methods (pinhole model, EMA,
  Hungarian matching, rule state machines) over black boxes; use learned
  models where they are clearly better, and know their failure modes.

## Rules

- `core/types.py` is the contract between modules. Don't change it without a
  heads-up to the whole team.
- Every threshold goes in `configs/default.yaml` (in the module's own section)
  with a comment on where the value comes from (a measurement, a sweep, a
  source); never hard-code one.
- Coordinates are pixels in the mirrored full-res frame; filters use `Frame.t`
  (capture time), not frame counts. Index keypoints by name via
  `core/skeleton.py`.
- Keep the module interface: `__init__(cfg)`, `update(...)` (pose:
  `detect(frame)`), `reset(pid=None)`, `draw_debug(canvas, state)`, optional
  `log_fields(state)`. Reset per-player state when `Player.reentered` is True.
- Reuse `core/filters.py`, `core/geometry.py` and `scene/draw.py` instead of
  writing new copies.
- New dependencies go in the pinned `requirements.txt` with a comment.
- Branch per task; merge to `main` only through a PR reviewed by Task 5.
- Never open the webcam without asking; develop and test on recorded clips.
- No claim without evidence: a number in the report or the docs comes from a
  script run on a log, and a proposed fix is a hypothesis until a clip shows it.
  Record every change, why, and which clip showed it in `docs/merge_log.md`.

## Environment and commands

Ubuntu 24.04 LTS, conda env `ipcvgame` (Python 3.12); see README.md for setup.
Measured on a desktop with an RTX 3070 Ti; the demo laptop (HP OMEN 15,
Ryzen 7 4800H) runs the game on its CPU, so keep the CPU-only path working
(`--set pose.device=cpu --set identity.reid_device=cpu`).

```bash
pytest                                                    # before every PR
ruff check src tests tools
python tools/run_module.py --until <stage> --source data/clips/<clip>.mp4
python -m ipcvgame.app.main --headless --max-frames 100   # quick smoke run
python tools/eval_<face|motion|identity|gestures|pipeline>.py <clips or logs>
python tools/label_clip.py <clip> --mode identity|gestures
```

## Recording test clips

- Record through the same capture path as the game (`tools/record_clip.py` or
  `--record`); keep the `<clip>.timestamps.csv` sidecar named after its clip.
  Check the camera's real frame rate: in dim light it drops (15 → 8.5 fps).
- Name clips `<scenario>_<distance>_<light>_<take>.mp4` and keep a scenario
  sheet while filming: who, distance from the lens (tape-measured), light, what
  was done on which cue. Half of the ground truth is free if written down then.
- Film each scenario with every pair of subjects available, with repetitions:
  standing still and moving at 1, 2, 3 m; hand over face, dim and bright;
  10+ crossings (metronome); leave and re-enter, newcomer, bystander in view;
  each gesture 10+ times per player on cue; transitional moves (arms raised one
  after the other, walking) as negatives; free gameplay.
- Keep tuning clips and test clips apart; report test-clip numbers.
- Raw footage of people is not submitted; submit the per-frame logs
  (`results/logs/`), tables and labels instead.

## Image processing and computer vision, per task

### Image formation (all tasks)

- Pinhole model x = f·X/Z; depth from a known size Z = f·S/s (shoulder width,
  torso length). Wrong f or S is a scale error; a turning body shrinks the
  shoulders, not the torso. f = (w/2)/tan(hfov/2), better from a checkerboard
  calibration (`cv2.calibrateCamera`) of the camera actually used.
- Mirror once at capture. Models label sides as in a photo: on the mirrored
  frame `L_WRIST` is the player's own right wrist. Rules that only need
  "opposite order" stay mirror-independent.
- Low light: longer exposure (motion blur, lower frame rate) and more gain
  (noise). `--light 0.3 --noise 8` isolates the light from everything else; a
  real dim clip shows the camera's own reaction.
- Compare clothing in HSV or Lab, not RGB; lightness alone can separate black
  from brown when hue drifts with white balance.

### Task 1: face tracking and AR

- Detect faces (YuNet) inside each player's head crop from the pose keypoints:
  face ↔ player by construction. Small faces fail first (3 m, ~110 px crops):
  upscale small crops; bridge misses with Lucas–Kanade optical flow for a
  limited time, then fade the effect.
- Head pose from landmarks with `cv2.solvePnP` and a canonical 3D face; roll
  from the eye line, scale from the inter-ocular distance; sprites with
  `cv2.warpAffine` and alpha blending.
- Smooth anchors (not all landmarks) with a One Euro filter: cut-off =
  min_cutoff + β·speed.
- Metrics: detection rate per distance and light; jitter as the RMS second
  difference ‖p₊₁ − 2p + p₋₁‖ (steady motion cancels out); alignment error
  between the drawn anchor and the raw detection; stage time.

### Task 2: pose and motion

- YOLO-pose once per frame for all people; input size trades speed against
  small, far players; FP16 on a GPU.
- Confidence gate; hold a missing keypoint briefly, then mark it invalid and
  snap back when it returns. Normalise distances and speeds by shoulder width.
- Filters on real time steps: EMA weight a = 1 − e^(−dt/τ), steady-motion lag
  dt(1−a)/a (it barely smooths at low frame rates); One Euro adapts to speed
  (the game's filter); a constant-velocity Kalman filter predicts through
  dropouts.
- Metrics: detection rate per keypoint group; still and moving jitter on the
  same frames for every filter compared; filter delay as the time shift that
  best aligns smoothed and raw tracks; bridged dropouts. Keypoint error needs
  hand-marked keypoints.

### Task 3: multiplayer identity

- Association: IoU or keypoint-similarity cost, Hungarian algorithm; Kalman
  prediction through short occlusions; track birth after a few frames, death
  after a buffer.
- Re-identification with appearance embeddings (OSNet) or colour histograms
  (Bhattacharyya); update a stored appearance only from clean, unoverlapped
  crops. Sweep the similarity threshold: too low links new tracks to the wrong
  player, too high leaves returning players without an id.
- Plan for a third person: a newcomer in a leaving player's spot can inherit
  the track; a bystander can take a free slot.
- Metrics: identity accuracy on labelled frames, ID switches, crossings kept,
  re-entries recovered and recovery time (IDF1); depth error against measured
  distances.

### Task 4: gestures and game control

- Rules on normalised geometry (joint angles, reach in shoulder widths, zones
  relative to the body) with hysteresis: confirm time, release time,
  refractory period, one action at a time.
- Single-frame rules fire on transitional poses (an arm swinging up passes
  through a punch); check temporal features (direction and size of the
  wrist's movement over the last 0.2 s) against labelled negatives.
- Metrics: label the start of every intended action; precision, recall,
  confusion between gestures, latency from movement start to the event and to
  the screen. Latency = movement + confirm time + filter delay + frame time +
  camera.

### Task 5: scene and real-time integration

- Single-threaded pipeline of typed stages, a guard per stage with a stub
  fallback, a per-stage profiler and per-frame logs.
- Report mean, 95th percentile and slowest frames (FPS "5 % low"), freezes,
  failures, start-up time, CPU/GPU/memory, on the demo machine in the demo
  configuration.
- Speed-ups, measured before and after and checked for accuracy loss: smaller
  input size, FP16, export (ONNX / OpenVINO / TensorRT), capture on a
  background thread keeping only the newest frame.

## Evaluation and reporting

- Expected outcome → metrics → protocol → clips. One script per task that
  accepts clips and logs and prints Markdown tables; its metric maths
  unit-tested on synthetic data with known answers.
- Ground truth: identity labels every 0.5 s (player A, B, someone else),
  action start frames, measured distances. Labels describe the people, not the
  tracker, so one label file scores every run.
- Report per condition (distance, light, motion, subjects) with the number of
  trials; median and 95th percentile for timings. Failure cases are results:
  show the frames, name the cause, propose one measured improvement.
- Labels, analyses and text made with AI are declared in
  `AI-and-external-tools.md`, with how they were checked.

## Definition of done, per module

- [ ] Code behind the shared interface; thresholds in the config with their source.
- [ ] Unit tests; the full pipeline runs with the module on recorded clips.
- [ ] Evaluation script and tables on the test clips, per condition.
- [ ] Measured on the demo machine in the demo configuration.
- [ ] Failure cases with frames; a limitation and one measured improvement.
- [ ] Report section, poster box and AI declaration by the owner.
