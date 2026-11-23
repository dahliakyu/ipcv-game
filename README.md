# Colour Clash: a webcam paint battle for two players

University of Twente, Image Processing and Computer Vision (IPCV) 2026-1A, group project. Group: 4.

Two players stand in front of one webcam and fight with their bodies. A punch throws a ball of paint at the other player; crossing the arms raises a shield; drinks fall from the top of the frame and are caught with the mouth for health, attack or defense; an orb drifts through the room, and the player who grabs it glows and can fire a magic beam with both hands up. Every hit leaves paint on the target's body, which moves with them, and paint that flies past lands on the wall behind them. The first player to run out of health loses.

## Tasks and owners

| Task (project brief §3) | What it does in the game | Code | Owner |
| --- | --- | --- | --- |
| 1 Face tracking and augmented effects | Finds each player's face (YuNet on a head crop from the pose, MediaPipe Face Landmarker for head rotation and expressions); draws a team hat, war paint and a mustache that follow the head; the hat flies off on a hit; cheeks bulge when drinking; the mouth opening catches drinks | `src/ipcvgame/face/` | Eline Kanters (s3197581) |
| 2 Body pose estimation and motion tracking | YOLO26s-pose: 17 keypoints for every person in one pass; smoothing, short dropouts held, velocities; body measurements for the calibration | `src/ipcvgame/pose/` | Riyasdeen Samsudeen (s3941981) |
| 3 Multiplayer detection and identity tracking | Stable Player 1 and Player 2 through crossing, occlusion and re-entry (BoT-SORT on the pose boxes, OSNet re-identification); position in metres | `src/ipcvgame/identity/` | Lizzy-Milou de Bruijn (s3065642) |
| 4 Player interaction and game control | Rule-based gestures (punch, hands up, arms crossed) and the game rules: attacks, shield, damage, drinks, the orb, rounds | `src/ipcvgame/interaction/` | Pjotr van der Made (s3207447) |
| 5 Scene processing and real-time integration | The pipeline and its failure handling, webcam and clip input, logging; the game view: pixel look, paint on the bodies, the room reacting to the fight (wall paint, light, shake), sprites, health bars and screens | `src/ipcvgame/core/`, `io/`, `scene/`, `app/`, `tools/` | Ellen Zheng (s3502155) |

## How to play

1. **Start.** Both players stand in their start place (P1 left, P2 right) until the game sees them. Each player then learns the moves one at a time while the game measures them: standing still with the arms down (shoulder width and arm length), a punch, hands up and crossed arms. Stand about 2–2.5 m from the camera and face it. A 3-2-1 countdown starts the round.
2. **Fight.** Gestures count only while facing the camera, and after each one return to a normal position (arms down) before the next.

   | Move | Effect |
   | --- | --- |
   | Punch: one straight arm out to the side, the other bent | Throws a paint blob at the other player (10 damage) |
   | Cross the wrists in front of the chest | Shield for 1 s: blocks blobs and the beam |
   | Open the mouth under a falling drink, then close it | Green: +10–30 health; orange: damage dealt ×1.2–1.6; purple: 20–50% less damage taken; the buffs last 10 s |
   | Touch the orb with a hand | You hold the orb: your body glows |
   | Hands up, holding the orb | Magic beam across the frame (12 damage); the orb is used up |

3. **End.** At 0 health the other player wins. Both players holding their hands up for 2 s start the next round.

## Requirements

- Ubuntu 24.04 (developed and tested there), a webcam, and Python 3.12. MediaPipe has no wheels for newer Python versions.
- An NVIDIA GPU with a driver for CUDA 12.8 runs it at full speed (24–33 FPS on an RTX 3070 Ti). Without one the game runs on the CPU at 7.6–11 FPS (11 FPS on a Ryzen 7 5700G desktop, 7.6 FPS live on the demo laptop). The demo laptop, an HP OMEN Laptop 15-en0xxx (AMD Ryzen 7 4800H), runs it on the CPU.
- About 2 × 3 m of free floor in front of the camera.

## Setup

```bash
git clone https://github.com/dahliakyu/ipcv-game.git ipcv-game && cd ipcv-game     # or unpack the ZIP

# conda (used for the demo) ...
conda create -n ipcvgame python=3.12 -y && conda activate ipcvgame
# ... or a plain venv
python3.12 -m venv .venv && source .venv/bin/activate

# PyTorch with CUDA 12.8; on a CPU-only machine replace cu128 with cpu
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"   # True on a GPU machine
```

`environment.yml` does the same in one step: `conda env create -f environment.yml && conda activate ipcvgame && pip install -e .`

### Model weights

Model weights are not in the repository or the ZIP. Download them into `models/` with the exact versions below; the checksums are those of the files the game was tested with.

| File | What for | Licence |
| --- | --- | --- |
| `yolo26s-pose.pt` | Body pose (task 2), Ultralytics assets release v8.4.0 | AGPL-3.0 |
| `face_detection_yunet_2023mar.onnx` | Face detection (task 1), OpenCV model zoo | MIT |
| `face_landmarker.task` | Head rotation, mouth opening (task 1), MediaPipe model version 1 | Apache 2.0 |
| `osnet_x1_0_msmt17_combineall.pth` | Recognising a player who comes back (task 3), OSNet MSMT17 `combineall` from the authors' Hugging Face repo | MIT |
| `selfie_segmenter_landscape.tflite` | Outline of the people, so wall paint stays behind them (task 5), MediaPipe model version 1 | Apache 2.0 |

```bash
mkdir -p models
wget -P models https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt
wget -P models https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
wget -P models https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o models/osnet_x1_0_msmt17_combineall.pth https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
wget -P models https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite

sha256sum models/*
# a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646  models/yolo26s-pose.pt
# 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  models/face_detection_yunet_2023mar.onnx
# 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  models/face_landmarker.task
# 48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727  models/osnet_x1_0_msmt17_combineall.pth
# 490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e  models/selfie_segmenter_landscape.tflite
```

The pose model is also downloaded automatically on the first run if it is missing. Without the Selfie Segmenter the wall paint falls back to a coarser mask built from the keypoints; the other files are required. The OSNet network code is included in `src/ipcvgame/identity/osnet.py`, so `torchreid` is not needed.

## Run

```bash
python -m ipcvgame.app.main                                          # the game, on the webcam
python -m ipcvgame.app.main --solo                                   # one player, to try it alone
python -m ipcvgame.app.main --source data/clips/session.mp4 --no-intro   # a recorded clip, playing at once
```

| Option | Effect |
| --- | --- |
| `--source <index or path>` | Webcam index or video file (default: `source:` in the config) |
| `--no-intro` | Play from the first frame, without start places, calibration and countdown (for clips recorded without the intro) |
| `--solo` | The intro and rounds need only P1; a solo round never ends (keys `r` / `R`) |
| `--until <stage>` | Run up to `pose`, `identity`, `motion`, `face`, `gestures` or `game` and show that stage's debug overlay |
| `--set <key>=<value>` | Override one config value, e.g. `--set render.arena_shake_fw=0` (no shake on hits) |
| `--config <file>` | A YAML file on top of `configs/default.yaml` |
| `--headless`, `--max-frames N` | No window; stop after N frames |
| `--log <file>`, `--no-log` | Per-frame CSV path (default `logs/run-<time>.csv`), or none |
| `--record <clip>`, `--record-view [<clip>]` | Also save the raw input (replayable), or the view as shown |

Keys in the game window: `q` / `Esc` quit, `p` pause, `r` new round with the same players (from the countdown), `R` reset everything (players are identified again, back to the start places), `b` throw a test blob at P1, `n` give P1 the orb (both for trying the game alone).

A module that raises an error does not stop the game: that stage is skipped for the frame and its name is shown at the top right; identity, motion and face switch to a simple stand-in after failing for 1 s (`pipeline:` in the config; `--set pipeline.catch_errors=false` stops at once instead).

## Evaluation

The report's numbers (`docs/project_report.md`) come from one script per task. Each prints Markdown tables per clip and player, and accepts either a clip (the game is run on it first, headless, and its per-frame log saved to `logs/eval/`) or an existing log.

**Reproduce the report from the per-frame logs** (no footage needed; labels are read from `data/labels/`). The logs (`results/logs/`, ~70 MB of CSV and NPZ files) are not in the git repository; they are submitted separately and go into `results/logs/`:

```bash
python tools/eval_face.py      results/logs/task1_face/*.csv
python tools/eval_motion.py    results/logs/task2_motion/final/*.csv          # needs the .kpts.npz next to each log
python results/filter_sweep_summary.py                                         # Task 2 filter comparison
python tools/eval_identity.py  results/logs/task3_identity/*.csv
python tools/eval_gestures.py  results/logs/task4_gestures/final/*.csv
python tools/eval_pipeline.py  results/logs/task5_pipeline/gpu/*.csv
```

| Table in `results/` | Report section |
| --- | --- |
| `task1_face.md` | Task 1 |
| `task2_motion_final.md`, `task2_motion_ema.md`, `task2_filter_sweep.md` | Task 2: the final game (One Euro filter), the EMA on the same clips, the sweep over filter settings |
| `task3_identity.md`, `task3_reid_threshold_sweep.md` | Task 3: tracks and labelled metrics; the re-identification threshold from 0.4 to 0.8 |
| `task4_gestures_final.md`, `task4_gestures_final_phone_and_laptop.md` | Task 4, final game; the phone videos and the laptop's move tests were labelled at a coarser frame step and are scored with `--pre-s 0.4` |
| `task4_gestures_first_run.md`, `task4_gestures_veto_with_oneeuro.md`, `task4_gestures_veto_with_oneeuro_phone_and_laptop.md` | Task 4: the first run (EMA filter) and the rising-wrist veto that was tried and switched off |
| `task5_pipeline_gpu.md`, `task5_pipeline_cpu.md`, `task5_pipeline_laptop_live.md` | Task 5: desktop with GPU, desktop CPU only, the live demo on the laptop |

Logs are in `results/logs/task<N>_<stage>/`: one CSV per run with one row per frame, its `.info.json` (settings, start-up time, resources) and, for Task 2, `.kpts.npz` (every raw and smoothed keypoint). A log is named `<stage>-<clip>[-<tag>].csv`; tags: `final` (the final game), `light0.3-noise8` (synthetic low light), `th0.x` (re-identification threshold), `cpu` (everything on the CPU), `ema` and `oe<min cutoff>-<beta>` (filter sweep), `veto-oe` (gesture runs with the punch veto on). Untagged Task 2 and Task 4 logs are the first run (EMA filter). The Task 1, 3 and 5 runs do not depend on the keypoint filter or the punch veto.

| Clip name in the report | Clip file (log name) | Recorded |
| --- | --- | --- |
| *Still* 1, 2, 3 m | `face_still`, `face_still_2m`, `face_still_3m` | 29 Oct, laptop, 15 fps |
| *Moving* 1, 2, 3 m | `face_move_1m`, `face_move_2m`, `face_move_3m` | same |
| *Hand over face* | `face_occlude_1m_dim`, `face_occlude_1m_mid_to_light`, `face_occlude_2m_mid_to_light`, `face_occlude_3m` | same |
| *Swap* | `face_swap_1m` | same |
| *Crossing ×4*, *Crossing ×22* | `id_cross_bright`, `id_cross_bright_metronome` | same |
| *Re-entry*, *New person* | `id_reentry`, `id_new_person` | same |
| *Depth* 1, 2, 3 m | `id_depth_1m`, `id_depth_2m`, `id_depth_3m_0.5m_side` | same |
| *Punch reps*, *Shield reps*, *Beam reps* | `Shoot_test`, `shield_test`, `Laser_test` | 31 Oct, 30 fps |
| *Phone moves 1*, *Phone moves 2* | `wa_1030`, `wa_1031` (phone videos of 30 and 31 Oct) | phone, 25 fps |
| *Demo game* | `demo`; live log `laptop_live/run-20261029-154414.csv` | 29 Oct, laptop |
| *Laptop game* | `session2` | 31 Oct, laptop |
| *Desktop game*, *Solo shields* | `session`, `shield` | 26 and 28 Oct, desktop webcam |

**Evaluate new clips:**

```bash
python tools/record_clip.py crossing                   # record a raw clip from the webcam (space: start/stop)
python tools/label_clip.py data/clips/crossing.mp4 --mode identity   # click who is who, one frame every 0.5 s
python tools/label_clip.py data/clips/moves.mp4 --mode gestures      # mark the start of every intended action
python tools/eval_identity.py data/clips/crossing.mp4
```

| Option (all scripts) | Meaning |
| --- | --- |
| `--set KEY=VALUE` | Change the config for this run, e.g. `--set identity.reid_threshold=0.5`; repeatable |
| `--tag NAME` | Added to the log name and the table, to compare settings side by side |
| `--light 0.3 --noise 8` | Synthetic low light: pixel values × 0.3 plus Gaussian noise (σ 8 grey levels) |
| `--intro` / `--no-intro` | Run the game intro, or play from the first frame (default: intro for identity only) |
| `--markdown FILE` | Also save the tables |

- **Labels** (`tools/label_clip.py`, saved to `data/labels/<clip>.<mode>.csv`) describe the people, not the tracker, so one label file scores every run of the clip. Identity: left click = player A, right click = player B, middle click = anyone else; a frame with no click means nobody is in view. Gestures: keys `1` punch, `2` arms crossed, `3` hands up, then click on the person; label every intended action, since events without a label count as false triggers. A row `<frame>,<t>,_end,,` limits scoring to the frames before it.
- **What each column means** is in the docstring of each script and the protocol of each task in the report.
- **Clips** are stored un-mirrored with a `<clip>.timestamps.csv` file of the real capture times, so a replay sees the same time steps as the live run; keep that file named after its clip. Clips recorded before 1 Nov say 30 fps in the file while the frames came in slower, so video players show them sped up (the game is not affected): `python tools/retime_clip.py <clips> --apply` fixes the file's frame rate without re-encoding.
- **Raw footage** is not submitted (it shows the group members) and is available on request; the processed data is the per-frame logs (`results/logs/`, submitted separately), the tables (`results/*.md`) and the hand labels (`data/labels/`).
- Every game run also writes `logs/run-<time>.csv`: one row per frame with the time of each stage in ms, each player's status, box and position, and each module's own columns (`face_p1_roll`, `game_p1_health`, …).

## Tests and debugging

```bash
pytest                                                 # unit and pipeline tests; no camera or models needed
ruff check src tests tools                             # lint
python tools/run_module.py --until identity --source data/clips/session.mp4   # one stage's debug view
python -m ipcvgame.app.main --record data/clips/session.mp4   # record while playing
```

## File structure

```
README.md                  this file
requirements.txt           pinned Python dependencies (PyTorch is installed separately, see Setup)
environment.yml            conda environment (Python 3.12 + requirements.txt)
pyproject.toml             package, pytest and ruff settings
configs/default.yaml       every threshold and setting, one section per stage, each with the reason for its value
src/ipcvgame/
  core/                    shared contracts and helpers
    types.py               the data passed between stages (Frame, PoseDetection, Player, Motion, Face, GestureEvent, FrameState)
    skeleton.py            keypoint names (COCO-17)
    filters.py             One Euro, EMA and Kalman filters
    geometry.py            boxes, crops, angles, bilinear mapping
    timing.py, config.py   per-stage profiler; YAML config with overrides
  io/
    source.py              webcam and video file behind one interface (mirrored, real timestamps)
    recorder.py            clip recording and per-frame CSV logs
  pose/                    task 2
    estimator.py           YOLO-pose wrapper
    motion.py              keypoint smoothing, dropout hold, velocities, wrist speeds; jitter logging
    measure.py             shoulder width, arm and torso length
  identity/                task 3
    tracker.py             BoT-SORT tracks to persistent players, re-entry, position in metres
    tracks.py, embedding.py, osnet.py   track helpers; OSNet appearance vectors (network code from torchreid)
  face/                    task 1
    tracker.py             face per player from the head crop, smoothing, anchors for effects
    head_pose.py           3D head pose (solvePnP / MediaPipe) and 3D anchor points
    landmarker.py          MediaPipe Face Landmarker (head rotation, blendshapes)
    effects.py             face effects: hat, war paint, mustache, hit reaction, cheek bulge
  interaction/             task 4
    gestures.py            gesture rules -> events
    game.py                game rules, phases, drinks, orb, damage, paint positions
  scene/                   task 5: how the game looks
    renderer.py            the game view, in drawing order
    arena.py, segment.py   the room reacting to the fight; person mask (Selfie Segmenter)
    paint.py               paint splashes on the bodies, the paint blob
    orb.py                 the orb, its slot and the glow of the player holding it
    fire_ring.py, magic_beam.py, animation.py   sprite effects (shield, beam)
    instructions.py, draw.py   intro figures and texts; colours and drawing helpers
  app/
    pipeline.py            pose -> identity -> motion -> face -> gestures -> game -> render, a guard per stage
    main.py                command line, window and keys
tools/                     record_clip.py, run_module.py, retime_clip.py, label_clip.py, eval_*.py (one per task) and eval_common.py
tests/                     pytest, one file per module
assets/                    sprites (face, drinks, shield, beam; sources below)
docs/
  merge_log.md             every change and design decision, step by step, with the evidence
  project_report.md        the technical evaluation report, with the limitations and future improvements (section 7)
results/                   the evaluation tables behind the report; logs/ (per-frame logs) is submitted separately
data/labels/               hand labels of the clips (who is who, action starts)
  poster_metrics.md        the poster's performance and limitations text
  oral_exam/               study notes per task (design choices, functions, options tested, issues, questions) and general.md (concept, IPCV theory, code)
  IPCV_2026-1A_Project_Description_v1-1.pdf   the project brief
AI-and-external-tools.md   AI declaration, skills, models, libraries and assets
data/clips/, models/, logs/   not in the repository (clips shared separately, weights downloaded, logs written per run)
```

## Dependencies

All Python packages are pinned in `requirements.txt`:

| Package | Version | Used for |
| --- | --- | --- |
| torch, torchvision | 2.11.0, 0.26.0 | YOLO and OSNet inference (CUDA 12.8 build) |
| ultralytics | 8.4.164 | YOLO26-pose and the BoT-SORT tracker |
| mediapipe | 1.0.1 | Face Landmarker, Selfie Segmenter |
| opencv-python, opencv-contrib-python | 5.0.0.93 | Image processing, YuNet, drawing, video I/O (both at the same version) |
| numpy, scipy | 2.5.2, 1.18.1 | Arrays; Hungarian matching |
| filterpy | 1.4.5 | Kalman filters |
| lap | 0.5.13 | BoT-SORT's assignment |
| PyYAML | 6.0.3 | Config |
| pandas, matplotlib | 3.0.6, 3.11.2 | Log analysis for the report |
| psutil, nvidia-ml-py | 7.2.2, 13.615.71 | CPU, memory and GPU use in `tools/eval_pipeline.py` |
| pytest, ruff | 9.1.1, 0.16.9 | Tests, lint |


## Documentation

- `docs/merge_log.md`: what was built, in what order, why, and with which clip evidence.
- `README.md`, "Evaluation": how to run the evaluation scripts.
- `docs/project_report.md`: the technical evaluation report (methods, measured results, conclusions, limitations and future improvements); its tables and logs are in `results/`, the hand labels in `data/labels/`.
- AI and external-tools statement: `AI-and-external-tools.md`.

## Third-party models, code and assets

Pretrained models: Ultralytics YOLO26 (AGPL-3.0), MediaPipe Face Landmarker and Selfie Segmenter (Apache 2.0), YuNet from the OpenCV model zoo (MIT), OSNet by Kaiyang Zhou (MIT, code and weights). All are used for coursework and listed in the AI and external-tools statement.

| Assets | Source | Licence |
| --- | --- | --- |
| `assets/magic_effect/` (the beam's ball and band) | "Magic Attack" by NYKNCK, itch.io, https://nyknck.itch.io/magic-attack (purchased) | None stated beyond the purchase; used with attribution for coursework only |
| `assets/shield/FX019/` (shield ring of fire), `assets/shield/FX017/` (unused) | Pixel effect packs FX019 and FX017 by NYKNCK, itch.io, https://nyknck.itch.io (purchased) | None stated beyond the purchase; used with attribution for coursework only |
| `assets/beam/` (fire blast, unused since the paint blob) | "Fire Attack FX - Pixel Art Effect - FX061" by NYKNCK, itch.io, https://nyknck.itch.io/fx061 (purchased) | None stated beyond the purchase; used with attribution for coursework only |
| `assets/Fancy Drinks/Sprites/individual_sprites/` (the drinks) | "Fancy Drinks \|\| Pixel Asset Pack" by Pop Shop Packs, itch.io, https://pop-shop-packs.itch.io/fancy-drinks (paid) | Use in any game project allowed; reselling or redistributing the files as assets is not allowed |
| `assets/face/` (hat, mustache) | From the group's face-tracking script. Mustache: FlyClipart, https://flyclipart.com/es/mostacho-blend-s-pixel-art-gray-world-of-warcraft-hd-png-download-1153372; hat: a Pinterest board, https://id.pinterest.com/tahaharnounes360/ | None stated; used with attribution for coursework only |
