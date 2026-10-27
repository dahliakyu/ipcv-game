# IPCV webcam multiplayer game

University of Twente, IPCV 2026-1A. Architecture, data contracts and plan:
[docs/phase-1/setup-guide.md](docs/phase-1/setup-guide.md).
Current status and per-task to-do lists: [docs/phase-1/status.md](docs/phase-1/status.md).

## Setup (Ubuntu 24.04, Python 3.12)

Python 3.12 is required: MediaPipe does not ship wheels for newer versions yet.

```bash
# conda (what the demo laptop uses)
conda create -n ipcvgame python=3.12 -y && conda activate ipcvgame
# ...or a plain venv
python3.12 -m venv .venv && source .venv/bin/activate

# GPU (CUDA 12.8). CPU-only: replace cu128 with cpu.
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
```

Model weights are not in the repo. `models/yolo26n-pose.pt` is downloaded
automatically on first run; to fetch it by hand:

```bash
mkdir -p models && wget -P models https://github.com/ultralytics/assets/releases/latest/download/yolo26n-pose.pt
```

The face detector (YuNet, OpenCV model zoo) is not downloaded automatically:

```bash
wget -P models https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
sha256sum models/face_detection_yunet_2023mar.onnx   # 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

The MediaPipe Face Landmarker (head rotation, mouth opening, blinks) is not
downloaded automatically either:

```bash
wget -P models https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
sha256sum models/face_landmarker.task   # 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

The OSNet person re-identification weights (Task 3: recognising a player who
comes back) are not downloaded automatically either. They are the MSMT17
`combineall` model from the authors' Hugging Face repo
([kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet)), saved under a
shorter name:

```bash
curl -L -o models/osnet_x1_0_msmt17_combineall.pth https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
sha256sum models/osnet_x1_0_msmt17_combineall.pth   # 48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727
```

The network code is copied into `identity/osnet.py`, so `torchreid` is not
needed (the Hugging Face page's usage example installs it).

Test clips go in `data/clips/` (gitignored, shared via the drive). Copy each
`.mp4` together with its `.timestamps.csv`.

## Run

```bash
python -m ipcvgame.app.main                                   # webcam (configs/default.yaml)
python -m ipcvgame.app.main --source data/clips/crossing.mp4  # recorded clip
python tools/run_module.py --until identity --source data/clips/crossing.mp4
python -m ipcvgame.app.main --set pose.det_conf=0.5           # config override
```

Keys: `q`/`Esc` quit, `p` pause, `r` reset module state.
`--until {pose,identity,motion,face,gestures,game}` runs the pipeline up to that
module and shows only its `draw_debug` overlay. `--headless --max-frames N`
runs without a window.

## Record test clips and logs

```bash
python tools/record_clip.py crossing          # space = start/stop -> data/clips/crossing.mp4
python -m ipcvgame.app.main --record data/clips/session.mp4   # record while the game runs
python -m ipcvgame.app.main --until game --record data/clips/session.mp4 --record-view
                                              # + the shown view -> data/clips/session_view.mp4
```

`--record-view [MP4]` saves the window's view (effects, debug overlays,
timings) as shown, mirrored; it is for watching and the report, not for
replaying through the pipeline. The debug views from `--until face` on draw
the face effects under their overlay (`render.debug_face_effects`).

Clips are stored raw (un-mirrored) with a `<clip>.timestamps.csv` sidecar of
real capture times; `VideoFile` replays those times so filters see the true dt.
Every run also writes `logs/run-<time>.csv` (frame_id, t, per-stage ms,
per-player status/pos/box; `--no-log` to skip). To add your own columns, give
your module a `log_fields(state) -> dict` method with keys named
`<stage>_p<pid>_<field>` (e.g. `face_p1_roll`).

## Layout

```
configs/default.yaml     every threshold, one section per module
src/ipcvgame/
  core/                  Task 5: types (contract), skeleton, filters, geometry, timing, config
  io/                    Task 5: source (webcam / video file), recorder (clips, CSV logs)
  pose/                  Task 2: estimator (YOLO-pose), motion (smoothing, signals)
  identity/              Task 3: tracker
  face/                  Task 1: tracker, effects
  interaction/           Task 4: gestures, game
  scene/                 Task 5: renderer, shared drawing helpers
  app/                   Task 5: pipeline, main
tools/                   run_module.py, record_clip.py, eval_*.py (per task)
tests/                   pytest; no model or camera needed
assets/                  sprites and other media the game draws (e.g. assets/face/hat.png)
data/clips/, models/, logs/   gitignored
```

## Naming conventions

Full version with examples: [docs/phase-1/status.md](docs/phase-1/status.md#naming-conventions).

- **Stages** The code names each pipeline step by what it
  does, and the same name is used for `--until`, the config section and the
  CSV timing column (`ms_<stage>`):

  | Stage | `pose` | `identity` | `motion` | `face` | `gestures` | `game` | `render` |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Task | 2 | 3 | 2 | 1 | 4 | 4 | 5 |

  Task numbers appear only in branch names: `task<N>-<name>`, e.g. `task3-identity`.
- **Players** are `pid` integers from 1: `state.players[1]` in code, `P1` on
  screen, `p1_<field>` in the CSV, and `<stage>_p1_<field>` for your own columns.
- **Keypoints**: `sk.L_WRIST` in code, `l_wrist` in strings. Frames are
  mirrored, so `L_*` is the player's own **right** side.
- **Shared strings** (signals, anchors, gesture names, clip names) are
  lowercase `snake_case`: `r_wrist_speed`, `above_head`, `crossing.mp4`.
- **Files**: `tools/eval_<stage>.py`, `tests/test_<module>.py`,
  `data/clips/<scenario>.mp4`.

## Develop

```bash
pytest          # filters, recorder, end-to-end pipeline smoke tests
ruff check src tests tools
```

Every module has `__init__(cfg)`, `update(...)` (pose: `detect(frame)`),
`reset(pid=None)` and `draw_debug(canvas, state)`, plus an optional
`log_fields(state)`. Replace your stub in place and keep the signature;
`core/types.py` changes need a team heads-up. Work on your task branch and
merge through a PR.

## Third-party licences

Ultralytics YOLO (AGPL-3.0), MediaPipe (Apache 2.0), the YuNet face
detector from the OpenCV model zoo (MIT) and OSNet from torchreid (MIT; code
and weights by Kaiyang Zhou) are used as pretrained models; all
are fine for coursework and are listed in the AI and external-tools statement.

Sprites from third parties:

| File | Source | Licence |
| --- | --- | --- |
| `assets/magic_effect/` (power-attack ball and beam) | "Magic Attack" by NYKNCK, itch.io, https://nyknck.itch.io/magic-attack (purchased) | None stated on the page beyond the purchase; used with attribution for coursework only |
| `assets/beam/` (normal-attack fire blast) | "Fire Attack FX - Pixel Art Effect - FX061" by NYKNCK, itch.io, https://nyknck.itch.io/fx061 (purchased) | None stated on the page beyond the purchase; used with attribution for coursework only |
