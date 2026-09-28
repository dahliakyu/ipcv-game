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
```

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

Ultralytics YOLO (AGPL-3.0) and MediaPipe (Apache 2.0) are used as pretrained
models; both are fine for coursework and are listed in the AI and
external-tools statement.
