# IPCV webcam multiplayer game (University of Twente, IPCV 2026-1A)

Group project: real-time webcam game, ≥2 players on one webcam, five compulsory
tasks (face AR, pose/motion, identity tracking, interaction, scene integration).
Game design is not final yet; we are in phase 1: face tracking, motion/gesture
tracking, identity.

Architecture, data contracts, ownership and plan:
@docs/phase-1/setup-guide.md

Current status and per-task to-do lists: docs/phase-1/status.md

Project brief: docs/phase-1/IPCV_2026-1A_Project_Description_v1-1.pdf

Candidate modules and libraries: @docs/candidate-modules-and-libraries.md

## Ownership

Each task has one owner, and each owner must be able to explain their own code.

| Task | Packages |
| --- | --- |
| 1 Face + AR | `face/` |
| 2 Pose + motion | `pose/` |
| 3 Identity | `identity/` |
| 4 Interaction | `interaction/` |
| 5 Integration | `core/`, `io/`, `scene/`, `app/`, `tools/` (shared infrastructure) |

- Work only in the packages of the task the current user owns. If it is not
  clear which task that is, ask before editing.
- In another task's packages, only build stubs or interfaces unless that
  task's owner asks for more.
- Owners must understand every line in their module: explain design choices
  and trade-offs, and prefer explainable methods over black boxes.

## Rules

- `core/types.py` is the contract between modules. Don't change it without a
  heads-up to the whole team.
- Every threshold goes in `configs/default.yaml` (in the module's own section)
  with a comment on why it has that value; never hard-code one.
- Coordinates are pixels in the mirrored full-res frame; filters use `Frame.t`,
  not frame counts. Index keypoints by name via `core/skeleton.py`.
- Keep the module interface: `__init__(cfg)`, `update(...)` (pose:
  `detect(frame)`), `reset(pid=None)`, `draw_debug(canvas, state)`, optional
  `log_fields(state)`. Reset per-player state when `Player.reentered` is True.
- Reuse `core/filters.py`, `core/geometry.py` and `scene/draw.py` instead of
  writing new copies.
- New dependencies go in the pinned `requirements.txt` with a comment.
- Branch per task; merge to `main` only through a PR reviewed by Task 5.

## Environment and commands

Ubuntu 24.04; demo machine has an RTX 3070 Ti. Conda env `ipcvgame`
(Python 3.12); see README.md for setup.

```bash
pytest                                                    # before every PR
ruff check src tests tools
python tools/run_module.py --until <stage> --source data/clips/<clip>.mp4
python -m ipcvgame.app.main --headless --max-frames 100   # quick smoke run
```
