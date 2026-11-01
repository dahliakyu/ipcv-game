# AI and external-tools statement

Group 4, IPCV 2026-1A: Eline Kanters (s3197581), Riyasdeen Samsudeen
(s3941981), Lizzy-Milou de Bruijn (s3065642), Pjotr van der Made (s3207447),
Ellen Zheng (s3502155).

## AI declaration

Every member is responsible for, and must be able to explain, the code in
their own module (`README.md`, "Tasks and owners").

### Task 5, integration (Ellen Zheng): Claude Code (Anthropic, model Claude Opus 5.5)

- **Architecture and integration code.** The shared contracts (`core/`), the
  pipeline and its failure handling (`app/`), input, recording and logging
  (`io/`), the renderer and scene effects (`scene/`) were written with Claude
  Code, step by step from Ellen's instructions and design decisions, each step
  reviewed and tested on recorded clips (`docs/merge_log.md`).
- **Merging the teammates' code.** Porting the members' prototype code
  (Colour Clash `bodypose.py`, `game.py`, the face-tracking script) into the
  shared modules, with every change and its reason in `docs/merge_log.md`.
- **Evaluation.** The evaluation scripts (`tools/eval_*.py`, `tools/label_clip.py`,
  `tools/retime_clip.py`), the analysis of the clips and the tables in `results/`.
- **Ground-truth labels.** The identity labels and gesture start frames in
  `data/labels/` were made by Claude: people were pre-assigned by clothing
  colour and checked frame by frame on contact sheets, and gesture starts were
  read from frame sequences (every frame for the 30 fps and 5 fps clips, every
  5th or 10th frame for the two phone videos). Which punches in the first
  desktop game were real was judged by Ellen.
- **Changes from the evaluation.** The One Euro motion backend
  (`pose/motion.py`, now the default) and a rising-wrist veto for punches
  (`interaction/gestures.py`, tried and switched off after the labelled clips
  showed it cost real punches) were proposed and implemented by Claude on
  Ellen's request, from the measured clip data, in the Task 2 and Task 4
  modules.
- **Writing.** Drafts of the documentation (`README.md`, `docs/`), the
  technical evaluation report (`docs/project_report.md`) and the poster's
  metrics section (`docs/poster_metrics.md`), edited by the group.

### Tasks 1–4: the members' modules

Each member wrote a prototype for their task outside the shared code base
(listed below). Those prototypes were merged into the shared modules by the
Task 5 owner with Claude Code, and later extended there. The members have not
declared their own AI use for their prototypes: **[each member: fill in the AI
tools used for your prototype, or "none"]**. What follows is the AI-assisted
part of each module, from `docs/merge_log.md` ("Change Overviews" lists, per
function, where the teammates' code went; each task's "Modifications" section
gives the reasons and the clip evidence). Moving code into the shared
interface (constants into `configs/default.yaml`, frames into seconds,
keypoints by name) was done throughout with Claude and is not repeated per task.

#### Task 1, face tracking and AR (Eline Kanters)

- **Member's code:** `face_track/facial_tracking_yunet.py`: YuNet detection,
  the key-point measurements and anchor geometry, Lucas–Kanade flow fallback,
  One Euro smoothing, hat and mustache overlay, the metrics.
- **AI-assisted (Claude Code, during and after the merge):** detection in
  per-player head crops instead of the full frame with greedy matching; the
  3D head pose (`face/head_pose.py`, `solvePnP` on a canonical head) and 3D
  anchors; head rotation, mouth opening and blinks from MediaPipe Face
  Landmarker (`face/landmarker.py`); the hat hanging from its visible brim;
  shrinking large head crops; faster sprite pasting; the face effects added
  for the game (team colour and war paint, hit reaction, name tag and status
  icons, cheek bulge); the evaluation script `tools/eval_face.py`.

#### Task 2, pose and motion (Riyasdeen Samsudeen)

- **Member's code:** `Project/bodypose.py`, `KeypointTracker`: confidence
  gate, hold of missing keypoints, EMA smoothing, snap back; the pose model
  wrapper.
- **AI-assisted (Claude Code):** the smoothing moved into its own motion stage
  (`pose/motion.py`) in seconds instead of frames; velocities and the
  wrist-speed signals; the motion log columns; the One Euro backend, chosen and
  tuned from the clip measurements, now the default (2026-11-01); the pose
  calibration measurements for the game start (`pose/measure.py`); the
  evaluation script `tools/eval_motion.py`.

#### Task 3, identity (Lizzy-Milou de Bruijn)

- **Member's code:** `Project/` identity tracker: `PlayerIdentityTracker`
  (OSNet embeddings, cosine matching to stored players, new-player
  confirmation), `osnet_embedding.py`, BoT-SORT tracking in `main.py`, the OSNet
  test.
- **AI-assisted (Claude Code):** `torchreid` replaced by a copy of its
  `osnet.py`; BoT-SORT on the pose model's boxes instead of a second detector;
  the occluded/lost states and the `reentered` flag; linked tracks handled
  first, slow retry when both slots are taken; periodic update of the stored
  vectors and the link check that swaps mixed-up players back; only tracks
  with a visible face become players; renumbering left to right and the
  refreshed vector at calibration; the position in metres (`Player.pos`); the
  evaluation script `tools/eval_identity.py` and the re-identification
  threshold sweep.

#### Task 4, interaction and game (Pjotr van der Made)

- **Member's code:** the gesture rules in `Project/bodypose.py` (hands up,
  arms crossed, punch, with the confirm and punch-hold timers) and `game.py`
  (players, bolts, beam, shield, damage, its tests).
- **AI-assisted (Claude Code):** gestures as events with one action at a time
  and a release time; the punch reach measured sideways and the wrist kept at
  shoulder level; a rising-wrist veto for punches (2026-11-01, tried and
  switched off, `gestures.punch_max_rise_sw`); the shield blocking attacks; attacks aimed at the other
  player and starting at the punching hand; gestures only while facing the
  camera; the game start (phases, instruction panels, calibration); the
  drinks caught with the mouth; the paint blob and splashes; the orb and the
  final smash; the evaluation script `tools/eval_gestures.py`.

### Skills

Instruction files for Claude Code used or written for the project:

- `CLAUDE.md` (project instructions for Claude Code, loaded in every session): the ownership and coding rules, plus the group's working method and the image-processing and computer-vision knowledge each task needs (pinhole model and calibration, filtering, detection, tracking and re-identification, gesture rules, real-time integration, evaluation), written with Claude from the experience of this project.
- [fill in: other skills used]

## External tools

### Pretrained models

| Model | Used for | Source | Licence |
| --- | --- | --- | --- |
| YOLO26s-pose | Body keypoints of every person (Task 2) | Ultralytics | AGPL-3.0 |
| BoT-SORT (in Ultralytics) | Short-term person tracks (Task 3) | Ultralytics | AGPL-3.0 |
| OSNet x1.0 (MSMT17) | Appearance vectors for re-identification (Task 3); code vendored as `identity/osnet.py` | Kaiyang Zhou, deep-person-reid | MIT |
| YuNet (2023mar) | Face detection on head crops (Task 1) | OpenCV model zoo | MIT |
| MediaPipe Face Landmarker | Head rotation, mouth opening (Tasks 1, 4) | Google | Apache 2.0 |
| MediaPipe Selfie Segmenter | Person mask for the wall paint and glow (Task 5) | Google | Apache 2.0 |

### Libraries and programs

All Python packages and versions are pinned in `requirements.txt` (PyTorch,
Ultralytics, MediaPipe, OpenCV, NumPy, SciPy, filterpy, lap, PyYAML, pandas,
matplotlib, psutil, nvidia-ml-py, pytest, ruff). ffmpeg sets the frame rate of
recorded clips (`tools/retime_clip.py`).

### Algorithms from the literature

- One Euro filter: Casiez, Roussel, Vogel, "1€ Filter", CHI 2012
  (`core/filters.py`; face anchors, keypoints).
- OSNet: Zhou et al., "Omni-Scale Feature Learning for Person
  Re-Identification", ICCV 2019.
- BoT-SORT: Aharon, Orfaig, Bobrovsky, "BoT-SORT: Robust Associations
  Multi-Pedestrian Tracking", 2022.

### Assets

| Assets | Source | Licence |
| --- | --- | --- |
| `assets/magic_effect/` (beam) | "Magic Attack" by NYKNCK, itch.io, https://nyknck.itch.io/magic-attack (purchased) | None stated beyond the purchase; coursework only, with attribution |
| `assets/shield/FX019/` (shield ring of fire), `FX017/` (unused) | Pixel effect packs FX019, FX017 by NYKNCK, itch.io, https://nyknck.itch.io (purchased) | As above |
| `assets/beam/` (unused) | "Fire Attack FX - FX061" by NYKNCK, itch.io, https://nyknck.itch.io/fx061 (purchased) | As above |
| `assets/Fancy Drinks/.../individual_sprites/` (drinks) | "Fancy Drinks Pixel Asset Pack" by Pop Shop Packs, itch.io, https://pop-shop-packs.itch.io/fancy-drinks (paid) | Use in game projects allowed; no redistribution as assets |
| `assets/face/` mustache | FlyClipart, https://flyclipart.com/es/mostacho-blend-s-pixel-art-gray-world-of-warcraft-hd-png-download-1153372 | None stated; coursework only, with attribution |
| `assets/face/` hat | Pinterest, https://id.pinterest.com/tahaharnounes360/ | None stated; coursework only, with attribution |
