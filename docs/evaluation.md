Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Evaluation: how to get the report numbers

Every task evaluates its module on the same recorded clips, so the results in the report are reproducible and comparable. This page explains the workflow and each evaluation script. Each task owner adds a section for their own script (see [Adding a script](#adding-an-evaluation-script)).

## The workflow

1. **Clips.** Put test videos in `data/clips/` (gitignored; shared through the drive).
   - Record your own with `python tools/record_clip.py <scenario>`. That writes `<scenario>.mp4` plus a `<scenario>.timestamps.csv` file with the real capture times, so the filters see the true time step on replay.
   - Downloaded videos work too. Without the timestamps file, time comes from the video's own frame rate.
   - Name clips after the scenario (`normal_play`, `crossing`, `occlusion`, `reentry`, `fast_arms`, `poor_lighting`). The evaluation scripts label their results with the clip name, so the name *is* the condition in the report table.
2. **Run logs.** Each pipeline run writes one CSV with a row per frame (see [Run logs](#run-logs)). Evaluation scripts either run the pipeline on a clip themselves or read an existing log.
3. **Evaluation scripts.** `tools/eval_<stage>.py` turns the logs into a table per clip and player, printed as Markdown so it can go straight into the report.

All commands run from the repo root in the `ipcvgame` env, and need no webcam.

## Run logs

`python -m ipcvgame.app.main --source <clip> --log <file.csv>` writes the log. Add `--headless` for no window, and `--until <stage>` to stop after your stage, which is faster.

| Columns | Content |
| --- | --- |
| `frame_id`, `t` | Frame number and time in seconds since the first frame |
| `ms_<stage>`, `ms_total` | Time per pipeline stage and for the whole frame (ms) |
| `p<pid>_status`, `p<pid>_cx`, `p<pid>_cy`, … | Per player, from the identity stage |
| `<stage>_p<pid>_<field>` | A module's own columns, from its `log_fields(state)` method |

## Task 1: face tracking (`tools/eval_face.py`)

### Running it

```bash
# Run the pipeline on each clip (headless, up to the face stage), then evaluate
python tools/eval_face.py data/clips/*.mp4

# Also save the table for the report
python tools/eval_face.py data/clips/*.mp4 --markdown results/face.md

# Evaluate logs that already exist (no re-run)
python tools/eval_face.py logs/eval/face-crossing.csv

# Parameter experiment: --set changes the config for this run, --tag labels it
python tools/eval_face.py data/clips/*.mp4 --tag noflow --set face.flow_max_s=0
python tools/eval_face.py data/clips/*.mp4 --tag beta0  --set face.filter_beta=0.0
```

Logs go to `logs/eval/face-<clip>[-<tag>].csv` (gitignored). `--log-dir` changes the folder.

### What it measures

One row per clip and player.

| Column | Meaning |
| --- | --- |
| `frames` | Frames in which the player exists (the identity stage reports them) |
| `detect %`, `flow %`, `hold %`, `no face %` | Where the face came from in those frames. **detect**: YuNet found it. **flow**: YuNet missed it and optical flow carried it. **hold**: neither, so the last position is shown and fades out. **no face**: no face yet, or it was lost |
| `jitter raw px`, `jitter smooth px` | Frame-to-frame shake of the mouth anchor: the RMS of the second difference ‖p₊₁ − 2p + p₋₁‖ over three consecutive *detect* frames. Steady motion cancels out in a second difference, so what remains is shake. *raw* is the unsmoothed YuNet point, *smooth* the One-Euro-filtered anchor the effect is drawn at |
| `jitter raw %w`, `jitter smooth %w` | The same, as a percentage of the face width, so near and far players compare |
| `lag px`, `lag p95 px`, `lag %w` | Alignment error: distance between the smoothed anchor and the raw measurement on detect frames (mean, 95th percentile, mean as % of face width). This is what smoothing costs: the effect trails the face |
| `crop px` | Mean side of the head crop YuNet runs on. Detection time grows with its area |
| `face ms`, `face ms p95` | Time of the face stage per frame (mean, 95th percentile) |

### Reading the results

- **Jitter versus lag is the filter trade-off.** Stronger smoothing (lower `face.filter_min_cutoff`, lower `face.filter_beta`) lowers jitter and raises lag. Run the same clips with a few `--set` values and `--tag`s, and report both columns side by side.
- **Measure jitter on still clips.** On clips with fast movement (jumping, fast arms), the second difference also contains real acceleration. Use those clips for coverage and lag instead.
- **Coverage per condition.** `detect %` per clip (`normal_play` versus `poor_lighting`, and so on) shows how robust the detector is. `flow %` shows how much the optical-flow fallback adds: compare against a `--tag noflow --set face.flow_max_s=0` run, where those frames become `hold`.
- **No face swaps by construction.** Each face is detected inside its own player's head crop, so a face cannot jump to the other player. Identity switches are a Task 3 metric (`tools/eval_identity.py`).
- **Speed.** `face ms` covers the face stage only. For the full-pipeline frame rate, run the game normally (without `--until`) and look at `ms_total` in the log.

### Example

`two_people_fitness_challenge.webm`, a downloaded 428×527 test clip (not one of the shared clips), on 2026-10-25, with and without optical flow. Columns shortened:

| clip | player | detect % | flow % | hold % | no face % | jitter raw px | jitter smooth px | lag px | face ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_people_fitness_challenge | P1 | 76.2 | 23.8 | 0.00 | 0.00 | 2.61 | 1.73 | 2.12 | 2.60 |
| two_people_fitness_challenge | P2 | 78.5 | 13.8 | 0.00 | 7.69 | 3.98 | 2.53 | 2.63 | 2.60 |
| two_people_fitness_challenge-noflow | P1 | 76.2 | 0.00 | 23.8 | 0.00 | 2.61 | 1.72 | 2.16 | 2.16 |
| two_people_fitness_challenge-noflow | P2 | 78.5 | 0.00 | 13.8 | 7.69 | 3.98 | 2.64 | 2.60 | 2.16 |

The filter roughly halves the shake (2.6 → 1.7 px for P1) at about 2 px of lag. Without flow, the frames YuNet misses (up to 24%) show a frozen, fading hat instead of a tracked one; flow costs about 0.4 ms per frame.

## Other tasks

Each owner adds a section like the one above when their script exists. Planned (see [status.md](phase-1/status.md)):

| Script | Task | Measures |
| --- | --- | --- |
| `tools/eval_motion.py` | 2 | Raw versus smoothed keypoint jitter, lag |
| `tools/eval_identity.py` | 3 | Identity switches on hand-labelled clips |
| `tools/eval_gestures.py` | 4 | True and false gesture triggers on the clips |

## Adding an evaluation script

- **Name:** `tools/eval_<stage>.py`, where `<stage>` is the pipeline stage name (`face`, `motion`, `identity`, `gestures`), not the task number.
- **Columns:** write them from your module's `log_fields(state)` as `<stage>_p<pid>_<field>`, so the run logger picks them up.
- **Input:** accept clips (run the pipeline with `--headless --until <stage> --log …`) and existing logs. `tools/eval_face.py` can be copied as a template.
- **Output:** print one Markdown row per clip and player, and support `--markdown <file>`.
- **Tests:** put the metric maths in functions and test them on a synthetic log with known values, as `tests/test_eval_face.py` does.
- **Docs:** add a section to this page with the commands, a column table, and how to read the results.
