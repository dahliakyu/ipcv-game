Oct 28, 2026 · @el created with help of Claude Opus 5.5

# Known limitations and improvements

This page collects what the game does not do well, with the evidence, what was done about it, and what would improve it. It is the source for the limitations and future-work parts of the report. Add an entry when a test or a webcam run shows a limit that stays in the final version.

Each entry gives:

- **Seen:** what goes wrong, on which clip or log, with numbers.
- **Cause:** why, as far as the clips show.
- **Done:** what the code does about it now.
- **Improvement:** what would fix it, and what that would need.

Clips in `data/clips/` are gitignored; share the ones named here through the drive, so the numbers can be reproduced.

## Gestures

The project uses hand-written rules on the pose keypoints for every gesture (`interaction/gestures.py`), with no trained classifier (decided 2026-10-28). The rules need no training data and each threshold can be explained; the price is the limits below.

### Arms crossed: missed when the hands rest on the opposite shoulders

**Seen.** `data/clips/shield.mp4` (24 s, one player, recorded with `--solo --no-intro --record`) has 10 shield attempts. The rule sent 3 `arms_crossed` events; the live run (`logs/run-20261028-113950.csv`) raised 2 shields. In `data/clips/session.mp4`, P2 crosses the same way at about 39, 56 and 77 s, and the rule misses all three. The game itself drops nothing: every `arms_crossed` event in the logs raised a shield.

**Cause.** The rule (`detect_crossed()`) needs the two wrist keypoints in the opposite left/right order to the shoulders. A common way to cross the arms is high, with each hand on the opposite shoulder. The pose model then does not find the wrists where they are. It either places each wrist on its own side, or puts one wrist low on the body where there is no hand. Across these attempts the wrists stay 0.1–0.5 shoulder widths (sw) short of swapping. The 3 hits were the attempts where the hands met in the middle of the chest. Webcam tests also show that the misses come mostly with baggy or dark clothes: then the forearms barely stand out against the torso, so the model has little to place the wrists by. In both clips with misses the players wear such tops (a loose corduroy jacket, a black jumper), and the wrist keypoints were mostly confident (0.4–0.98) but in the wrong place, so the keypoint smoothing (`motion:`, an EMA with a 0.05 s time constant) cannot help: it averages positions, it does not correct them.

**Done.**

- The calibration panel (`scene/instructions.py`) teaches the cross the rule detects: "CROSS WRISTS IN FRONT OF CHEST", with a figure whose wrists cross just past the middle of the chest.
- The small pose model replaces nano (below), which places the wrists better in these poses. The rule is unchanged.

Two rules that ignore the left/right labels were tried on the clips and not kept:

| Rule | `shield.mp4` (10 attempts) | False events |
| --- | --- | --- |
| Current: wrists swapped | 3 | none seen |
| Both wrists in front of the upper chest (between the shoulders, at most 0.8 sw below them) | 10 | Fists held in front of the chest in `session.mp4`; 12 events on the arm swing of a runner in `two_people_jump.mp4` |
| As above, plus one wrist at shoulder height (±0.3 sw) within 0.3 sw of the middle | 11 (one attempt split in two) | Guard fists in `session.mp4`; 7 events in `two_people_jump.mp4` |

The rules that catch the high cross also fire on hands held in front of the chest, which happens during normal play.

**Then the hands themselves were looked for** with MediaPipe Hand Landmarker on a crop of each player's upper body (2026-10-28), to find where the hands are and which hand is which, independent of the arm keypoints. It did not work for this case:

- in 6 of the 10 attempts on `shield.mp4` it found no hand in any frame, and in the others only for a few frames, also with a tighter crop of the shoulders and chest. Hands gripping a shoulder show their back and curled fingers, and its palm detector looks for an open palm;
- its left/right label was consistent with the arm keypoints in only 66% of hands on `session.mp4` with the arms clearly apart, so it could not tell which hand had crossed;
- it costs 11–17 ms per crop on the CPU.

**Then a bigger pose model** (2026-10-28), with the same rule. Nano is the old model, small the new default (`pose.weights`):

| Model | `shield.mp4`: attempts caught (of 10) | `session.mp4`: real crosses / false | Jump and fitness clips | Pose time |
| --- | --- | --- | --- | --- |
| Nano (`yolo26n-pose`) | 3 | 5 / 2 (a player walking in side-on, one bent over at the frame edge) | none | ~10 ms |
| Small (`yolo26s-pose`) | 9 (one fired twice) | 4 / 0 (finds a high cross nano misses, misses two nano finds) | none | ~10 ms |
| Medium (`yolo26m-pose`) | 4 | not tested | not tested | ~10 ms |

Punches (16) and hands up (3) on `session.mp4` were the same with nano and small. The small model is the default now. Times are on the GPU with fp16; on a CPU the small model is slower.

**Improvement.**

- A rule on more than the wrists, such as the elbows and the forearm direction, tuned on clips labelled frame by frame (crosses, guard fists, arm swings).
- A learned classifier on the same labelled clips (see "No learned gesture classifier" below).
- More clips with other players and clothes: the comparison above has two people and three tops.

### Facing the camera: the body is checked with the arms down only

**Seen.** A gesture counts only while the player faces the camera (`Game.facing_camera()`). The body's turn comes from the shoulder-to-torso ratio against the calibration, but the gestures move the shoulders themselves: on `session.mp4`, a player crossing her arms while facing the camera read 22–38°, and hands up 30°.

**Cause.** Crossing the arms rolls the shoulders in, and raising both hands lifts them; both narrow the shoulder width in the image, which reads as a turn.

**Done.** The body's turn is taken only from frames with the arms down, and the last value is used for the gesture (`_track_body_yaw()`). The one-action-at-a-time rule makes the player return to a normal position between actions, so the value is recent. The head is checked in every frame.

**Improvement.** A turn that starts during a gesture is not caught. The hips' width against the calibration would not change with the arms, but the hip keypoints are less precise; tested on clips with labelled turns, it could replace or confirm the shoulders.

### Punch: decided per frame from the pose, not from the movement

**Seen.** The punch rule from `bodypose.py` first read a straight hanging arm next to a bent arm as a punch: 268 false punch frames on the two test clips, 33% of the player-frames on the jump clip (`merge_log.md`, Task 4, step 2). When a player raises both arms one after the other, the first arm passes through the punch pose, so a punch fires and the major attack is blocked (`merge_log.md`, "One action at a time").

**Cause.** `detect_punch()` checks one frame at a time: one arm straight and out to the side, the other bent. It does not use speed, so a slow arm raise to the side and a fast punch look the same.

**Done.**

- The reach is measured sideways (merge step 2b), so a hanging arm is no longer a punch.
- After any gesture event, the player must be back in a normal position before the next one fires. A raise then gives one action (the punch), never two.

**Improvement.** Use the movement as well as the pose: wrist speed from `Motion.vel` in sw/s, or the pose averaged over the last ~0.2 s, so that a punch is fast and a raise is not. That is one more rule threshold, or one of the inputs of a classifier (below).

### No learned gesture classifier

**Decision.** A trained gesture classifier was planned in `extra_features.md` (feature 8 before the renumbering) and is dropped (2026-10-28). The gestures stay rule-based.

**Why it would help.** A classifier learns from examples where the rules need one more condition per new failure, like the two limits above. It can also use the movement over time instead of one frame.

**What it would need.**

- **Inputs** per player and frame, in shoulder widths so the distance to the camera does not matter: each elbow and wrist relative to its shoulder, both elbow angles, wrist speed, and the same averaged over ~0.2 s.
- **Labelled clips:** each gesture marked frame by frame, plus hard negatives (hanging arms, running, waving, guard fists, stretching).
- **Model:** a small, explainable one first (logistic regression, k-NN or a random forest with `scikit-learn`), with the rule kept as a fallback in config.
- **Check:** train and test split by recording, not by frame. Precision and recall per player-frame and missed or extra events, for the rule and the classifier on the same clips.

## Depth

### Depth is calibrated, but no game rule uses it

**Seen.** Every frame the identity stage estimates each player's position in metres, `Player.pos` = (x, depth) (`identity/tracker.py`, `_position()`; `docs/merge_log.md`, "Game start", step 6). It is logged as `p<pid>_x_m` and `p<pid>_depth_m` and shown in `--until identity`. The calibration also makes the two players line up by depth: the neutral pose is measured only while their depths are less than `game.line_up_depth_m` (0.4 m) apart, and the nearer player is told to step back (`Game._check_line_up()`; "Game start: Modifications"). No game rule reads the depth, though. On `session.mp4`, where the players were not asked to line up, they stood a median of 0.35 m apart by depth.

**Cause.** The depth and the line-up were built for a dodge: a player stepping ~0.5 m towards the camera, off the line both players stand on, would make a paint blob miss. The Task 5 owner dropped the dodge on 2026-10-29, before it was built; its design is in the git history of `docs/extra_features.md` (feature 5 there before the drop).

**Done.**

- Depth stays: it is Task 3's spatial output, which the brief asks for ("x-y location, not in pixel, while perhaps in physical unit, maybe adding depth"). After the calibration it is scaled by the torso length, so it does not change when a player turns.
- The line-up stays in the calibration for now (Task 5 owner, 2026-10-29), although nothing needs it any more. It makes the calibration stricter: two players must stand at about the same distance before they are measured.
- The depth is not checked against a measured distance yet: the field of view (`identity.camera_hfov_deg`, 70°) and the shoulder width (`identity.shoulder_m`, 0.39 m, the same for everyone) are assumed. This is a follow-up in `docs/extra_features.md` ("Depth on the webcam").

**Improvement.**

- Use the depth in a game rule: the dropped dodge, or, for example, attack size or damage scaled by distance.
- If no rule uses it by the final version, remove the line-up from the calibration (`_check_line_up()`, `line_up_hint()`, the "STAND LEVEL" and "STEP BACK" texts, `line_up_depth_m`, `line_up_hysteresis`, the `game_level` and `game_line_gap` columns), so players are measured wherever they stand.
- To report how reliable the depth is: stand at measured distances (e.g. 1.5, 2, 2.5, 3 m), compare `p<pid>_depth_m` in the log, and set `camera_hfov_deg` from the camera's spec.
