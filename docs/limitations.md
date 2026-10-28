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

**Cause.** The rule (`detect_crossed()`) needs the two wrist keypoints in the opposite left/right order to the shoulders. A common way to cross the arms is high, with each hand on the opposite shoulder. The pose model then does not find the wrists where they are. It either places each wrist on its own side, or puts one wrist low on the body where there is no hand. Across these attempts the wrists stay 0.1–0.5 shoulder widths (sw) short of swapping. The 3 hits were the attempts where the hands met in the middle of the chest.

**Done.** The calibration panel (`scene/instructions.py`) teaches the cross the rule detects: "CROSS WRISTS IN FRONT OF CHEST", with a figure whose wrists cross just past the middle of the chest. The rule is unchanged.

Two rules that ignore the left/right labels were tried on the clips and not kept:

| Rule | `shield.mp4` (10 attempts) | False events |
| --- | --- | --- |
| Current: wrists swapped | 3 | none seen |
| Both wrists in front of the upper chest (between the shoulders, at most 0.8 sw below them) | 10 | Fists held in front of the chest in `session.mp4`; 12 events on the arm swing of a runner in `two_people_jump.mp4` |
| As above, plus one wrist at shoulder height (±0.3 sw) within 0.3 sw of the middle | 11 (one attempt split in two) | Guard fists in `session.mp4`; 7 events in `two_people_jump.mp4` |

The rules that catch the high cross also fire on hands held in front of the chest, which happens during normal play.

**Improvement.**

- Find the hands themselves: MediaPipe Hand Landmarker on crops around the wrist keypoints gives the real hand position, also on a shoulder.
- A rule on more than the wrists, such as the elbows and the forearm direction, tuned on clips labelled frame by frame (crosses, guard fists, arm swings).
- A learned classifier on the same labelled clips (see "No learned gesture classifier" below).

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
