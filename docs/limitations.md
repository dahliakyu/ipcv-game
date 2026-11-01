Oct 28, 2026 · @el created with help of Claude Opus 5.5

# Limitations and future improvements

This page collects what the final version of the game does not do well, with the evidence, what was done about it, and what would improve it; then the features that were designed but not built, and the checks that are still open. It is the source for the limitations and future-work parts of the report. What was built, and why, is in [merge_log.md](merge_log.md).

Each limitation gives:

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

**Improvement.** Use the movement as well as the pose, so that a punch is fast and a raise is not. The wrist speed is available since 2026-11-01 (`Motion.signals`, `l_wrist_speed` / `r_wrist_speed` in shoulder widths per second, logged as `motion_p<pid>_*_wrist_speed`), but no rule uses it. On `session.mp4` the fastest wrist in the 0.3 s before each punch event reached a median of 8.2 sw/s for one player and 3.3 sw/s for the other (8 punches each), against a median of 0.7 and 1.0 sw/s over all frames (90th percentile 5.0 and 3.9). One of the second player's punches peaked at only 0.8 sw/s, so a speed threshold alone would miss slow punches: it would need tuning on clips with labelled punches and raises, or be one of the inputs of a classifier (below).

### No learned gesture classifier

**Decision.** A trained gesture classifier was planned during the merge and dropped on 2026-10-28. The gestures stay rule-based.

**Why it would help.** A classifier learns from examples where the rules need one more condition per new failure, like the two limits above. It can also use the movement over time instead of one frame.

**What it would need.**

- **Inputs** per player and frame, in shoulder widths so the distance to the camera does not matter: each elbow and wrist relative to its shoulder, both elbow angles, wrist speed, and the same averaged over ~0.2 s.
- **Labelled clips:** each gesture marked frame by frame, plus hard negatives (hanging arms, running, waving, guard fists, stretching).
- **Model:** a small, explainable one first (logistic regression, k-NN or a random forest with `scikit-learn`), with the rule kept as a fallback in config.
- **Check:** train and test split by recording, not by frame. Precision and recall per player-frame and missed or extra events, for the rule and the classifier on the same clips.

## Game

### The orb can be grabbed by accident

**Seen.** On `shield.mp4` the player grabbed the orb at 12.7 s while making the shield: a hand on the opposite shoulder passed the orb as it drifted by at shoulder height. On `session.mp4` both grabs (9.6 s and 78.7 s) came from players simply moving.

**Cause.** A wrist within `game.orb_reach_sw` (0.5 shoulder widths) of the orb's edge for `orb_grab_s` (0.1 s) grabs it, whatever the hand is doing. The orb bounces at shoulder and head height, where the hands are during the shield and the punch.

**Done.** The 0.1 s hold keeps a one-frame keypoint glitch from grabbing it. The orb was first collected by punching it, which the punch rarely did (one collection in 110 s on `session.mp4`), so grabbing replaced it.

**Improvement.** Only a hand above the shoulder grabs, or a longer hold (~0.3 s) with the hand open towards the orb.

## Performance

### Rendering cost grows with resolution and wall paint

**Seen.** The pixel look costs ~14 ms per frame on a 4K clip and ~4.5 ms at 720p. The face effects and status add ~0.6 ms at 720p. The arena (wall paint, `scene/arena.py`) costs 0.3 ms with a clean wall and 4.6 ms once it has paint, 6.8 ms during a beam, mostly the Selfie Segmenter on the CPU (see "Scene").

**Improvement.** Run the webcam at 720p (the default); run the segmenter every other frame or on the GPU.

## Depth

### The camera is not calibrated

**Seen.** Every distance in metres (`Player.pos`) and the head pose (`solvePnP`) use an assumed camera: a 70° horizontal field of view (`identity.camera_hfov_deg`), a focal length of `face.focal_per_width` × the image width, the principal point in the middle and no lens distortion. None of it was measured for the webcams the group uses.

**Cause.** Ideally the camera is calibrated with a checkerboard (IPCV Tutorial 3: `cv2.findChessboardCorners` on ~20 views, `cv2.calibrateCamera`), which gives the focal length, principal point and distortion. A calibration holds for one camera at one resolution with autofocus off, and the group develops and records on several webcams and laptops, so one calibration cannot be shipped with the game, and asking every machine to calibrate first was judged too much for a game that should start at once. A calibrated mode (a `--calibrated <file>` argument loading a per-camera file) was designed on 2026-10-31 and dropped on 2026-11-01.

**Done.** All game rules use pixels or ratios of the player's own measurements (shoulder widths, the calibrated neutral pose), which do not need the camera's intrinsics. Only the logged depth and the head pose's absolute angles depend on the assumed values.

**Improvement.** Calibrate the demo webcam with a checkerboard and load its intrinsics from a config file when present: the depth would then be in true metres (still assuming one shoulder width for everyone, or measured with an ArUco marker held during the neutral pose), and the head pose would gain accuracy at the image edges.

### Depth is estimated, but no game rule uses it

**Seen.** Every frame the identity stage estimates each player's position in metres, `Player.pos` = (x, depth) (`identity/tracker.py`, `_position()`; `docs/merge_log.md`, "Game start", step 6). It is logged as `p<pid>_x_m` and `p<pid>_depth_m` and shown in `--until identity`. No game rule reads the depth. On `session.mp4` the two players stood a median of 0.35 m apart by depth.

**Cause.** The depth was built for a dodge: a player stepping ~0.5 m towards the camera would make a paint blob miss. The dodge was dropped on 2026-10-29, before it was built (design under "Future improvements").

**Done.**

- Depth stays: it is the identity task's spatial output, which the brief asks for ("x-y location, not in pixel, while perhaps in physical unit, maybe adding depth"). After the pose calibration it is scaled by the torso length, so it does not change when a player turns.
- The depth is not checked against a measured distance: the field of view (`identity.camera_hfov_deg`, 70°) and the shoulder width (`identity.shoulder_m`, 0.39 m, the same for everyone) are assumed (see "The camera is not calibrated").

**Improvement.**

- Use the depth in a game rule: the dropped dodge, or, for example, attack size or damage scaled by distance.
- To report how reliable the depth is: stand at measured distances (e.g. 1.5, 2, 2.5, 3 m), compare `p<pid>_depth_m` in the log, and set `camera_hfov_deg` from the camera's spec.

## Scene

### Wall paint needs a person mask, which costs ~4 ms a frame

**Seen.** Paint that misses lands on the wall *behind* the players, and during a beam the room dims around them (`scene/arena.py`); both need to know where the people are. The first mask, built from the keypoints, stopped the wall paint along straight capsule edges, with a strip of bare wall between the paint and a thin arm (webcam, 2026-11-01; on `session.mp4` at 100 s, a straight cut beside the raised arm of the right-hand player). With MediaPipe's Selfie Segmenter the paint meets the real outline of arms, shoulders and hair on the same frame. The arena then costs 4.6 ms median per frame once the wall has paint (6.8 ms during a beam; 0.3 ms with a clean wall), on `session.mp4` with the CPU segmenter (2.5–4 ms of it); a frame took 35.8 ms median in total.

**Cause.** The segmenter runs on every frame that has wall paint or a dimmed room, on the CPU, at its own input size (256 x 144); its mask has a soft band of ~5 small-image px at the outline, ramped narrower (`render.arena_segmenter_ramp`). It marks anyone in view as a person, not only the players, and sometimes a patch of furniture (a bag on the sofa at the left edge of `session.mp4`), where the wall then stays bare. Without the model file the arena falls back to the keypoint mask (`person_mask()`): a coat, long hair or a hand past the wrist keypoint count as wall there, and a player without a torso leaves a box-shaped hole.

**Done.** Segmenter mask (`render.arena_segmenter`) with the keypoint mask as fallback; wall paint at 85% opacity on the pixel look's small image; the beam leaves no stripe on the wall.

**Improvement.**

- Run the segmenter every other frame (the players move little between two frames) or with MediaPipe's GPU delegate.
- Per-player masks from YOLO-seg if a game rule needs to know whose outline it is (costs a second YOLO run per frame).
- Check on the webcam with two players in different clothes.

## Pipeline

### A failing module is skipped, not fixed

**Seen.** Tested with modules made to raise (`tests/test_pipeline.py`): the game keeps running, the failing stage is named at the top right and logged as `pipeline_failed`.

**Cause.** Each stage runs inside a guard (`Pipeline._guard()`): an exception gives that stage an empty output for the frame. Pose, gestures and the game have no stub, so while they fail there are no players, no gestures, or a frozen game; identity, motion and face switch to their stub backend after `pipeline.fallback_after_s` (1 s) of failing on every frame, and stay on it for the rest of the run.

**Done.** One traceback per stage is printed, then one line per new run of failures; `pipeline_fallback` in the log shows which stages run on a stub. `--set pipeline.catch_errors=false` turns the guard off to debug a module.

**Improvement.** Try the real module again after a while on its stub; a CPU fallback for the pose model if CUDA fails.

## Future improvements

Designed during the project but not built. The designs are condensed here; the full versions are in the git history of `docs/extra_features.md` (removed on 2026-11-01).

### Final smash: paint rain that lasts until the caster blinks

**Now.** A player who grabs the orb glows; hands up then fires the magic beam (12 damage) and both players' orbs are emptied ([merge_log.md](merge_log.md), "Final smash"). This was an interim before a playtest and is the final version.

**Design.** Like the Smash Ball in Smash Bros: hands up with a full orb sends the orb up out of the frame (the room dims with a flash in the caster's colour), a target mark appears under the opponent, and paint drops rain on them, densest in the middle of a drop zone ~2 body widths wide that follows the opponent at ~0.3 frame widths/s, so a quick long step to the side dodges some drops. The rain lasts only while the caster does not blink: the first blink (or the caster's face not being measured, so turning away cannot hide a blink) ends it, with a 6 s cap; the caster's eyes glow in their team colour while it lasts. Each drop does a little damage (~1) and adds a small splash on the body; the shield blocks drops while it is up. Attacking becomes a staring contest.

**Needs.** A blink detector: one was built (per-player baseline, hysteresis, 60–500 ms, no blink while looking down) and removed on 2026-11-01 when the rain was dropped ([merge_log.md](merge_log.md), "Final smash", steps 3 and 6). At play distance (~2.3 m) MediaPipe's eye scores hardly moved: its design thresholds found one blink in 110 s on `session.mp4`, so it would first need clips of deliberate blinks to tune, or the eye aspect ratio from the face landmarks instead of the blendshape score. Also the caster measured every frame during the rain, a `FinalSmash` replacing `PaintBeam`, and drop splashes small enough to keep the body paint under ~2 ms per player at ~5 drops a second. A solo test path: a key that lets a test rain fall on P1.

### Both players attacking at the same time

**Now.** Blobs and beams of the two players fly through each other and both hits land; when both would finish the round in the same frame, the attack fired first wins (`_apply_damage()` ignores everything once `winner_id` is set).

**Design.** The attack level is the attack drink's multiplier (`GamePlayer.attack`, 1 without one). Two blobs whose paths cross in a frame: the higher level flies on and the other bursts mid-air; with equal levels both burst. Final blows in the same frame: apply all damage first, then the higher level wins, or a draw ("DRAW", both colours) with equal levels; that needs a `result` instead of only `winner_id`.

### Dodge by stepping forward

**Design** (dropped 2026-10-29). Each player's depth measured in the pose calibration, with both players standing at the same distance, is their fight line; a player at least ~0.5 m closer (back under ~0.3 m, at most ~1 s per dodge, no attacking meanwhile) lets a normal blob pass and fly on to the wall behind them, with "MISS!" above the head. It needs a depth signal faster than the smoothed `Player.pos` (0.3 s lag; a blob crosses the frame in ~1 s), and a calibrated camera for the threshold to mean a real 0.5 m (see "The camera is not calibrated"); without calibration the same rule is "about 20% closer", from the player's own torso ratio.

### Other ideas, not designed

- Paint splats on the face (in face coordinates) as damage.
- A "paint spit" attack: hold the mouth wide open to fire a blob from the mouth.
- Real props as consumables: YOLO detection of an apple, banana, bottle or cup (COCO classes) instead of virtual drinks; needs a second detection model per frame.

## Checks still open

What was checked on recorded clips and in tests only, or not at all.

| Check | Why | Where |
| --- | --- | --- |
| Balance over real rounds | Damage, speeds and the shield time are the teammates' `game.py` values; the beam (12 damage) may feel small for a grabbed orb | `game:` config |
| Webcam check of the game start | The intro was checked on clips only: start places, pose calibration, countdown, result and replay with two players; the P1-left swap; whether the 30° facing rule suits real players | `--until game` |
| Webcam check of the drinks | One webcam run (`logs/run-20261028-231534.csv`, which set `drink_hold_s` to 2.5 s). Catching and drinking alone and with two, `jawOpen` 0.5 / 0.25, the hold time, the buff values, the spawn rate and the new fall time (6 s) | `game:` `drink_*` keys |
| Webcam check of the face effects and face stage | War paint and hat band with real head turns, name tag steadiness, hat knock-off, chin up and down, mouth bars, `hat_depth_mm` | `face:`, `render:` config |
| Jitter on a still clip | All test clips are moving; shrinking crops and the head pose may add shake that only a still clip shows | `tools/eval_face.py` |
| Identity on real crossings | Identity was checked on edited copies of `session.mp4` (a player blacked out, halves swapped), not on real crossing, occlusion and re-entry clips; also re-entry and a player turning away live | `--until identity`, `identity:` config |
| `identity.reid_threshold` (0.6, from `bodypose.py`; to check in testing) | The cosine similarity of OSNet appearance vectors at which a new track becomes a known player (`identity/tracker.py`, `_find_player()`), e.g. a player walking back in, and at which the periodic re-check moves a link to the other player. Never tuned: there are no real re-entry clips. With the Hugging Face weights, same-player similarities on the fitness test clip are often under 0.6, so a returning player could become a new player instead of getting their P1/P2 back. Log `identity_p<pid>_*` on re-entry and crossing clips and set it between the same-player and other-player similarities | `identity:` config |
| `identity.botsort.track_buffer` is in frames | Ultralytics counts frames: the occluded time is 1 s at 30 FPS but 1.7 s at 17 FPS | `identity.botsort` |
| Depth against a tape measure | Stand at 1.5, 2, 2.5 and 3 m and compare `p<pid>_depth_m` | `identity:` config |
| Gesture evaluation | `tools/eval_gestures.py` needs clips with labelled gestures; the test clips have none | `tools/` |
| MediaPipe GPU delegate | The face landmarker costs ~8 ms per frame on the CPU, the Selfie Segmenter 2.5–4 ms | `face/landmarker.py`, `scene/segment.py` |
| Asset packs in the final ZIP | All of `assets/Fancy Drinks/` (paid; redistributing the files as assets is not allowed) is committed; keep only the sprites in use, or leave the pack out with a download note. `assets/beam/` (fire blast) is unused since the paint blob | `assets/`, `README.md` |
| Remove `garb/` | The teammates' original code, kept locally for the merge (gitignored); everything in it is merged or deliberately left out | `.gitignore` |
