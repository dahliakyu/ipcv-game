# Technical Evaluation Report – Webcam-Based Interactive Game Design

**Group number:** IPCV_Project_Group_4

**Group members:**

- Eline Kanters (s3197581)
- Riyasdeen Samsudeen (s3941981)
- Lizzy-Milou de Bruijn (s3065642)
- Pjotr van der Made (s3207447)
- Ellen Zheng (s3502155)

> **Important note (from the template):**
> *The texts below in italics are instructions on the suggested evaluation metric. The final grade of this technical evaluation report depends on the comprehensiveness and scientific critical thinking of the conducted evaluation, experiments, results and analysis.*
>
> *No strict page limit, but it should give sufficient information while being concise and clear. The suggested length is 5 to 8 pages. Above 8 pages is allowed, but not encouraged.*

## 1. Final system snapshots and figures to show the demo

*Add snapshots/figures to show the final performance of the designed interactive game.*

<!-- TODO: snapshots/figures -->

In Colour Clash, two players stand in front of one webcam, Player 1 in red and Player 2 in blue, and fight with their bodies: a punch throws a paint blob at the other player and crossed arms raise a shield for one second. Drinks fall from the top of the screen and are caught with an open mouth for health, attack or defence buffs, and a player who grabs the bouncing orb with a hand can raise both arms to fire a stronger magic beam. The first player to bring the other's health to zero wins the round.

## Evaluation set-up (all tasks)

**Clips.** All measurements come from webcam clips of group members replayed through the final pipeline. A clip replays its recorded capture times, so the filters see the true time steps and every run is repeatable. The raw clips are not submitted (available on request); the per-frame logs of every run (submitted separately), the result tables (`results/`) and the hand labels (`data/labels/`) are, and `README.md` ("Evaluation") lists the files per clip set.

| Clip set | Recorded | Content | Used for |
| --- | --- | --- | --- |
| *Still* (1, 2, 3 m) | 29 Oct, laptop webcam, 15 fps | Two players standing still at 1, 2 and 3 m | Tasks 1, 2, 3 (jitter, depth) |
| *Moving* (1, 2, 3 m) | same | Two players moving their heads and bodies | Tasks 1, 2 |
| *Hand over face* (1 m dim, 1 and 2 m changing light, 3 m) | same (8.5 fps in the dim clip) | Hands over the face; dim and changing light | Task 1 |
| *Swap* | same | Two players swap places; a third person stands at the back wall | Tasks 1, 3 |
| *Crossing ×4*, *Crossing ×22* | same | 4 and 22 crossings (40 s and 116 s, the second to a metronome) | Task 3 |
| *Re-entry*, *New person* | same | Players leave and return; one player replaced by a new person | Task 3 |
| *Depth* (1, 2, 3 m) | same | Two players standing at measured distances (at 3 m also 0.5 m to the side) | Task 3 (depth) |
| *Punch reps*, *Shield reps*, *Beam reps* | 31 Oct, 30 fps | Two players repeat punches (11), shields (15), hands-up beams (14) | Task 4 |
| *Phone moves 1*, *Phone moves 2* | 30 and 31 Oct, phone, 25 fps | A third player: punches at 1–3 m, T-poses, hands up, crossed arms (113 s and 60 s) | Task 4 |
| *Demo game* + its live log | 29 Oct, laptop, older game version | 100 s of two-player play | Tasks 4, 5 |
| *Laptop game* | 31 Oct, laptop, live game, ~5 fps | 49 s of move tests by one player, then 188 s of two-player play | Tasks 4, 5 |
| *Desktop game*, *Solo shields* | 26 and 28 Oct, desktop webcam | 110 s of two-player play; 10 shields by one player | Tasks 1, 2, 4, 5 |

**Hardware.** Most clips were recorded live on the laptop; all were replayed on the desktop PC, in two configurations: with the GPU, and CPU only (as at the demo, where the laptop runs without CUDA). Live laptop numbers come from the game's own per-frame log of the demo session.

| | Desktop PC (replay, measurements) | Laptop (recording, live demo) |
| --- | --- | --- |
| Model | Desktop | HP OMEN Laptop 15-en0xxx |
| CPU | AMD Ryzen 7 5700G, 8 cores / 16 threads, 3.8–4.6 GHz | AMD Ryzen 7 4800H, 8 cores / 16 threads, 2.9–4.2 GHz |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8 GB (driver 580.159, CUDA 12.8) | Not used (CUDA set-up problem) |
| RAM | 16 GB | 16 GB |
| OS | Ubuntu 24.04 LTS | Ubuntu 24.04 LTS |
| Camera | – (clips) | Built-in webcam, 1280 × 720 |
| Software | Python 3.12, PyTorch 2.11, Ultralytics 8.4.164 (YOLO26s-pose, 640 px input), OpenCV 5.0, MediaPipe 1.0.1 | Same repository |
| GPU configuration | Pose (FP16) and OSNet on CUDA; face (YuNet, MediaPipe) on the CPU | – |
| CPU-only configuration | Everything on the CPU (`--set pose.device=cpu --set identity.reid_device=cpu`) | Everything on the CPU |


**Lighting.** Besides the dim and changing-light clips, every script can replay a clip with all pixel values multiplied by 0.3 plus Gaussian noise (σ = 8 grey levels): the same scene with less light and the camera gain's noise, so differences come from the light alone (`--light 0.3 --noise 8`, on the *Desktop game*).

**Ground truth.** Identity: in one frame every 0.5 s (480 frames), every visible person was marked as player A, player B or someone else (C: a bystander or newcomer), at their box centre. Gestures: the first frame of every intended action, per player, read at 0.03–0.2 s resolution (every frame; every 5th or 10th frame for the phone videos, where only recall and precision are reported per clip). The labels were made with an AI assistant (Claude; see the AI statement): people were pre-assigned by clothing colour, then every labelled frame was checked on contact sheets and corrected; which punches in the *Desktop game* were intended was judged by a group member. They are in `data/labels/` and can be reviewed with `tools/label_clip.py`. Scripts: `tools/eval_<face|motion|identity|gestures|pipeline>.py` (how to run them: `README.md`, "Evaluation"); result tables and logs in `results/`.

## 2. Tech-evaluation: Face tracking and augmented effects

**Responsible student:** Eline Kanters (s3197581)

**Expected outcome:** Smooth, stable tracking of at least two faces, with jitter-resistant effects that remain correctly aligned and assigned during movement, scale change, and short occlusion.

### Experimental design and protocol

Two subjects per clip, standing still (*Still*), moving (*Moving*) and covering their face with a hand (*Hand over face*) at 1, 2 and 3 m, plus a place swap with a bystander (*Swap*): 11 clips, 22 player-runs, 7,074 player-frames. Impact factors: distance (face width 300 → 110 px), motion, occlusion, light (dim clip; synthetic darkening), subjects. Each clip is run up to the face stage; `tools/eval_face.py` reads per frame where the face came from (YuNet detection, optical flow, hold), the raw lip point and the smoothed anchor the effect is drawn at.

### Measured results

- **Two-face detection rate** (YuNet detections / frames the player is tracked):

  | Distance | Still | Moving | Hand over face |
  | --- | --- | --- | --- |
  | 1 m | 100 / 100 % | 100 / 95.6 % | 100 / 94.5 % |
  | 2 m | 100 / 100 % | 97.1 / 100 % | 100 / 100 % |
  | 3 m | 100 / 74.2 % | 94.6 / 81.8 % | 91.9 / 99.1 % |

  Frames YuNet misses are carried by optical flow (up to 25.8 %, P2 at 3 m); no face was lost (*no face* 0 %) except in *Swap* (19.9 % for the player who walked out of view).
- **Jitter** (RMS second difference of the mouth anchor over three consecutive detections, raw → smoothed): still 1.3–4.9 → 0.46–2.1 px (−57 to −66 %; 1.0–2.7 % of the face width, 7.0 % for the player at 3 m whose face was 26 % optical flow); moving 3.2–6.0 → 2.0–5.2 px (−15 to −45 %, real head motion stays).
- **Effect alignment error** (distance between the drawn anchor and the raw detection): 0.36–1.1 px still and 0.9–1.7 px moving (mean; 95th percentile ≤ 4.3 px), 0.7–4 % of the face width.
- **Identity switches / multiple subjects:** each face is searched only in its own player's head crop, so a face cannot move to the other player by itself; the effect follows the player identity of Task 3. In *Swap* it therefore went wrong exactly when identity did (a bystander took a player slot; Task 3).
- **Latency:** face stage 10.3–14.3 ms per frame (95th percentile ≤ 15.9 ms) on the desktop CPU; detection to the drawn effect is the whole frame, 29–39 ms median (Task 5). On the laptop (live log) 22 ms.
- **Lighting:** in the dim clip detection stays at 100 / 88.8 % (flow covers the rest), but the camera itself slowed to 8.5 fps. Synthetic darkening of *Desktop game* lowers detection from 95.6 / 99.3 % to 54.1 / 51.7 % (flow then covers 16–41 %) and raises raw jitter from 5.2 to 12.6 px (smoothed 3.9 → 8.2 px).

### Technical conclusion

The expected outcome is largely achieved: both faces are found in 100 % of the frames up to 2 m and in 82–100 % while moving at 3 m, with optical flow covering the misses, and each effect stays with its own player because faces are searched only inside that player's head crop. The strongest evidence is the still-frame test, where the One Euro filter removes 57–66 % of the anchor jitter while the effect stays within 0.4–1.1 px of the detected face. The main limitation is small and dark faces: at 3 m (face crop about 110 px) one player's detection fell to 74 %, and in synthetically dark light to 52–54 %, so the effect then rides on optical flow for longer stretches. A justified improvement is to upscale head crops below about 150 px before running YuNet: detection only dropped where the head crop shrank to about 110 px (at 1–2 m, with 160–300 px crops, it stayed at 97–100 %), and the extra cost grows only with the small crop area (the face stage now takes 10–14 ms).

## 3. Tech-evaluation: Body pose estimation and motion tracking

**Responsible student:** Riyasdeen Samsudeen (s3941981)

**Expected outcome:** Responsive body, limb, or hand keypoints converted into smooth control signals, with controlled behaviour for uncertain or missing keypoints

### Experimental design and protocol

Two subjects per clip, still and moving at 1–3 m, crossing (*Crossing*) and playing (*Demo game*, *Desktop game*): 20 player-runs. The 17 COCO keypoints count as detected when their confidence is ≥ 0.35 (`motion.kpt_conf_threshold`). A frame counts as *still* when every trusted keypoint, averaged over 5 frames, moves slower than 0.5 shoulder widths/s; this split uses the raw keypoints, so every filter is scored on the same frames. `tools/eval_motion.py` stores every raw and smoothed keypoint per frame. Impact factors: distance, motion speed, people at the frame edge, light, frame rate. The smoothing filter itself was chosen by experiment (below): first an exponential moving average (EMA, time constant τ = 0.048 s, from the group's prototype), then a One Euro filter, swept over 9 settings.

### Measured results

- **Keypoint detection rate:** upper body (shoulders, elbows, wrists: what the gestures read) 98–100 % on the still and moving clips; nose 97.5–100 %, shoulders 100 %, elbows 100 %, wrists 96.5–100 %. Lower on crossings (62–96 %), where players walk partly out of the frame, and for hands held behind the back (wrists 64.6 %). Knees and ankles are mostly outside the frame (0–100 % and 0–33 %). A detection *error* needs hand-marked keypoint positions, which we did not make; the still-frame jitter below is the model's frame-to-frame precision.
- **Smoothing filter: EMA versus One Euro.** The EMA moves a = 1 − e^(−dt/τ) of the way to each detection; the One Euro filter is an EMA whose cut-off frequency rises with the keypoint's speed (min cutoff 0.5 Hz + 0.03 Hz per px/s, the best of the 9 settings swept), so it smooths still keypoints strongly and fast ones lightly. Jitter is the RMS second difference ‖p₊₁ − 2p + p₋₁‖ per trusted keypoint, on the same frames for both filters:

  | | Raw | EMA | One Euro (final) |
  | --- | --- | --- | --- |
  | Still jitter, 15 fps clips (median reduction) | 2.8–15 px | −34 % | **−63 %** (−37 to −81 %) |
  | Still jitter at 6.7 fps (*Demo game*) | 6.9 / 7.5 px | −4 % | **−28 %** |
  | Delay in fast moves (punch and beam clips; *Desktop game*, *Crossing*) | – | 20–28.5 ms | **10–18 ms** |
  | Delay in slow head moves (*Still*, *Moving*) | – | 14–20 ms | 18–96 ms |

  We switched the game to the One Euro filter for three measured reasons: it removes about twice as much jitter from a standing player; it trails *less* in the fast moves the gestures depend on (the delay drops where the speed rises); and it keeps smoothing at low frame rates, where the EMA's per-frame weight approaches 1 (at the *Demo game*'s 6.7 fps the EMA removed 4 % of the jitter, the One Euro filter 28 %), which matters because the demo laptop runs the game on its CPU at 7–11 FPS. Its cost is a larger delay on slow movements (up to ~100 ms), where the lag is a few pixels and not visible in play. With the same gesture rules, the One Euro filter made every action fire 30–90 ms sooner on the repetition clips (`results/task4_gestures_first_run.md` against `task4_gestures_final.md`). The EMA's measured delay matched its theory (dt(1−a)/a = 22 ms at 15 fps; measured 14–22 ms).
- **Motion signal delay:** frame to usable signal: pose + identity + motion ≈ 10.5 ms on the desktop GPU, plus the filter delay above (10–18 ms in fast moves): 21–29 ms, plus the camera's own latency (not measured).
- **Missing keypoints:** a keypoint under the threshold keeps its last position for up to `hold_s` = 0.33 s, then becomes invalid and snaps back when seen again (with no speed from the jump). Upper-body keypoints held: 0–0.7 % of keypoint-frames on the still and moving clips (6.5 % with hands behind the back, 1.3–9 % on crossings); dropouts bridged by the hold: 44–100 % (median dropout 0.13–0.8 s).
- **Module latency:** pose 9.6–10.1 ms mean (95th percentile ≤ 11 ms) on the RTX 3070 Ti, 49–52 ms median on the desktop CPU, 77 ms live on the laptop CPU; motion 0.1 ms.
- **Lighting:** synthetic darkening of *Desktop game*: upper-body detection 95.3 / 99.3 % → 93.6 / 99.7 %, wrists 91 / 98 % → 87 / 99.7 %; still raw jitter 4.7 / 6.0 → 10.1 / 8.8 px (the noise adds jitter), smoothed 1.9 / 2.7 → 5.8 / 2.9 px.

### Technical conclusion

The expected outcome is achieved: the upper-body keypoints the game uses are detected in 98–100 % of frames, the smoothed signal is available 21–29 ms after the frame, and missing keypoints are held for 0.33 s, which bridges 44–100 % of the dropouts without the game seeing them. The strongest evidence is the filter comparison on identical frames: the One Euro filter removed a median 63 % of the still-frame jitter against 34 % for the EMA, while trailing less in fast moves (10–18 vs 20–28.5 ms), and it still smooths at the laptop's low frame rate (−28 % vs −4 % at 6.7 fps), which is why it replaced the EMA. The main limitation is that keypoint accuracy itself was not measured (no hand-marked keypoints), and that keypoints of players partly outside the frame drop out (62–82 % upper-body detection on the crossing clips). A justified improvement is a constant-velocity Kalman filter (already in `core/filters.py`) for the wrists during dropouts: it would predict a fast-moving wrist through the 0.13–0.8 s gaps instead of holding it at its last position, which is where a held keypoint is most wrong.

## 4. Tech-evaluation: Multiplayer detection and identity tracking

**Responsible student:** Lizzy-Milou de Bruijn (s3065642)

**Expected outcome:** At least two players retain persistent identities, controls, scores, and effects through crossing, occlusion, temporary disappearance, and re-entry.

### Experimental design and protocol

Two subjects cross 4 times (*Crossing ×4*) and 22 times to a metronome (*Crossing ×22*), leave and re-enter (*Re-entry*, and 10 exits at the frame edge in the metronome clip), are replaced by a new person (*New person*) or swap places with a bystander in view (*Swap*): 480 hand-labelled frames, 847 player-frames. Each clip is run through the game without the intro; `tools/eval_identity.py` puts every active player on the labelled person nearest to its box centre (within 15 % of the frame width). Depth: two subjects at measured distances of 1, 2 and 3 m (*Depth* and the face clips, 24 player-runs).

### Measured results

| Clip | Labelled frames | Identity accuracy | Switches | Crossings kept | Re-entries recovered |
| --- | --- | --- | --- | --- | --- |
| *Crossing ×4* | 82 | 100 % | 0 | 4 / 4 | 2 / 2 |
| *Crossing ×22* | 233 | 100 % | 0 | 22 / 22 | 10 / 10 |
| *Re-entry* | 69 | 100 % | 0 | – | 2 / 2 (0.72 s) |
| *New person* | 32 | 66.1 % | 1 | – | – |
| *Swap* | 65 | 42.3 % | 2 | – | 4 / 5 |

- **Identity accuracy / switches / success rate:** 100 % of player-frames correct and 0 switches over 26 crossings and 14 re-entries with the two players only (681 player-frames). Two failure cases: a new person who appears where a player just left inherits that player's id (32 % of player-frames on the newcomer; the BoT-SORT track carried over, so OSNet never checked); and a bystander standing at the back wall (~6 m) took the second player slot for the whole clip (49 % of player-frames on the bystander, 48 % of the real players' appearances without an id).
- **Occlusion / re-entry recovery:** 14 / 14 returns got their own id back, after 0.72 s (median, labelled) in *Re-entry*; OSNet re-identified them 0.26–0.53 s after their track appeared. No extra ids were created in any clip.
- **Spatial error:** depth (pinhole model on the shoulder width, no camera calibration) is consistently too large: median 2.10 / 3.42 / 4.76 m at 1 / 2 / 3 m (error +1.1 / +1.4 / +1.8 m), but precise: interquartile range 0.06–0.3 m for a standing player. One overall scale factor (0.60) would leave ≤ 0.26 m: the error comes from the assumed focal length (`camera_hfov_deg` set for the desktop webcam, the clips are from a laptop camera) and the assumed shoulder width.
- **Latency:** identity stage 2.6–6.8 ms median, 12.4–15.6 ms at the 95th percentile (frames where OSNet runs) on the GPU; on the desktop CPU 0.7 ms median and 27–31 ms when OSNet runs (23 ms median in *Demo game*, where extra people keep OSNet busy); 33 ms median live on the laptop.
- **Lighting:** dim clip: 100 % active, 0 switches. Synthetic darkening of *Desktop game*: no extra ids, players active 99–100 % (unchanged).
- **Re-identification threshold** (`identity.reid_threshold`, the cosine similarity a new track needs to get a stored player's id): swept from 0.4 to 0.8 on the labelled clips.

  | Threshold | *Crossing ×22*: switches, accuracy | *Re-entry*: players without an id, recovery | *Swap*: switches |
  | --- | --- | --- | --- |
  | 0.4 | 24, 69.6 % (16 of 22 crossings kept) | 4.3 %, 0.27 s | 19 |
  | 0.5 | 0, 100 % | 4.3 %, 0.27 s | 9 |
  | **0.6** (used) | 0, 100 % | 6.4 %, 0.72 s | 2 |
  | 0.7 | 0, 100 % | 8.5 %, 0.72 s | 0 |
  | 0.8 | 0, 100 % | 39.4 %, 0.99 s (OSNet re-identified returning players only after 7–14 s) | 0 |

  Too low, and a new track is linked to the wrong player (24 switches at 0.4); too high, and returning players stay without an id for seconds (0.8). The two-player clips are perfect from 0.5 to 0.7; 0.6 lies in the middle of that range, and 0.7 would trade slightly slower recovery for fewer switches with a bystander in view.

### Technical conclusion

The expected outcome is achieved for two players: through 26 crossings and 14 exits and re-entries both kept their identities in 100 % of 681 labelled player-frames, with no switch and no extra id, and returning players were recognised again within 0.3–0.7 s. The strongest evidence is the 22-crossing metronome clip, which stays perfect for any re-identification threshold from 0.5 to 0.7, so the result does not hinge on one tuned value. The main limitation is a third person in view: a newcomer who appears where a player just left inherits that player's id, because the BoT-SORT track carries over and OSNet never checks it, and a bystander can take a free player slot; depth estimates are also 1.1–1.8 m too large with the uncalibrated laptop camera. A justified improvement is to let the periodic OSNet check of linked tracks also unlink a track whose appearance no longer matches its own player (similarity below `reid_threshold` against every stored player) instead of only swapping two players' links: the newcomer's track is exactly such a case, and the check already runs every 0.5 s.

## 5. Tech-evaluation: Player interaction and game control

**Responsible student:** Pjotr van der Made (s3207447)

**Expected outcome:** At least two distinguishable actions per player produce reliable commands, meaningful multiplayer interaction, correct state updates, and clear visual feedback.

### Experimental design and protocol

Repeated actions by three subjects, with the start of every intended action hand-labelled: two subjects at about 2 m repeat each action on cue (*Punch reps* 11 punches, *Shield reps* 15 shields, *Beam reps* 14 hands-up beams; 30 fps); one of them tests the moves at the start of the *Laptop game* at ~5 fps, the demo laptop's frame rate (5 punches, 3 beams); a third subject was filmed with a phone (25 fps) punching at 1–3 m, holding T-poses (both arms straight out, a pose that must not fire a punch) and crossing his arms (*Phone moves 1*, *Phone moves 2*: 40 punches, 1 beam, 1 shield). In total 56 punches, 16 shields and 18 beams. Each clip is run through the game without the intro; `tools/eval_gestures.py` matches every labelled action with the first event of the same gesture and player from 0.2 s before (0.4 s for the clips labelled at a coarser frame step) to 1 s after its labelled start; events without a label are false triggers. Plus 10 solo shields (*Solo shields*) and free play (*Desktop game*). Impact factors: subject, distance, frame rate, the two players acting at once, light.

### Measured results

| Action | Intended | Recognised (recall) | False triggers (another gesture's move) | Precision | Start of move → event, median / p95 | → on screen |
| --- | --- | --- | --- | --- | --- | --- |
| Punch | 56 | 49 (88 %) | 6 (6) | 89 % | 159–280 / 325–472 ms | 180–303 ms |
| Shield (arms crossed) | 16 | 13 (81 %) | 11 (1) | 54 % | 217–320 / 619 ms | 246–344 ms |
| Beam (hands up) | 18 | 14 (78 %) | 0 | 100 % | 176–400 / 308–400 ms | 203–424 ms |

Per clip (recognised / intended, false triggers): *Punch reps* 9/11, 0; *Shield reps* 12/15, 6 shields + 1 punch; *Beam reps* 10/14, 4 punches + 1 shield; *Laptop game* moves 8/8, 1 shield; *Phone moves 1* 13/16, 2 shields; *Phone moves 2* 24/26, 1 punch + 1 shield. Solo (*Solo shields*): 10 / 10 shields.

- **Distinguishable actions:** the three actions are rarely confused with each other except in one direction: an arm swinging up into the hands-up pose passes through the punch pose, fires a punch (4 times in *Beam reps*), and the one-action-at-a-time rule then blocks the intended beam (all 4 missed beams). Holding a T-pose, where both arms are straight, never fired a punch (0 in 3 holds), because the punch rule needs the other arm bent.
- **A rule tried and dropped:** to stop those swing punches we tried a veto, *no punch while the wrist rose more than 0.8 shoulder widths in the last 0.2 s* (a minimum wrist speed would not help: the swings, 9.6–19 sw/s, were faster than the punches, 2.4–5.2 sw/s). It fixed the beams (18/18) but cost 8 of the third subject's 25 punches, which he throws upward from the hip: real punches rose 0.3–1.28 sw, false swings 0.89–1.55 sw, so no threshold separates them. Over all labelled clips, punches 41/56 with the veto against 49/56 without; we kept the higher punch recall and switched the veto off (`gestures.punch_max_rise_sw: 0`).
- **False shields:** in *Shield reps*, a held shield firing a second time after a short keypoint dropout (4), arms crossed low at the waist while walking (2); in the other clips, arms crossed or folded in front of the body while walking, turning or between two moves (5).
- **Missed actions:** shields crossed high at the shoulders fail the chest-zone rule (all 3 misses); punches: 7 not detected (2 in *Punch reps*, 5 in the phone videos).
- **Command latency:** 159–280 ms median from the start of the move to the event at 25–30 fps: mostly the movement itself (the rule fires when the arm is straight or the arms are crossed), plus the 0.03 s confirmation for shield and beam and 10–18 ms of smoothing; the frame with the event is on screen ~25 ms later. At the laptop's ~5 fps every action was still recognised (8/8), but 585–600 ms after its start: one frame is 0.1–0.45 s there.
- **Two players at once:** actions of both players within 0.5 s: 7 / 10 trials with both recognised (the misses are the beams blocked by a swing punch, and one shield).
- **Lighting:** synthetic darkening of the *Desktop game* raised the shield events from 5 to 14 and blocked 6 actions because the face was not found (*not facing*), against 0 in normal light; punch events stayed at 16.

### Technical conclusion

The expected outcome is achieved for the punch and the beam and partly for the shield: over 90 labelled actions by three subjects, punches are recognised with 88 % recall and 89 % precision, beams with 78 % recall and no false trigger, shields with 81 % recall but only 54 % precision, and every action still works at the demo laptop's ~5 fps. The strongest evidence is the labelled repetitions across subjects, distances and frame rates, in which each false trigger and miss could be traced to a cause, and which also showed that a rule tuned on two subjects (the rising-wrist veto) failed on a third. The main limitation is that the rules judge the pose frame by frame: an arm swinging up through the punch pose fires a punch and blocks the beam, and the shield fires on low or folded crossed arms and twice after a dropout. A justified improvement is a short decision window for the punch: wait ~0.15 s after the punch pose appears and drop it if the arm keeps rising towards hands up, which would remove the swing punches (and so the blocked beams) for every punching style at the cost of 0.15 s latency, instead of a fixed threshold on one feature.

## 6. Tech-evaluation: Scene processing and real-time integration

**Responsible student:** Ellen Zheng (s3502155)

**Expected outcome:** All five tasks operate as one responsive multiplayer game with meaningful scene changes, measured end-to-end performance, and controlled recovery from difficult conditions.

### Experimental design and protocol

The full game (all stages, render included, no intro) on the gameplay clips on the desktop PC, ~12 min in total (*Desktop game*, *Solo shields*, *Demo game*, *Crossing ×22*, *Laptop game*), with CPU, memory and GPU sampled twice a second; *Desktop game* and *Demo game* again with everything on the CPU, as at the demo; and the game's own per-frame log of the live demo on the laptop (100 s, CPU only, older game version). `tools/eval_pipeline.py`: frame time from the frame entering the pipeline to the drawn view (`ms_total`), per-stage times, freezes (frames > 200 ms), failing stages.

### Measured results

- **End-to-end frame rate:**

  | Run | Mean FPS | 5 % low | 1 % low | Median / p95 frame time |
  | --- | --- | --- | --- | --- |
  | Desktop, GPU (RTX 3070 Ti), 4 gameplay clips | 24.3–33.0 | 19.7–23.4 | 16.6–21.3 | 29–39 / 43–51 ms |
  | Desktop, CPU only (Ryzen 7 5700G), 2 gameplay clips | 11.0–11.2 | 8.4–8.6 | 7.2–7.4 | 86–92 / 116–119 ms |
  | Laptop, CPU only (Ryzen 7 4800H), live demo, older version | 7.6 | 6.3 | 5.6 | 131 / 160 ms |

  Live, the laptop delivered 6.7 frames per second to the game (*Demo game*) and 2.4 during the *Laptop game* game. Where the time goes (median ms per frame):

  | Stage | Desktop GPU | Desktop CPU only | Laptop CPU (live) |
  | --- | --- | --- | --- |
  | Pose | 10 | 49–52 | 77 |
  | Identity (p95, OSNet) | 0.6 (13) | 0.7–23 (27–31) | 33 (40) |
  | Face | 11.5–14 | 15 | 22 |
  | Render | 4–11 | 10–13 | 8 |
  | Motion, gestures, game | ≤ 0.1 each | ≤ 0.1 each | ≤ 0.25 each |

  On the CPU the pose model takes over half of the frame.
- **Resources (desktop):** with the GPU 1.3–1.6 CPU cores, 2.3–2.4 GB RAM, GPU 10–18 % busy, 1.4–1.6 GB GPU memory (106–120 MB of it PyTorch); CPU only 5.9–6.3 cores busy (of 16 threads), 1.3 GB RAM.
- **Long-run stability:** 0 failed frames and no fallback to a stub in ~13 min of play (all clips and the live log); 1–4 frames over 200 ms per run (longest 717 ms on the laptop, 590 ms on the desktop).
- **Start-up:** 0.5 s imports + 3.7 s model loading and warm-up with the GPU (1.4 s CPU only, no CUDA initialisation); the first frame is ready about 4.3 s (2 s) after start. A new round (key r) and a full reset (R) keep the loaded models.
- **Fast motion:** wrist speeds still tracked: 99th percentile 4.9–10.7 shoulder widths/s (about 2–4 m/s with a 0.39 m shoulder width), above the 3.3–8.2 sw/s measured just before punches.
- **Lighting:** synthetic darkening leaves the frame time unchanged (35.9 vs 36.9 ms median); its effect is on detection (Tasks 1, 4). Live, dim light did lower the frame rate: the laptop camera dropped from 15 to 8.5 fps in the dim clip.

### Technical conclusion

The expected outcome is achieved on the desktop with the GPU (24–33 FPS, 29–39 ms per frame, no failed frames in about 13 minutes, start-up in 4.3 s) but only partly in the demo configuration: everything on the CPU runs at 11 FPS on the desktop and ran at 7.6 FPS live on the laptop (older version), below the camera's 15–30 fps. The strongest evidence is the per-stage timing in the three configurations: the pose model alone takes 49–77 ms per frame on a CPU against 10 ms on the GPU, more than half of every CPU frame, while all game logic stays under 0.3 ms. Robustness held up: no stage failed or fell back to a stub in any run, and the frame time was unaffected by light. A justified improvement for the CPU demo is to export the pose model to ONNX/OpenVINO and run it at a smaller input size (e.g. 480 instead of 640 px, roughly halving the pixels), rather than switching to the nano model, which recognised only 3 of 10 crossed-arm shields against 9 of 10 for the small model.

## 7. Reflection: limitations and future improvements

### Limitations of the system

- **Gestures are judged frame by frame.** An arm swinging up to hands-up passes through the punch pose and fires a punch, which blocks the beam (4 of the 4 missed beams). A rising-wrist veto fixed this for two players but cost a third player 8 of 25 punches, and a minimum wrist speed would not help either (the swings are faster than the punches); no single threshold on the movement separates them.
- **Shields depend on clothing.** The shield works well when the forearms are bare or clearly visible, but badly with bulky or dark-coloured clothes: the forearms then barely stand out against the torso, and with the hands on the opposite shoulders the pose model places the wrists on their own side, so the rule (wrists in swapped order, in the chest zone) fails. Rules that ignore the left/right order caught every attempt on the *Solo shields* but also fired on guard fists and running; MediaPipe's hand model found no hand in 6 of 10 such crosses. The small pose model (9 of 10 caught) replaced nano (3 of 10). A held shield can also fire twice after a short keypoint dropout.
- **Facing the camera** is checked from the shoulder width against the calibration, but crossing the arms or raising them narrows the shoulders too (22–38° read while facing the camera), so the body's turn is only taken from frames with the arms down.
- **A third person.** A newcomer stepping into a leaving player's place inherits the id (the BoT-SORT track carries over, OSNet never checks it), and a bystander in view can take a free player slot.
- **Uncalibrated camera.** Depth (`Player.pos`) and the head pose assume a 70° field of view and a 0.39 m shoulder width; on the laptop camera depth reads 1.1–1.8 m too far, while being precise (IQR 0.06–0.3 m). A per-camera calibration mode was designed and dropped because every member develops on a different webcam. No game rule uses the depth, so the game is unaffected; it is the spatial output the brief asks for.
- **CPU-only demo.** Without CUDA the laptop runs at 7.6 FPS and the desktop at 11 FPS: the pose model takes over half of every frame. Gestures still work at ~5 fps, but every action then reacts 0.6 s after the move starts.
- **The orb is grabbed by accident**: a wrist within 0.5 shoulder widths of it for 0.1 s grabs it, also a hand passing it during a shield.
- **Robustness.** A failing module is skipped, and identity, motion and face fall back to their stubs after 1 s; pose, gestures and the game have no stub, so while they fail there are no players or no actions.

### Limitations of the evaluation

- **Few subjects:** each test used two or three people, mostly group members (three subjects in the labelled gesture clips); the thresholds were tuned on clips that are also part of the evaluation, so the numbers are optimistic for new players (the veto's failure on a third player shows how much).
- **Ground truth made with an AI assistant**, checked frame by frame, with the phone videos labelled at a coarser frame step (0.2–0.4 s); keypoint accuracy was not measured (no hand-marked keypoints).
- **Lighting:** besides one dim clip, low light was simulated (darkening plus noise); a real dark room also brings motion blur and the camera's lower frame rate (15 → 8.5 fps in the dim clip).
- **Latency** is measured from the frame entering the pipeline; the camera's own exposure and transfer time is not included.

### Future improvements

- **Punch decision window:** wait ~0.15 s after the punch pose appears and drop it if the arm keeps rising towards hands-up; or a small classifier (logistic regression or random forest on wrist and elbow positions and speeds in shoulder widths) trained on the labelled clips (56 punches, 18 beams, 16 shields), tested on a held-out player.
- **Shield:** raise the top of the chest zone (all 3 missed shields were crossed at shoulder height), and use the elbows and forearm direction instead of the wrists alone.
- **Identity:** let the periodic OSNet check unlink a track that matches no stored player, and accept new players only in the start places.
- **Camera:** calibrate the demo camera with a checkerboard (`cv2.calibrateCamera`) and load its intrinsics when present.
- **Speed:** export the pose model to ONNX/OpenVINO and run it at a smaller input size for the CPU demo; run the person segmenter every other frame (~4 ms).
- **Smoothing:** a constant-velocity Kalman filter to predict fast wrists through dropouts instead of holding them still.
- **Game features designed but not built:** a final smash of paint rain that lasts until the caster blinks (the blink detector hardly reacted at play distance, ~2.3 m); a rule for both players attacking at once (higher attack level wins, otherwise a draw); dodging a paint blob by stepping ~0.5 m towards the camera, which needs a calibrated camera and a faster depth signal.
- **Checks still open:** game balance over real rounds; webcam checks of the game start, the drinks and the face effects with two players.
