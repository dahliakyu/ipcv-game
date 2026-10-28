Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. Feature 1 is being built; the others are not started. Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

## The merge is done

The features build on the Colour Clash game, merged from the teammates' code (originals in `garb/rest/`, local only). Details and reasons are in [merge_log.md](merge_log.md):

1. **Gestures (Task 4, with Task 2 smoothing):** `interaction/gestures.py` sends `punch`, `hands_up` and `arms_crossed` events (merge log, Task 4, steps 1–4).
2. **Game logic (Task 4):** `interaction/game.py` runs the game from those events (Task 4, steps 5–9 and Task 4 modifications).
3. **Identity (Task 3):** `identity/tracker.py`, BoT-SORT on the pose boxes and OSNet re-identification (Task 3).
4. **Rendering (Task 5):** `scene/renderer.py` draws the game (Task 5: Renderer, and Task 5 modifications).

### Where things are

| To change | Code | Config | Tests |
| --- | --- | --- | --- |
| Gesture rules and events | `interaction/gestures.py`: `detect_*()`, `GestureRecognizer` | `gestures:` | `tests/test_gestures.py` |
| Game rules: actions, hits, damage, round end | `interaction/game.py`: `Game.update()` → `_act()` (events to actions, `facing()`) → `advance()` (movement, hits, `_apply_damage()`); state in `GamePlayer`, `Projectile`, `PaintBeam`, `Shield`, `PaintSplat` | `game:` | `tests/test_game.py` (`frame_state()` builds fake frames) |
| How the game looks | `scene/renderer.py`: `Renderer.draw()` (order: camera, paint, `draw_game()`, face effects, health bars, `_draw_hud()`, `_draw_game_over()`); sprites in `scene/fire_bolt.py`, `magic_beam.py`, `fire_ring.py` (frames loaded by `scene/animation.py`, `load_frames()`); helpers in `scene/draw.py` (`player_color()`, `blend_rgba()`, `shift_hue()`) | `render:` | `tests/test_renderer.py` |
| Face effects and face data | `face/effects.py` (`FaceEffects.draw()`, stateless), `face/tracker.py` (`Face`, anchors), `face/landmarker.py` (blendshapes) | `face:` | `tests/test_face.py`, `test_effects.py` |
| Players and identity | `identity/tracker.py`; `Player.status`, `reentered` | `identity:` | `tests/test_identity.py` |
| Keys, pipeline order | `app/main.py` (`q`, `p`, `r` new round, `R` full reset), `app/pipeline.py` | | `tests/test_pipeline.py` |

Each module reads only its own config section; the renderer reads game timings from `Game` (`beam_charge_s`, `shield_opacity()`, `torso()`, …), not from `game:`. Try a change with `python -m ipcvgame.app.main --source data/clips/session.mp4` (a full recorded round), `--until game` for the logic as plain shapes, and `--record-view <clip>` to save what the window shows. New CSV columns go in a module's `log_fields()`.

### What the face stage already gives you

`state.faces[pid]` (see [merge_log.md, Task 1](merge_log.md#task-1-face-tracking-and-ar)):

| Field | Use |
| --- | --- |
| `anchors["above_head"]`, `["hat"]`, `["forehead"]`, `["mouth"]` | Where to draw; they follow head tilt and turn |
| `scale` | Face width in px: size effects with it, so near and far players look the same |
| `roll`, `yaw` | Rotate sprites with the head; squash them by cos(yaw) when the head turns |
| `alpha` | Fade every face-anchored effect with it, so effects disappear smoothly when a face is lost |
| `blendshapes` | MediaPipe scores 0–1: `jawOpen`, `eyeBlinkLeft`/`Right`, `mouthSmileLeft`/`Right`, `cheekPuff`, … Empty when not measured. On our mirrored frame, `…Left` is the player's own right side |

## Features

Renumbered on 2026-10-28, with the game start first. Old numbers in commits and `merge_log.md` map as follows:

| Now | Was |
| --- | --- |
| 1. Game start | 12 |
| 2. Face effects | 1 (team colour), 2 (hit reaction), 3 (status above the head), 7 (cheek bulge) |
| 3. Final smash | 4 (super charge, reset by blinking), 15 (orb in four parts), 16 (paint from above) |
| 4. Consumables | 6 |
| 5. Aiming the normal attack | 11 |
| 6. Normal attack as splashed paint | 13 (paint stains where the hit landed) |
| 7. Both players attacking at once | 14 |

Dropped: 5 (taunt), 8 (learned punch classifier: the gestures stay rule-based; its limits are in [limitations.md](limitations.md)), 9 (charge bar: the orb unlocks the final smash) and 10 (weapon pick-ups).

### 1. Game start: instructions and pose calibration

**Status.** In progress (the "Feature 12:" commits; it was feature 12 until the backlog was renumbered on 2026-10-28). Steps 1–4 are done; see the implementation steps below.

**What.** A game UI around the round: a start screen, a short instruction and calibration phase in which each player learns and performs the three moves, a countdown, and a result screen. Today the game starts the moment the players are seen.

**Owners.**
- Task 4: the game phases in `interaction/game.py` (a state machine) and the calibration rules.
- Task 5: the screens in the renderer.
- Task 3: storing each player's appearance during calibration.
- Task 2: body measurements from the keypoints.

**Design.**

- **Phases:** `waiting` → `calibration` → `countdown` → `playing` → `result`. Damage, bolts and beams only count in `playing`.
  - **Play again** (decided 2026-10-28): from `result`, both players with hands up for ~2 s (or key `r`) go straight to `countdown`. The same players keep their calibration, so they do not see the instructions again.
  - **Back to `waiting`:** key `R` (full reset). Identity then forgets the players too; during a round no new player can appear, because identity keeps P1 and P2 until a reset.
- **Waiting.** "Player 1: stand on the left / Player 2: on the right". A start place is ticked in its team colour once someone stands in it: a player identity sees (`status == "active"`) with a visible face, in that half of the frame (decided 2026-10-28). The tick goes by position, not by player number, because identity numbers players in the order it first sees them.
- **Instructions with calibration.** One move at a time per player (decided 2026-10-28), on a panel on the player's own half of the frame: first the neutral pose (below), then punch, hands up and arms crossed. Each move shows a small picture of the pose and its effect, and is ticked off when its gesture event arrives. A move not done within ~8 s flashes as a reminder. Nothing fires during this phase, so it doubles as a practice round.
- **Pose calibration** while the players stand in a neutral pose, facing the camera, for ~2 s:
  - shoulder width and arm length per player (median over the frames), so gesture thresholds in shoulder widths can be checked per player, e.g. a short arm still reaches `punch_min_reach`;
  - the shoulder width at this distance is the reference for `Player.pos` depth (pinhole model);
  - a clean OSNet vector per player (standing apart, front-facing): it replaces the first-sight vector, which may be from walking in at an angle.
- **Countdown.** 3-2-1 in the middle of the frame; then `playing`.
- **Result.** Winner in their team colour, final health, "hands up to play again". The renderer's end panel (`_draw_game_over()`, dimmed frame) is the starting point, and the top bar already shows "STEP INTO VIEW" while nobody is seen (`_draw_hud()`).

**Check.** Two new players get through the instructions in under ~30 s; a player who does not perform a move is reminded; the calibration values are logged per player (`log_fields`) and stay stable when the round restarts.

**Implementation steps.** One step at a time, each checked by the Task 5 owner before the next. Every new number goes in the module's config section with a reason.

1. **Game phases** (Task 4, `interaction/game.py`). *Done.*
   - `Game.phase` with the five phases and the time since the phase started.
   - Transitions:
     - `waiting` → `calibration`: both start places taken for ~1 s (`place_taken()`: someone active with a face, by shoulder midpoint, in the left or right half).
     - `calibration` → `countdown`: both players have done every move.
     - `countdown` → `playing`: after 3 s.
     - `playing` → `result`: a winner.
     - `result` → `countdown`: both players hold hands up for ~2 s.
   - In `calibration`, gesture events tick off the player's current move instead of firing; `advance()` runs only in `playing`.
   - Keys: `r` = new round, into `countdown`; `R` = everything reset, into `waiting`.
   - `game.intro: false` (or `main.py --no-intro`) starts in `playing` as before, for the tests and clips recorded without the intro.
   - `game.round_players: 1` (or `main.py --solo`) tests the intro with one player: only P1 is waited for, calibrated and asked for the replay. A solo round has no opponent, so it never ends; use `r` or `R`. Steps 2–5 keep this working: the screens show only P1's place and panel, and the P1-left swap is skipped.
   - The phase is shown in `--until game` and logged as `game_phase`; the next calibration move as `game_p<pid>_move`.
   - As built: `Game.phase`, `phase_elapsed()`, `current_move(pid)`, `new_round()` (key `r`) and `reset()` (key `R`). `GamePlayer.move` / `move_t` hold each player's calibration progress. Holds (`ready_s`, `replay_hold_s`) survive short dropouts (`hold_gap_s`). Hands up for the replay uses the gesture rule `detect_hands_up()` per frame, because an event only marks its start. A ticked move is logged as the action `learned_<move>`.
2. **Waiting, countdown and result screens** (Task 5, `scene/renderer.py`). *Done.*
   - Waiting: the start places (`start_place()`), dashed with "STAND HERE" until taken, then solid with a tick and "READY". With one player, a single place in the middle.
   - Countdown: 3-2-1 in the middle, each number settling over its second; "FIGHT!" for `fight_s` once the round starts.
   - Result: `_draw_game_over()` with each player's final HP in their colour, and "HANDS UP TO PLAY AGAIN" with a bar that fills during the hold (`Game.hold_fraction()`).
   - The top bar says what the phase expects (`status_text()`). Health bars are shown from the countdown on.
   - New `render:` keys: `start_place_w`, `start_place_y`, `start_text_scale`, `countdown_h`, `fight_s`.
3. **Instruction panels** (Task 5, `scene/renderer.py`, `scene/instructions.py`). *Done.*
   - Per player, at the bottom of their half (`instruction_box()`), over the legs so the face, arms and hat stay visible. One panel in the middle with `--solo`.
   - A stick figure of the current move: fixed joint positions in shoulder widths, drawn with OpenCV lines (`draw_figure()`), mirrored for P2 so the punch points at the opponent. A test checks that each figure, as keypoints, passes the gesture rule of its own move and no other.
   - The move's name in the team colour, how to do it and its effect (`MOVE_TEXT`), and one dot per move (done, now, to do).
   - "NICE!" with a tick for `nice_s` after a move, "DONE!" when all are done. After `remind_s` without the move, the frame flashes and "TRY IT NOW!" appears.
   - The neutral pose already has a figure and text, ready for step 4.
   - Cross arms (changed 2026-10-28): the panel teaches wrists crossed just past the middle of the chest ("CROSS WRISTS IN FRONT OF CHEST"). With the hands on the opposite shoulders, the rule missed 7 of 10 attempts on `data/clips/shield.mp4`; see [limitations.md](limitations.md).
   - New `render:` keys: `instruction_w`, `instruction_y`, `instruction_alpha`, `instruction_title_scale`, `instruction_text_scale`, `remind_s`, `remind_hz`, `nice_s`.
4. **Pose calibration** (Task 2 measurements, Task 4 rules). *Done.*
   - The neutral pose is the first calibration move (`CALIBRATION_MOVES`), checked by the game every frame (`Game._neutral_pose()`). It needs:
     - the player in view with the head turned at most `neutral_max_yaw_deg`;
     - shoulders, elbows, wrists and hips seen;
     - both wrists at least `neutral_wrist_below_sw` below the shoulder line, and slower than `neutral_max_speed_sw`.
   - Held for `neutral_s` (gaps up to `hold_gap_s`), it stores the median shoulder width and arm length (shoulder–elbow–wrist) of one sample per frame (`pose/measure.py`: `shoulder_px()`, `arm_px()`) in `GamePlayer.shoulder_px` / `arm_px`. A hold that starts over drops its samples.
   - Kept on a new round, cleared by `reset()`. `Game.arm_sw()` gives the arm in shoulder widths; under `min_arm_sw` (`short_arm()`), the punch panel adds "STRETCH YOUR ARM ALL THE WAY".
   - Panel: a bar fills while the pose is held (`neutral_fraction()`), then "NICE! MEASURED".
   - Logged as `game_p<pid>_shoulder_px`, `_arm_px` and `_arm_sw`, and shown in `--until game`.
   - On `session.mp4`: the arms measured 1.67 and 1.56 sw. With lowered arms, wrists moved at a median 0.34 sw/s (90th percentile 0.81, under the 1.0 limit), and the head was turned less than 14° in 90% of frames.
5. **Identity at calibration** (Task 3, `identity/tracker.py`, wired in `app/pipeline.py`).
   - P1 becomes the player in the left place when calibration starts (the links are swapped if needed; follow-up "P1 / P2 order"), so the place colours and the player colours match from then on.
   - The OSNet vector taken during the neutral pose replaces the first-sight vector.
6. **Depth reference** (Task 3, `Player.pos`): depth relative to the calibration distance, from the calibrated shoulder width (pinhole model: depth ∝ 1 / shoulder width).
7. **Docs and webcam check.** Update the code map and the README keys; the Task 5 owner runs the full intro with two players on the webcam.

### 2. Face effects: team colour, hit reaction, status above the head, cheek bulge

**What.** Effects on and above each player's face that show who is who and what happens to them: the team colour on the face, a reaction to every hit, a name tag and status icons above the head, and puffed cheeks after eating. They share the face anchors and one per-player animation state.

**Owners.**

- Task 1: cheek anchors, the hat band, the hit animation and the cheek warp in `face/effects.py`.
- Task 4: the `hit` event in `interaction/game.py`; the `bite` event (feature 4).
- Task 5: the colour table and the status icons in the renderer, reading game state.

**Design.**

- **Shared: animation state.** `FaceEffects.draw` is stateless today. The hit reaction and the cheek bulge need state over time: the renderer passes the game's events and `frame.t`, and `FaceEffects` keeps one animation record per player, reset when `Player.reentered` is True. Every effect fades with `alpha`, sizes with `scale` and turns with `roll` and `yaw`. Use time, not frames.
- **Team colour,** so it is always clear who is who.
  - **One colour table.** Done in the game merge: `PLAYER_COLORS` in `scene/draw.py` (P1 red, P2 blue, the colours from `game.py`) is the only table; use `player_color(pid)`.
  - **Hat band.** Recolour the band pixels of `hat.png` once at load, one copy per team: select the red band by hue, shift its hue to the team colour, keep the shading.
  - **Cheek anchors.** Add `left_cheek` and `right_cheek` as 3D points on the head model, about (±45, 30, −5) mm, projected like the hat. They follow tilt and turn for free. The cheek bulge uses them too.
  - **War paint.** Two short stripes on each cheek at the cheek anchors, rotated by `roll`, squashed by cos(`yaw`) on the cheek turned away.
- **Hit reaction:** the hat flies off and the face flashes; the hat comes back after a moment.
  - **Hit event.** `Game` emits a `hit` event (pid, direction, damage) when `_apply_damage()` runs.
  - **Hat knocked off.** The hat detaches from its anchor with a velocity away from the attacker plus upwards (~1.5 face widths/s), spin (~360°/s) and gravity. It fades out over 0.6 s, then pops back on the head with a short scale-up (0.2 s).
  - **Flash.** For 150 ms, blend white then the attacker's team colour over an ellipse at `center` (size from `scale`, rotated by `roll`).
  - **Final smash hits** (feature 3) can also add a short shake of the hat anchor.
- **Status above the head:** a name tag and icons, so the state is visible on the player. The health bars stay pinned to the bottom corners (Task 5 modifications: a bar that moved with the player was too busy).
  - **Name tag.** "P1" at `above_head`, small, in the team colour; size from `scale` with a minimum so far players stay readable.
  - **Dizzy stars.** Below 25% HP: 3 stars orbit `above_head` on an ellipse in the head's plane (flattened by cos(pitch), turned by `roll`).
  - **Crown.** The player with more HP gets a small crown above the tag.
- **Cheek bulge:** after eating (the `bite` event, feature 4), the cheeks puff up for a second, chipmunk style.
  - **Warp.** A real image warp, not a sticker: a radial "bulge" (magnifying) warp around each cheek anchor with `cv2.remap`. Inside radius r (~0.3 face widths), the source radius is ρ' = ρ · (ρ / r)^k with k rising from 0 to ~0.5 and back over 1 s. A good image-processing technique to explain in the report.
  - **Cost.** Warp only the two small cheek boxes, not the whole frame.
  - **Optional.** The player can also puff their own cheeks: MediaPipe's `cheekPuff` > 0.5 shows the same bulge.

**Check.**

- Both players on the crossing clip keep their own colour; the stripes stay on the cheeks when the head turns.
- One knock-off per hit, and the hat always comes back; two hits in a row don't stack two hats.
- The icons stay above the hat (also with the chin up) and fade out with the face.
- The bulge stays on the cheeks while the head moves and turns, and ends cleanly; ~1 ms extra per face.

### 3. Final smash: a paint orb in four parts, raining until the caster blinks

**What.** The major attack becomes a final smash, like the Smash Ball in Smash Bros. Players collect a paint orb in four parts. With all four, hands up starts the attack: the orb flies up, bursts, and paint rains down on the opponent. **The rain lasts only while the caster does not blink:** the first blink ends it, and the orb is used up, so the player has to collect four new parts. Attacking becomes a staring contest that the other player can try to break. The opponent has two defences: getting out from under the direct paint drop, which is possible but very hard, or the shield. It replaces the magic beam.

**Owners.**

- Task 4: orb parts, the attack's timing, area and hits in `interaction/game.py` (replaces `PaintBeam`); the `blink` event in `interaction/gestures.py`.
- Task 1: the blink signal (already there: `blendshapes`) and the eye glow in `face/effects.py`.
- Task 5: part sprites, the orb icon in the HUD, and the rise, burst, drops and splashes in the renderer.

**Design.**

- **Orb parts.**
  - **Spawning:** one part at a time at a random free spot in the upper half of the frame (reachable with a raised hand), every ~5–8 s while fewer than four parts are in play; it slowly bobs.
  - **Collecting:** a wrist (smoothed keypoint, `state.motion`) within ~0.5 shoulder widths of the part, held for ~0.2 s, picks it up, as for the items in feature 4.
  - **Holding:** each player's parts show as a 4-slot orb icon by their health bar, in their team colour; the quarters fill in as they are collected. A player cannot hold more than four.
  - **Dropping:** 3 blob hits (feature 6) in a row knock one part out of the player; it flies off and lands as a part to collect again, so the other player can steal it.
- **Start.** Without all four parts, `hands_up` does nothing: the orb icon flashes "not ready", and the one-action-at-a-time reset still applies. With all four, the complete orb glows above the player's head, and `hands_up` starts the attack. The slots empty at the start: the orb is used up whether the rain lasts a second or much longer.
- **Duration: until the caster blinks.** Blinks count only during the attack. The rain stops at the first blink after the start, and also when the caster's face is not measured (`hold`, or `alpha` < 1), so turning away cannot hide a blink. A cap (~6 s, in config) keeps one attack from deciding the round alone. Damage per drop is fixed, so a longer stare means more damage.
- **Blink detection.** The raw score is not enough: on the test clips, `blink` = mean of `eyeBlinkLeft`/`Right` went over 0.5 in up to 26% of frames just because people looked down. Use:
  - a per-player baseline: the running median of the score over the last ~3 s, which follows lighting, glasses and eye shape;
  - a blink that starts at score > baseline + 0.35 (and > 0.5), and ends at score < baseline + 0.15 (hysteresis);
  - a duration of 60–500 ms (longer = eyes closed, not a blink; closed eyes also end the rain);
  - looking down (pitch < −20°, see `log_fields` `pitch_deg`) does not count as a blink.
- **Measurement rate.** MediaPipe measures each player at ~15 Hz (`face.landmarker_per_frame: 1`). A 100 ms blink can fall between two measurements, and a missed blink makes the rain last longer. During an attack, measure the caster every frame (as `landmarker_per_frame: 2` does, +~8 ms).
- **The attack, in phases:**
  1. **Burst:** the orb flies up out of the top of the frame (~0.3 s). The frame dims for a moment with a flash in the caster's colour, as Smash Bros does for a final smash.
  2. **Warning:** a target mark on the floor under the opponent (~0.4 s).
  3. **Rain:** paint drops fall over the drop zone, densest in the middle, until the caster blinks.
  4. **Splashes:** splash marks fade on the floor.
- **Drop zone and dodging.** The zone is centred on the opponent's body box, ~2 box widths wide, and follows the opponent slowly: it moves towards their current position at a limited speed (~0.3 frame widths/s). A player standing still takes the whole attack. Getting out of the direct drop needs a quick, long step to the side, faster than the zone follows; the zone then catches up again, so dodging works only now and then. Zone width and follow speed go in config and are tuned on the webcam.
- **Hits.** Each drop tests its path since the last frame against the target's box, as the blob does. Every hit adds a splash on the body (feature 6) and a little damage (e.g. 1 per drop, ~5 per second of rain; the beam did 12 per attack). The shield blocks the drops while it is up. It lasts `shield_s` (1 s), so it covers part of a long rain, not all of it.
- **Look.**
  - **Eye glow:** while the rain lasts, the caster's eyes glow in their team colour, so both players can see what ends it. On the blink, the glow shatters into particles, with a short "blink!" text at `above_head`.
  - **Rain:** drops in the caster's team colour, falling with gravity, slightly random sizes; splash sprites where they hit the floor (the bottom of the body box, or the frame bottom) or the player.
- **Replaces the beam.** `hands_up` no longer fires the magic beam. The `beam_ball` / `beam_band` drawing goes; the magic ball frames can stay for the burst. The calibration panel (feature 1) then shows hands up as the final smash, not "MAGIC BEAM".

**Check.**

- Orb: parts only disappear when collected or used; a player cannot hold more than four; a dropped part can be picked up by either player; `hands_up` does nothing without four parts.
- Blink: on the webcam, a deliberate blink always ends the rain; looking down at the floor for 2 s does not; turning away ends it. Log blinks and the rain's length per attack, and count false blinks on the clips.
- Attack: the rain follows the opponent at the set speed; standing still takes every drop, a quick long step out avoids some; the rain never lasts past the cap; the shield blocks the drops while it is up.

### 4. Consumables, Smash Bros style

**What.** Items drop from the sky during the round. A player can **eat** a falling item by catching it with their mouth (chomp), or, once it has landed, **grab** it from the floor with a hand. Each item has an effect tied to the existing mechanics. Power items raise the attack level, which decides who gets the shot when both players attack at once (feature 7).

**Owners.**

- Task 4: spawning, physics, pick-up rules and effects in `interaction/game.py`; the `bite` event in `interaction/gestures.py`.
- Task 1: mouth state (`jawOpen`, `anchors["mouth"]`).
- Task 2: wrist and ankle keypoints.
- Task 5: item sprites.

**Design.**

- **Spawning.** Every 6–10 s (random), one item appears at a random x over the play area, above the top of the frame, and falls with gravity (~1 frame height per 1.5 s at the end).
- **Eating.** The mouth is open when `jawOpen` > 0.5, and closed again below 0.25 (hysteresis).
  - An item that enters a circle of ~0.6 face widths around `anchors["mouth"]` while the mouth is open is "in the mouth": it stops falling and sticks to the mouth.
  - Closing the mouth eats it (`bite` event). An item not eaten within 1 s drops again.
  - The mouth anchor follows head tilt, so players can lean to catch items.
- **Floor.** The floor line is the larger y of the players' ankle keypoints, smoothed. A landed item lies there for 5 s, blinks for the last second, and disappears.
- **Grabbing.** A wrist keypoint within ~0.5 shoulder widths of the item for 0.2 s picks it up (crouching works naturally). Grabbed items take effect at once.
- **Item types**, using the existing mechanics:

  | Item | Effect |
  | --- | --- |
  | Paint thinner | +20 HP and removes some paint splats |
  | Shield snack | an immediate shield (`activate_shield`) |
  | Chili (power item) | +1 attack level for ~15 s, up to 3: more damage per blob, and the blob wins a clash (feature 7). Another chili adds a level and restarts the time |
  | Rotten item | splashes the eater with paint (−10 HP): a risk when grabbing blindly |

- **Attack level** (`GamePlayer.attack`, 1 without items): shown as small flames by the health bar, one per level, fading in the last 2 s.
- **Feedback.** Eating triggers the cheek bulge (feature 2); grabbing a short sparkle at the wrist.
- **Sprites (candidate):** `assets/Fancy Drinks/` (64 × 64 and 128 × 128 drinks and glasses, by Savannah; its Read Me states no licence). Not in git yet; add it with a README credit before using it.

**Check.** Items can be eaten and grabbed on the webcam; an item cannot be taken twice; mouth movements while talking do not eat items that are not near the mouth.

### 5. Aiming the normal attack: a pointing laser and paint blobs towards the opponent

**What.** The normal attack (a punch fires a paint blob, feature 6) no longer flies only sideways: it goes where the player aims, towards the opponent. A thin pointing laser from the hand shows the aim before firing. Only the normal attack is aimed; the final smash (feature 3) falls on the opponent and needs no aim.

**Owners.**

- Task 4: aim direction and 2D movement and hits in `interaction/game.py`.
- Task 2: arm direction from the keypoints.
- Task 5: the pointing laser and the blob drawn along its direction.

**Design.**

- **Aim direction.** The punching arm's direction, shoulder → wrist (`punching_hand()` already finds the arm). Aim assist: if the arm points within ~20° of the opponent's torso (shoulder midpoint to hip level, `body_position()`), snap to it, so a roughly correct punch hits.
- **Towards the opponent, not the middle.** Done for sideways attacks (Task 4 modifications): `Game.facing()` fires towards the nearest other player in view. What is left is the 2D direction below.
- **2D game logic.**
  - `Projectile` gets a velocity vector instead of `direction`.
  - Hits test the path since the last frame (a segment) against the target's body box in 2D.
  - The splash (feature 6) uses the 2D impact point.
- **Pointing laser.** While an arm is extended (straight but not yet a punch), a thin dashed line in the team colour runs from the wrist along the aim direction until it meets a body box or the frame edge. It fades with the gesture's state, and it is drawn only for the aiming player, to keep the frame readable.
- **Drawing.** The blob's trail follows the velocity vector, so the drawn blob needs no rotated sprite.

**Check.** Blobs hit an opponent who stands higher or lower (e.g. crouching) or in the same half of the frame; the pointing laser matches where the blob actually flies; aim assist does not snap to a player behind the shooter.

### 6. Normal attack as splashed paint that covers the body

**What.** A punch fires a blob of paint instead of the fire bolt. Where it hits, the paint splashes onto the opponent's body and stays there. Both are drawn by an algorithm, not from sprites: every splash has its own shape, and over a round the paint covers more and more of the body, so the damage shows on the player.

Today the game keeps each splat as a fraction of the body box, inside a fixed band (`splat_band`, 0.38–0.62), and the renderer stretches that band over the torso. So a hit on the shoulder or the stomach lands near the middle of the shirt, and the splat moves on the torso when the box changes shape after the hit.

**Owners.**

- Task 4: the blob's flight and the impact point in body coordinates (`PaintSplat`) in `interaction/game.py`.
- Task 2: the body keypoints for the body mask.
- Task 5: drawing the blob, the splashes and the paint layer in the renderer; this replaces `scene/fire_bolt.py` for the normal attack.

**Design.**

- **The blob in flight.** A ball of paint in the team colour, radius ~0.03 frame widths, that wobbles. Its outline is a circle whose radius changes with the angle: r(θ) = r₀ · (1 + Σₖ aₖ sin(kθ + φₖ(t))), for k = 2…4 with small aₖ and phases that turn over time. A few droplets trail behind it, shrinking and falling with gravity. Drawn with `cv2.fillPoly` (anti-aliased) plus one highlight spot, so it looks wet.
- **The splash at the hit,** generated once per hit from a random seed, so it stays the same in every later frame:
  - a central blob with the same outline formula, more irregular;
  - 5–10 spikes along random rays, thinner towards the tip, mostly in the blob's direction of flight (the paint flies on);
  - small droplets further out, sizes from a power law (many small, few large);
  - 1–3 drips running down from the lower edge over ~1 s.
- **Where it sits: body coordinates.** At the hit, the impact point becomes (`u`, `v`) on the torso from the smoothed keypoints (`Game.torso()`): `u` from the left to the right shoulder, `v` from the shoulder line (0) to the hips (1). The splash is stored in these coordinates, and a bilinear map over the four corners (both shoulders, both hips) puts it back on the body every frame, so the paint moves, leans and scales with the player. `splat_band` becomes a band in torso coordinates (e.g. `u` 0.1–0.9, `v` 0.05–0.9), so a hit on an arm still puts paint on the shirt.
- **Covering the body.** A body mask from the keypoints: the torso quad plus a capsule (a thick line) along each upper and lower arm and leg, with widths in shoulder widths. The paint is clipped to the mask, so a splash that reaches past the shoulder runs onto the arm, not into the background. Every hit adds a splash, and splashes grow as health drops (as the splat size does now), so a player near 0 HP is mostly paint.
- **Optional:**
  - shade the paint with the camera image's brightness (multiply), so the folds of the clothes show through;
  - hide splashes on a side of the body turned away from the camera.
- **Cost.** The shapes are generated once per hit. Each frame only maps a small paint layer per player onto the body and blends it. Target: under ~2 ms per player, measured in the run log.
- **Config.** Blob size and wobble, spike and droplet counts, and drip speed in `render:`; the hit band in `game:`.

**Check.** A punch that hits at shoulder height leaves paint near the shoulder, and the paint stays there while the player moves their arms and leans; no paint outside the body mask; no two splashes look the same; the render cost per frame is logged.

### 7. Both players attacking at the same time: the stronger attack gets the shot

**What.** Make simultaneous attacks fair and visible, and link them to the consumables (feature 4): power items raise a player's attack level, and when both attack at once, the higher attack level gets the shot. Today (`Game.advance()` in `interaction/game.py`, as in `game.py`):

- bolts and beams fly through each other, and both hits land;
- when both hits would finish the round in the same frame, the attack fired first wins: `_apply_damage()` ignores everything once `winner_id` is set, so the order of the `projectiles` list decides.

**Owners.** Task 4: collisions, attack levels and the tie rule in `interaction/game.py`. Task 5: the clash effects in the renderer.

**Design.**

- **Attack level:** `GamePlayer.attack`, 1 without items, raised by power items (feature 4).
- **Blob meets blob:** two blobs (feature 6) of different players whose paths cross in a frame (x segments overlap, y within the blob's size):
  - the blob with the higher attack level flies on and hits; the other bursts at the meeting point, a splash in mid-air in its colour;
  - with equal levels, both burst, with a splash in both team colours.
- **Final smash** (feature 3): it falls from above, so blobs never meet it, and a blob fired at a player under the rain still hits. If both players fire one, both fall; each player's shield blocks the drops on them.
- **Final blows in the same frame:** apply all of a frame's damage first, then decide: if both are at 0, the higher attack level wins; with equal levels it is a draw ("DRAW", both colours). Otherwise the one with health left wins. Needs a `result` instead of only `winner_id` (also used by feature 1's result screen).

**Check.** Unit tests: two blobs with equal levels cancel in the middle; a blob with the higher level goes through and hits; two finishing hits in one frame give a draw with equal levels, and a win for the higher level otherwise. On the webcam: both punch at once, with and without a power item.

## Technical follow-ups from the game and renderer merges

| Item | Why | Where |
| --- | --- | --- |
| Balance on the webcam | Damage, speeds and the shield time are `game.py`'s values; check them over a few real rounds | `game:` config |
| `tools/eval_gestures.py` | Needs clips with real, labelled gestures; the test clips have none | `tools/` |
| Render cost at high resolution | The pixel look costs ~14 ms per frame on a 4K clip (~4.5 ms at 720p) | `render.pixel_size`, `pixelate()` |

## Technical follow-ups from the Task 3 merge

| Item | Why | Where |
| --- | --- | --- |
| Crossing, occlusion and re-entry clips | Identity was checked on edited copies of the session clip (blacked out; halves swapped), not on real crossings | `data/clips/`; `tools/eval_identity.py` with hand labels |
| `reid_threshold` per weights | With the Hugging Face weights, same-player similarities on the fitness test clip are often under 0.6 | `identity:` config |
| Webcam check of identity | Re-entry, the coat fix (`min_face_conf`) and a player turning away live | `--until identity` |
| `track_buffer` in frames | Ultralytics counts frames: the occluded time is 1 s at 30 FPS but 1.7 s at 17 FPS | `identity.botsort` |
| P1 / P2 order | The first player confirmed is P1 (as in `bodypose.py`); with feature 1, the calibration can assign P1 to the left player | `identity/tracker.py` |

## Technical follow-ups from the Task 1 merge

| Item | Why | Where |
| --- | --- | --- |
| Webcam check of the face stage | The crop shrinking, head pose and MediaPipe changes (merge log, Task 1 modifications) were tested on clips and photos; check chin up and down, turning, mouth and blink bars, and `hat_depth_mm` live | `face:` config |
| Jitter on a still clip | The test clips are all moving; shrinking crops and the head pose may add shake that only a still clip shows | `tools/eval_face.py` on the shared `normal` clip |
| MediaPipe GPU delegate | MediaPipe costs ~8 ms per frame on the CPU; the GPU delegate may cut that | `face/landmarker.py` |
| Remove `garb/` | Everything in it is merged or deliberately left out (merge log); delete it and its `.gitignore` entry once the owners have checked the merge | `.gitignore` |
| `core/types.py` heads-up | Two fields were added during the merge: `Face.alpha` and `Face.blendshapes`; mention them in the merge PR | merge PR description |

## Considered, not planned

Mentioned during the merge but not chosen. Kept here in case there is time left:

- Paint splats drawn on the face (in face coordinates) as damage, instead of on the body box.
- A "paint spit" attack: hold the mouth wide open to fire a paint blob from `anchors["mouth"]`.
- Real props as consumables: YOLO detection of an apple, banana, bottle or cup (COCO classes) instead of virtual items. Needs a second detection model per frame.
