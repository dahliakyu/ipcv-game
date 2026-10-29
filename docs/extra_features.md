Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. Features move to [merge_log.md](merge_log.md) once built: so far the game start (["Game start"](merge_log.md#game-start-instructions-and-pose-calibration)), the face effects (["Face effects"](merge_log.md#face-effects-team-colour-hit-reaction-status-above-the-head-cheek-bulge)), the consumables (["Consumables"](merge_log.md#consumables-drinks-caught-with-the-mouth)) and the paint blob with splashes on the body (["Normal attack"](merge_log.md#normal-attack-paint-blob-and-splashes-on-the-body)). Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

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
| Game phases: start places, calibration, countdown, result | `interaction/game.py`: `Game.phase`, `_next_phase()`, `place_taken()`, `_measure()` (neutral pose), `_check_line_up()`, `_tick()` (moves); `app/pipeline.py`: `_calibrate_identity()`; measurements in `pose/measure.py` | `game:` (`intro`, `round_players`, `ready_s`, `neutral_*`, …) | `tests/test_game.py` |
| Game rules: actions, hits, damage, round end | `interaction/game.py`: `Game.update()` (torso corners: `torso_corners()`) → `_act()` (events to actions, `facing()`) and `_drink_with_mouths()` (mouths from `_track_mouth()`, buffs by `_apply_drink()`) → `advance()` (movement, drinks falling in `_advance_drinks()`, hits, `_apply_damage()` with the buffs and a splash from `_splash()`, `_tick_buffs()`; `test_hit()` for key `b`); state in `GamePlayer` (also `attack`, `defense`), `Projectile`, `PaintBeam`, `Shield`, `PaintSplat`, `Drink`; this frame's `hits` and `drunk` | `game:` | `tests/test_game.py` (`frame_state()` builds fake frames) |
| How the game looks | `scene/renderer.py`: `Renderer.draw()` (order: camera, paint, `draw_game()`, face effects, `_draw_status()` (name tag, crown, dizzy stars), health bars, `_draw_hud()`, `_draw_game_over()`); sprites in `scene/fire_bolt.py`, `magic_beam.py`, `fire_ring.py` (frames loaded by `scene/animation.py`, `load_frames()`); paint in `scene/paint.py` (`BodyPaint`, `splash_shapes()`, `body_mask()`, the blob `draw_blob()` and its burst `draw_burst()`); helpers in `scene/draw.py` (`player_color()`, `blend_rgba()`, `shift_hue()`); intro screens in `_draw_start_place()`, `_draw_instructions()` (stick figures and move texts in `scene/instructions.py`); drinks in `_draw_drinks()` (sprites from `drink_sprites()`, `draw_badge()`), `_draw_popups()` (pick-up text), `_draw_buffs()` (icons above the health bars), colours in `DRINK_COLORS` | `render:` | `tests/test_renderer.py` |
| Face effects and face data | `face/effects.py`: `FaceEffects.update()` (game hits, players holding a drink) and `draw()` (war paint, cheek bulge, mustache, flash, team hat, flying hats), per-player `_Anim`, `bulge(pid, t)` on drinking; `face/tracker.py` (`Face`, anchors), `face/head_pose.py` (3D anchor points), `face/landmarker.py` (blendshapes) | `face:` | `tests/test_face.py`, `test_effects.py` |
| Players and identity | `identity/tracker.py`; `Player.status`, `reentered`, `pos` (`_position()`: x and depth in metres, Task 3's spatial output); P1 on the left at calibration (`order_left_to_right()`) | `identity:` | `tests/test_identity.py` |
| Keys, flags, pipeline order | `app/main.py` (`q`, `p`, `r` new round from the countdown, `R` full reset to waiting, `b` test blob at P1; `--no-intro`, `--solo`), `app/pipeline.py` | | `tests/test_pipeline.py` |

Each module reads only its own config section; the renderer reads game timings from `Game` (`beam_charge_s`, `shield_opacity()`, `torso()`, …), not from `game:`. Try a change with `python -m ipcvgame.app.main --source data/clips/session.mp4` (a full recorded round), `--until game` for the logic as plain shapes, and `--record-view <clip>` to save what the window shows. New CSV columns go in a module's `log_fields()`.

### What the face stage already gives you

`state.faces[pid]` (see [merge_log.md, Task 1](merge_log.md#task-1-face-tracking-and-ar)):

| Field | Use |
| --- | --- |
| `anchors["above_head"]`, `["hat"]`, `["forehead"]`, `["mouth"]`, `["left_cheek"]`, `["right_cheek"]` | Where to draw; they follow head tilt and turn. The cheeks are on the image left and right |
| `scale` | Face width in px: size effects with it, so near and far players look the same |
| `roll`, `yaw` | Rotate sprites with the head; squash them by cos(yaw) when the head turns |
| `alpha` | Fade every face-anchored effect with it, so effects disappear smoothly when a face is lost |
| `blendshapes` | MediaPipe scores 0–1: `jawOpen`, `eyeBlinkLeft`/`Right`, `mouthSmileLeft`/`Right`, `cheekPuff`, … Empty when not measured. On our mirrored frame, `…Left` is the player's own right side |

## Features

Renumbered five times: twice on 2026-10-28 (the second time when the face effects were built), and three times on 2026-10-29 (when the consumables were built; when the normal attack was built and the dodge dropped; when the aiming was dropped). Old numbers in commits and `merge_log.md` map as follows:

| Now | Was after the normal attack (2026-10-29) | Was after the consumables (2026-10-29) | Was after the face effects (2026-10-28) | Was after the first renumbering (2026-10-28) | Was before |
| --- | --- | --- | --- | --- | --- |
| Game start: built, recorded in `merge_log.md` ("Game start") | — | — | — | — | 12 |
| Face effects: built, recorded in `merge_log.md` ("Face effects") | — | — | — | 1 | 1 (team colour), 2 (hit reaction), 3 (status above the head), 7 (cheek bulge) |
| Consumables: built, recorded in `merge_log.md` ("Consumables") | — | — | 2 | 3 | 6 |
| Normal attack as splashed paint: built, recorded in `merge_log.md` ("Normal attack") | — | 3 | 4 | 5 | 13 (paint stains where the hit landed) |
| 1. Final smash | 1 | 1 | 1 | 2 | 4 (super charge, reset by blinking), 15 (orb in four parts), 16 (paint from above) |
| 2. Both players attacking at once | 3 | 4 | 5 | 6 | 14 |
| Aiming the normal attack: dropped (2026-10-29) | 2 | 2 | 3 | 4 | 11 |
| Dodge by stepping forward: dropped (2026-10-29) | — | 5 | 6 | 7 (added on 2026-10-28) | — |

Dropped (numbers from before the first renumbering): 5 (taunt), 8 (learned punch classifier: the gestures stay rule-based; its limits are in [limitations.md](limitations.md)), 9 (charge bar: the orb unlocks the final smash) and 10 (weapon pick-ups). On 2026-10-29 the Task 5 owner also dropped the dodge by stepping forward (a normal blob missing a player who steps ~0.5 m towards the camera, judged by depth). Then the Task 5 owner dropped the aiming of the normal attack (a pointing laser from the hand and blobs flying in 2D towards where the arm points), to focus on the final smash and both players attacking at once: blobs keep flying sideways, towards the nearest other player (`Game.facing()`). The designs of both dropped features are in the git history of this file. The line-up in the calibration was built for the dodge (`merge_log.md`, "Game start: Modifications") and stays for now; the depth is calibrated but unused, see [limitations.md](limitations.md), "Depth is calibrated, but no game rule uses it".

### 1. Final smash: a paint orb in four parts, raining until the caster blinks

**What.** The major attack becomes a final smash, like the Smash Ball in Smash Bros. Players collect a paint orb in four parts: one small orb at a time bounces around the frame, and a punch that touches it collects it. With all four, the player's body glows in their team colour, and hands up starts the attack: the orb flies up, bursts, and paint rains down on the opponent. **The rain lasts only while the caster does not blink:** the first blink ends it, and both players' parts are used up, so both have to collect four new ones. Attacking becomes a staring contest that the other player can try to break. The opponent has two defences: getting out from under the direct paint drop, which is possible but very hard, or the shield. It replaces the magic beam.

**Builds on** (state on 2026-10-29):

- **Attack now:** `hands_up` fires the magic beam: `Game.beam_attack()`, `PaintBeam`, the `beam_*` keys in `game:`, drawn by `scene/magic_beam.py` with the `beam_ball` / `beam_band` keys in `render:`. Texts that name it: `CONTROLS` in `scene/renderer.py` ("HANDS UP: MAGIC BEAM") and `MOVE_TEXT["hands_up"]` in `scene/instructions.py`.
- **Normal attack:** the paint blob (`Projectile`, drawn by `draw_blob()` in `scene/paint.py`); every damaging hit adds a splash on the body (`Game._splash()`, `PaintSplat`; [merge_log.md, "Normal attack"](merge_log.md#normal-attack-paint-blob-and-splashes-on-the-body)).
- **Rules a hands-up gesture already goes through:** one action at a time (`gestures.py`: back to a normal position between actions), and facing the camera, checked in `Game.update()` before the action (`Game.facing_camera()`, head and body; merge_log "Game start: Modifications").
- **Shield:** `activate_shield()`, `shield_s` (1 s) in `game:`, `Game.shielded(pid)`.
- **Face data:** `state.faces[pid]`: `blendshapes` (`eyeBlinkLeft`, `eyeBlinkRight`), `alpha`, `valid`, `anchors["above_head"]`; the face stage logs `face_p<pid>_blink` and `face_p<pid>_pitch_deg`. The head pitch is only inside the face tracker (`_FaceState.pitch`), not in `Face`. The face tracker is Task 1's code.
- **Punch:** the event carries no hand position; `punching_hand()` in `game.py` picks the wrist reaching furthest to the side, as the blob's start does.
- **Test on clips:** combat with `--no-intro`; `session.mp4` never reaches the countdown with the intro.

**Owners.**

- Task 4: orb parts, the attack's timing, area and hits in `interaction/game.py` (replaces `PaintBeam`); the `blink` event in `interaction/gestures.py`.
- Task 1: the blink signal (already there: `blendshapes`) and the eye glow in `face/effects.py`.
- Task 5: part sprites, the orb icon in the HUD, and the rise, burst, drops and splashes in the renderer.

**Design.**

- **Orb parts** (decided 2026-10-29: one part at a time, collected by a punch, no knock-out).
  - **A part** is a small version of the orb, not a quarter of it. Only one is on the field at a time.
  - **Spawning:** the next part comes ~5–8 s after the last one was collected (or after the round or the last final smash started). It enters from the top edge at a random x and then bounces around like a consumable: a constant speed, reflected at the edges of a band at punching height (about head to chest of players at ~2.3 m), so it drifts past both players. No part spawns while a final smash runs.
  - **Collecting:** a `punch` event whose punching hand (`punching_hand()`) is within ~0.5 shoulder widths of the part's edge collects it. The punch also fires its blob as usual.
  - **Holding:** each player's parts show as four small orb slots by their health bar, filled in their team colour. A player with four collects no more; the other player still can.
  - **No knock-out:** parts are lost only when a final smash starts.
- **Start.** With four parts, the player's body glows in their team colour with "RAISE YOUR ARMS" on the chest (decided 2026-10-29, instead of the full orb above the head), and `hands_up` starts the attack. Without four, `hands_up` does nothing (the slots flash), and the one-action-at-a-time reset still applies. At the start, both players' parts go back to 0, whether the rain lasts a second or much longer, and parts start bouncing in again after the smash.
- **One player** (`--solo`, `--no-intro` alone): parts spawn and are collected the same way. With no opponent in view, the rain falls on the far half of the frame (the half away from the caster), so the caster can test the start and their own blink ending it. Test keys: `n` fills P1's orb (skips collecting); `v` lets a test rain fall on P1 in P2's colour for a fixed time (no blink ends it), to see drops hitting, the paint and the shield blocking.
- **Duration: until the caster blinks.** Blinks count only during the attack. The rain stops at the first blink after the start, and also when the caster's face is not measured (empty `Face.blendshapes`, or `alpha` < 1), so turning away cannot hide a blink. A cap (~6 s, in config) keeps one attack from deciding the round alone. Damage per drop is fixed, so a longer stare means more damage.
- **Blink detection.** The raw score is not enough: on the test clips, `blink` = mean of `eyeBlinkLeft`/`Right` went over 0.5 in up to 26% of frames just because people looked down. Use:
  - a per-player baseline: the running median of the score over the last ~3 s, which follows lighting, glasses and eye shape;
  - a blink that starts at score > baseline + 0.35 (and > 0.5), and ends at score < baseline + 0.15 (hysteresis);
  - a duration of 60–500 ms (longer = eyes closed, not a blink; closed eyes also end the rain);
  - looking down (pitch < −20°, see `log_fields` `pitch_deg`) does not count as a blink. The game gets the pitch from a new field `Face.pitch` (rad, > 0 = chin up) in `core/types.py`, filled from the tracker's smoothed pitch (decided 2026-10-29; heads-up in the merge PR, with `Face.alpha` and `Face.blendshapes`).
- **Measurement rate.** MediaPipe measures each player at ~15 Hz (`face.landmarker_per_frame: 1`). Between two runs, `Face.blendshapes` repeats the last result for up to `face.landmarker_max_age_s` (0.25 s) and is `{}` after that; it has no flag for a fresh result, so time a blink with `Frame.t`. A 100 ms blink can fall between two measurements, and a missed blink makes the rain last longer. During an attack, measure the caster every frame (as `landmarker_per_frame: 2` does, +~8 ms); that needs a way for the game to ask the face stage for it (e.g. through the pipeline, like the game start's `_calibrate_identity()`).
- **The attack, in phases:**
  1. **Burst:** the orb flies up out of the top of the frame (~0.3 s). The frame dims for a moment with a flash in the caster's colour, as Smash Bros does for a final smash.
  2. **Warning:** a target mark on the floor under the opponent (~0.4 s).
  3. **Rain:** paint drops fall over the drop zone, densest in the middle, until the caster blinks.
  4. **Splashes:** splash marks fade on the floor.
- **Drop zone and dodging.** The zone is centred on the opponent's body box, ~2 box widths wide, and follows the opponent slowly: it moves towards their current position at a limited speed (~0.3 frame widths/s). A player standing still takes the whole attack. Getting out of the direct drop needs a quick, long step to the side, faster than the zone follows; the zone then catches up again, so dodging works only now and then. Zone width and follow speed go in config and are tuned on the webcam.
- **Hits.** Each drop tests its path since the last frame against the target's box, as the blob does. Every hit adds a splash on the body (`_splash()`, as every damaging hit does; at ~5 drops a second, use smaller splashes for drops or merge them, so the paint stays under ~2 ms per player) and a little damage (e.g. 1 per drop, ~5 per second of rain; the beam did 12 per attack). The shield blocks the drops while it is up. It lasts `shield_s` (1 s), so it covers part of a long rain, not all of it.
- **Look.**
  - **Eye glow:** while the rain lasts, the caster's eyes glow in their team colour, so both players can see what ends it. On the blink, the glow shatters into particles, with a short "blink!" text at `above_head`.
  - **Rain:** drops in the caster's team colour, falling with gravity, slightly random sizes; splash sprites where they hit the floor (the bottom of the body box, or the frame bottom) or the player.
- **Replaces the beam.** `hands_up` no longer fires the magic beam. The `beam_ball` / `beam_band` drawing goes; the magic ball frames can stay for the burst. The calibration panel (game start, `merge_log.md`) then shows hands up as the final smash, not "MAGIC BEAM".

**Check.**

- Orb: only one part on the field at a time; it bounces inside its band; a punch that touches it collects it, a punch that misses does not; a player cannot hold more than four; `hands_up` does nothing without four parts; a final smash resets both players to 0.
- Blink: on the webcam, a deliberate blink always ends the rain; looking down at the floor for 2 s does not; turning away ends it. Log blinks and the rain's length per attack, and count false blinks on the clips.
- Attack: the rain follows the opponent at the set speed; standing still takes every drop, a quick long step out avoids some; the rain never lasts past the cap; the shield blocks the drops while it is up.
- One player: every step has a unit test with one player and a check with key `n` / `v` on `shield.mp4`.

**Implementation steps.** One step at a time, each checked before the next.

1. **Orb parts (game logic, Task 4).** `OrbPart` in `interaction/game.py`: spawning after `orb_every_s`, entering from the top and bouncing in the band `orb_band_h`; a punch collects it (`_collect_orb()`, hand within `orb_reach_sw` of its edge); `GamePlayer.orb` (0–4). Key `n` (`Game.test_orb()`). `draw_debug()` shows the part, its band and the counts; CSV columns. Tests: spawning, bouncing, collecting with one player, a missing punch, a full orb collects no more. Check: `shield.mp4` with `--no-intro`.
   - *As built (2026-10-29):* as planned; the CSV columns are `game_p<pid>_orb` and `game_orb_x` / `game_orb_y` (the part's centre, empty without one). The part bounces in `orb_band_h` 0.25–0.5 of the frame height, from the punching hands on `session.mp4` (0.36–0.49, 5th–95th percentile of 16 punches). New config keys in `game:`: `orb_every_s`, `orb_parts`, `orb_r_h`, `orb_band_h`, `orb_margin_w`, `orb_speed_w`, `orb_angle_deg`, `orb_reach_sw`, `orb_seed`. Key `n` in `app/main.py` → `Pipeline.test_orb()`; README lists it. `hands_up` still fires the beam until step 4.
   - *Clips* (`--no-intro`, `orb_seed: 1`): the first part comes at ~5.5 s and stays in its band (y −25 to 360 px of 720, x 129–1151 px). On `session.mp4` a punch by the right-hand player collected one at 19.4 s, the hand ~60 px from the part's centre; on `shield.mp4` (no punches) the part only bounces. Game stage ~0.1 ms per frame.
2. **Orb look (renderer, Task 5).** The bouncing part (a small glowing paint orb, team-neutral), a pick-up burst and text, the four slots by each health bar, and a player with a full orb glowing in their team colour, with "RAISE YOUR ARMS". Check rendered frames and `ms_render` on `shield.mp4` with key `n`.
   - *As built (2026-10-29):* drawing in a new `scene/orb.py`. `draw_part()`: a ball split into red and blue by an S-curve (yin-yang), turning at `orb_spin_hz`, with a pulsing pale halo, a dark rim and a shine. `draw_pop()`: a ring and drops in the collector's colour where the part was (`Game.orbs_taken`, this frame's (pid, x, y)), with "ORB n/4" / "FULL ORB!" as a pick-up text (the renderer's pop-ups now keep a colour, not a drink kind). `draw_slots()`: four small balls on the health bar's label row, at its inner end, pulsing once full. `draw_body_glow()`: the paint's `body_mask()` plus a head disc (`glow_head_fw` face widths) at a quarter of the resolution, blurred over `glow_reach_sw` shoulder widths, the blur doubled and capped so the halo is strongest at the body's edge, the team colour moved `glow_white` towards white, pulsing between `glow_alpha` at `glow_hz`, `glow_inside` of it on the body. "RAISE YOUR ARMS" on the shoulder line: beside the head it ran into the name tag, hat and crown. New `render:` keys `orb_spin_hz`, `orb_halo`, `orb_pulse_hz`, `orb_pop_s`, `orb_pop_ring`, `orb_pop_drops`, `glow_white`, `glow_head_fw`, `glow_reach_sw`, `glow_alpha`, `glow_hz`, `glow_inside`, `glow_scale`, `raise_text_scale`; CSV `render_p<pid>_glow_ms` while it glows.
   - *Clips* (key `n` in a script): the first glow (pure team colour, reach 0.3 sw) hardly showed on the red shirt at ~2.3 m on `session.mp4`; lighter colour and reach 0.5 fixed it. Glow cost: 1.8 ms median per player at ~2.3 m (`session.mp4`), 4.3 ms median, 9 ms 95th percentile close up (`shield.mp4`, the body box is most of the frame: the blend 1.9 ms, the blur 0.8 ms, the resize 0.5 ms per call). Filling the colour layer by a NumPy broadcast took another 3.4 ms; `cv2.rectangle` fills it. Only drawn while a full orb waits for the raised arms.
3. **Head pitch and blinks (Tasks 1 and 4).** `Face.pitch` (`core/types.py`, filled in `face/tracker.py`); `BlinkDetector` in `interaction/gestures.py` (baseline, hysteresis, 60–500 ms, pitch filter; `gestures: blink_*`), a `blink` event and CSV `gest_p<pid>_blink`. Count false blinks on `session.mp4`, `shield.mp4` and the test clips.
4. **The final smash (game logic, Task 4).** `FinalSmash` replaces `PaintBeam` and `beam_attack()`: hands up with four parts starts it and resets both orbs; phases burst, warning, rain; the drop zone follows the opponent at `smash_follow_w` (no opponent: the far half); drops fall with gravity and hit the target's box (path since the last frame); the shield blocks them; damage and a small splash per hit; the rain ends at the caster's first blink, when the caster's face is not measured, or at the cap. The pipeline asks the face stage to measure the caster every frame during the rain. Key `v` (`Game.test_smash()`). CSV: smash phase, its length and why it ended. The `beam_*` keys go.
5. **The final smash look (Tasks 5 and 1).** The burst (orb flying up, the frame dimmed with a flash in the caster's colour), the target mark, the drops and floor splashes, small body splashes within the paint budget (`render_p<pid>_paint_ms`), the eye glow and its shatter with "BLINK!" (`face/effects.py`). `scene/magic_beam.py`, its tests and `beam_*` render keys go; `CONTROLS` and `MOVE_TEXT["hands_up"]` name the final smash.
6. **Wrap-up.** `merge_log.md` section, this section removed and the backlog renumbered, `status.md`, and the limits found (e.g. blinks missed between measurements) in `limitations.md`.

### 2. Both players attacking at the same time: the stronger attack gets the shot

**What.** Make simultaneous attacks fair and visible, and link them to the consumables (built: [merge_log.md, "Consumables"](merge_log.md#consumables-drinks-caught-with-the-mouth)): an attack drink raises a player's attack multiplier, and when both attack at once, the higher attack multiplier gets the shot. Today (`Game.advance()` in `interaction/game.py`, as in `game.py`):

- bolts and beams fly through each other, and both hits land;
- when both hits would finish the round in the same frame, the attack fired first wins: `_apply_damage()` ignores everything once `winner_id` is set, so the order of the `projectiles` list decides.

**Owners.** Task 4: collisions, attack levels and the tie rule in `interaction/game.py`. Task 5: the clash effects in the renderer.

**Design.**

- **Attack level:** the attack multiplier from the consumables (`GamePlayer.attack`), 1 without an attack drink.
- **Blob meets blob:** two blobs of different players whose paths cross in a frame (x segments overlap, y within the blob's size):
  - the blob with the higher attack level flies on and hits; the other bursts at the meeting point, a splash in mid-air in its colour;
  - with equal levels, both burst, with a splash in both team colours.
- **Final smash** (feature 1): it falls from above, so blobs never meet it, and a blob fired at a player under the rain still hits. If both players fire one, both fall; each player's shield blocks the drops on them.
- **Final blows in the same frame:** apply all of a frame's damage first, then decide: if both are at 0, the higher attack level wins; with equal levels it is a draw ("DRAW", both colours). Otherwise the one with health left wins. Needs a `result` instead of only `winner_id` (also used by the game start's result screen, `merge_log.md`).

**Check.** Unit tests: two blobs with equal levels cancel in the middle; a blob with the higher level goes through and hits; two finishing hits in one frame give a draw with equal levels, and a win for the higher level otherwise. On the webcam: both punch at once, with and without a power item.

## Technical follow-ups from the game and renderer merges

| Item | Why | Where |
| --- | --- | --- |
| Balance on the webcam | Damage, speeds and the shield time are `game.py`'s values; check them over a few real rounds | `game:` config |
| `tools/eval_gestures.py` | Needs clips with real, labelled gestures; the test clips have none | `tools/` |
| Webcam check of the game start | The intro was checked on clips only: two players through the start places, calibration, countdown, result and replay; the P1-left swap with players who walked in the other way round; the line-up (whether 0.4 m by depth is too strict or too loose for two real players); the facing rule (whether 30° for head and body blocks players who glance at the opponent) | `--until game` |
| Depth on the webcam | `Player.pos` uses an assumed field of view and shoulder width; stand at a measured distance (e.g. 2 m), compare `p<pid>_depth_m` in the log, and set `camera_hfov_deg` from the camera's spec and `shoulder_m` to match | `identity:` config |
| Render cost at high resolution | The pixel look costs ~14 ms per frame on a 4K clip (~4.5 ms at 720p); the face effects and status add ~0.6 ms at 720p (render 4.3 → 5.0 ms on `session.mp4`) | `render.pixel_size`, `pixelate()` |
| Webcam check of the drinks | Checked on clips (no drinking) and in tests; one webcam run so far (`logs/run-20261028-231534.csv`, which set `drink_hold_s` to 2.5 s). Check alone (`--solo`) and with two: catching and drinking, whether `jawOpen` 0.5 / 0.25 and the 2.5 s hold fit, whether players know to close the mouth (a "CLOSE YOUR MOUTH" text while holding is an option), the buff values and the spawn rate | `game:` `drink_*` keys |
| Shield render time | On `shield.mp4` (2026-10-29, TeamViewer running), the render took ~51 ms on frames with the shield up and ~3 ms without; the same with drinks turned off. Not checked why yet | `scene/fire_ring.py`, `FireRing` |
| Asset packs in the repo | All of `assets/Fancy Drinks/` (paid; reselling or redistributing the files as assets is not allowed), including `Misc/`, and `assets/Jester/` are committed. If the repo or the final ZIP is shared, keep only the nine sprites in use, or keep the packs out of git with a download note in the README. `assets/beam/` (the fire blast, purchased) is unused since the paint blob (merge_log.md, "Normal attack", step 4): delete it and its README row | `assets/`, `.gitignore`, `README.md` |
| Webcam check of the face effects | Checked on clips only: the war paint and the hat band with real head turns; the name tag steady when standing still and keeping up when walking (`tag_filter_*`); the hat knock-off and flash in a real round; the dizzy stars drawn over the hat even when "behind" the head | `face:` and `render:` config |

## Technical follow-ups from the Task 3 merge

| Item | Why | Where |
| --- | --- | --- |
| Crossing, occlusion and re-entry clips | Identity was checked on edited copies of the session clip (blacked out; halves swapped), not on real crossings | `data/clips/`; `tools/eval_identity.py` with hand labels |
| `reid_threshold` per weights | With the Hugging Face weights, same-player similarities on the fitness test clip are often under 0.6 | `identity:` config |
| Webcam check of identity | Re-entry, the coat fix (`min_face_conf`) and a player turning away live | `--until identity` |
| `track_buffer` in frames | Ultralytics counts frames: the occluded time is 1 s at 30 FPS but 1.7 s at 17 FPS | `identity.botsort` |
| P1 / P2 order | Done in the game start (`merge_log.md`, step 5): when calibration starts, P1 becomes the player on the left. Without the intro (`--no-intro`), the first player confirmed is still P1, as in `bodypose.py` | `identity/tracker.py` |

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
