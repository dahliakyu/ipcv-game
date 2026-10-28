Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. None of them is built yet; the first one built, the game start, is recorded in [merge_log.md](merge_log.md#game-start-instructions-and-pose-calibration). Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

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
| Game rules: actions, hits, damage, round end | `interaction/game.py`: `Game.update()` → `_act()` (events to actions, `facing()`) → `advance()` (movement, hits, `_apply_damage()`); state in `GamePlayer`, `Projectile`, `PaintBeam`, `Shield`, `PaintSplat` | `game:` | `tests/test_game.py` (`frame_state()` builds fake frames) |
| How the game looks | `scene/renderer.py`: `Renderer.draw()` (order: camera, paint, `draw_game()`, face effects, health bars, `_draw_hud()`, `_draw_game_over()`); sprites in `scene/fire_bolt.py`, `magic_beam.py`, `fire_ring.py` (frames loaded by `scene/animation.py`, `load_frames()`); helpers in `scene/draw.py` (`player_color()`, `blend_rgba()`, `shift_hue()`); intro screens in `_draw_start_place()`, `_draw_instructions()` (stick figures and move texts in `scene/instructions.py`) | `render:` | `tests/test_renderer.py` |
| Face effects and face data | `face/effects.py` (`FaceEffects.draw()`, stateless), `face/tracker.py` (`Face`, anchors), `face/landmarker.py` (blendshapes) | `face:` | `tests/test_face.py`, `test_effects.py` |
| Players and identity | `identity/tracker.py`; `Player.status`, `reentered`, `pos` (`_position()`); P1 on the left at calibration (`order_left_to_right()`) | `identity:` | `tests/test_identity.py` |
| Keys, flags, pipeline order | `app/main.py` (`q`, `p`, `r` new round from the countdown, `R` full reset to waiting; `--no-intro`, `--solo`), `app/pipeline.py` | | `tests/test_pipeline.py` |

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

Renumbered on 2026-10-28. Old numbers in commits and `merge_log.md` map as follows:

| Now | Was |
| --- | --- |
| Game start: built, recorded in `merge_log.md` ("Game start") | 12 |
| 1. Face effects | 1 (team colour), 2 (hit reaction), 3 (status above the head), 7 (cheek bulge) |
| 2. Final smash | 4 (super charge, reset by blinking), 15 (orb in four parts), 16 (paint from above) |
| 3. Consumables | 6 |
| 4. Aiming the normal attack | 11 |
| 5. Normal attack as splashed paint | 13 (paint stains where the hit landed) |
| 6. Both players attacking at once | 14 |

Added on 2026-10-28: 7 (dodge by stepping forward).

Dropped: 5 (taunt), 8 (learned punch classifier: the gestures stay rule-based; its limits are in [limitations.md](limitations.md)), 9 (charge bar: the orb unlocks the final smash) and 10 (weapon pick-ups).

### 1. Face effects: team colour, hit reaction, status above the head, cheek bulge

**What.** Effects on and above each player's face that show who is who and what happens to them: the team colour on the face, a reaction to every hit, a name tag and status icons above the head, and puffed cheeks after eating. They share the face anchors and one per-player animation state.

**Owners.**

- Task 1: cheek anchors, the hat band, the hit animation and the cheek warp in `face/effects.py`.
- Task 4: the `hit` event in `interaction/game.py`; the `bite` event (feature 3).
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
  - **Final smash hits** (feature 2) can also add a short shake of the hat anchor.
- **Status above the head:** a name tag and icons, so the state is visible on the player. The health bars stay pinned to the bottom corners (Task 5 modifications: a bar that moved with the player was too busy).
  - **Name tag.** "P1" at `above_head`, small, in the team colour; size from `scale` with a minimum so far players stay readable.
  - **Dizzy stars.** Below 25% HP: 3 stars orbit `above_head` on an ellipse with a fixed height-to-width ratio from the config, turned by `roll` (`Face` has no pitch; decided 2026-10-28, so `core/types.py` stays as it is).
  - **Crown.** The player with more HP gets a small crown above the tag.
- **Cheek bulge:** after eating (the `bite` event, feature 3), the cheeks puff up for a second, chipmunk style.
  - **Warp.** A real image warp, not a sticker: a radial "bulge" (magnifying) warp around each cheek anchor with `cv2.remap`. Inside radius r (~0.3 face widths), the source radius is ρ' = ρ · (ρ / r)^k with k rising from 0 to ~0.5 and back over 1 s. A good image-processing technique to explain in the report.
  - **Cost.** Warp only the two small cheek boxes, not the whole frame.
  - **Trigger now.** Built before feature 3 (decided 2026-10-28): the player puffing their own cheeks (MediaPipe's `cheekPuff` > 0.5) starts the bulge. A `FaceEffects.bulge(pid, t)` call starts it too, for the `bite` event once feature 3 exists.

**Implementation steps.** One step at a time, each checked by the Task 5 owner before the next. Every new number goes in the module's config section with a reason.

1. **Team colour** (Task 1, `face/`). *Done.*
   - Cheek anchors `left_cheek` and `right_cheek`: 3D points on the head model (`anchor_points_mm()`), projected like the hat; 2D offsets in `geometry()` when there is no head pose; smoothed with the other anchors.
   - Hat band: the red band of `hat.png` selected by hue and turned to each team colour once at load (`shift_hue()` on the band pixels only), one hat per team.
   - War paint: two short stripes in the team colour at each cheek anchor, along the eye line (`roll`), shortened by cos(`yaw`) on the cheek turned away, faded by `alpha`.
   - As built: `left_cheek` / `right_cheek` are on the image left and right, like `head_model_mm`'s eyes. They sit `cheek_out` 0.33 face widths to each side and `cheek_drop` 0.22 below the eyes (the design's 45 and 30 mm), `cheek_depth_mm` 10 behind the eyes. `geometry()` now takes the face config; `_smoothed_geometry()` smooths every anchor it is given, so new anchors need no change there. In `face/effects.py`: `band_mask()` (hue around the circle, so 178 counts as red), `recolor()`, `FaceEffects.team_hat(pid)` made on first use, `war_paint()` and `fill_polys()` (anti-aliased, blended in the stripes' box only). New `face:` keys: `cheek_out`, `cheek_drop`, `cheek_depth_mm`, `hat_band_hue`, `hat_band_hue_tol`, `hat_band_min_sat`, `hat_band_min_val`, `warpaint_len`, `warpaint_thick`, `warpaint_gap`, `warpaint_opacity`. The band test on `hat.png`: hue less than 8 from red, saturation ≥ 80 and brightness ≥ 100; without the brightness limit, dark reddish pixels along the black outline were taken too.
   - On `session.mp4`: P1's band and stripes red, P2's blue; the stripes sit on the cheekbones, and at a turn of −31° the turned-away (image-left) stripes are shorter. The war paint adds ~0.16 ms per face (100 px face, 720p).
2. **Hit event** (Task 4, `interaction/game.py`). *Done.*
   - `_apply_damage()` records a hit (target, attacker, direction away from the attacker, damage) in `Game.hits`, cleared each frame like `Game.actions`; logged as the action `hit`.
   - As built: a `Hit` dataclass (`target_id`, `owner_id`, `direction`, `damage`). The direction is the way the bolt or beam travelled (+1 to the right, −1 to the left), passed into `_apply_damage()` by both callers, so the hat later flies the way the attack went. Only hits that do damage count: a shield block, a hit on a player already at 0 or after the win adds nothing. `Game.hits` is cleared at the start of `update()` and by `_clear_round()`; the target's action `hit` shows in `game_p<pid>_actions` and in `--until game`.
   - On `session.mp4` (`--no-intro`): 7 health drops for P1 and 10 for P2, each with exactly one `hit`.
3. **Hit reaction** (Task 1, `face/effects.py`; wired in `scene/renderer.py` and `app/pipeline.py`). *Done.*
   - `FaceEffects` keeps one animation record per player, fed with `Game.hits` and `Frame.t`; `reset(pid)` on `Player.reentered`.
   - Hat knocked off: it leaves the anchor with a velocity away from the attacker and upwards, spin and gravity, fades out, then pops back with a short scale-up. A hit while the hat is off does not throw a second hat.
   - Flash: white, then the attacker's colour, over an ellipse at the face centre, sized by `scale`, turned by `roll`.
   - As built: `FaceEffects.update(hits, faces, t)` starts the reaction; `draw(canvas, faces, t)` plays it. Without `t` (the `--until` debug views, which have no game) it draws the faces as before. The state is one `_Anim` record per player: when the hat was knocked off, from where, at what face width, roll and direction, and when the flash started and who hit. A player without a visible face gets no reaction. Every hit restarts the flash; the hat is knocked off again only once it is back on the head (`hat_fly_s` + `hat_pop_s`). The flying hat stays where it was thrown in the frame, not on the moving head, and is drawn even if the face is lost meanwhile. Its path in face widths: x = direction × vx × t, y = −vy × t + g t² / 2, angle = roll + direction × spin × t; fully visible for the first half of the flight, then fading. Back on the head it grows with an ease-out over `hat_pop_s`. The flash is an ellipse polygon (`ellipse()`) filled by `fill_polys()`, drawn over the war paint and mustache and under the hat. The renderer calls `update()` with `game.hits` before drawing; `Pipeline` resets the effects on `reentered` and with key `R`. New `face:` keys: `hat_fly_s`, `hat_pop_s`, `hat_fly_v`, `hat_gravity`, `hat_spin_deg`, `flash_s`, `flash_axes`, `flash_opacity`.
   - On `session.mp4` (P1's beam hits P2 at frame 180; the clip runs at ~17 FPS, so ~4 frames per 0.25 s): white flash, then red; the hat arcs to the left, the way the beam went, spinning, fades out by 0.6 s and is back on the head by 0.78 s. With the design's 1.5 face widths/s, the hat moved only ~0.3 face widths before it faded and looked like a wobble, so the throw is 3 face widths/s sideways and up, with gravity 14 face widths/s².
4. **Status above the head** (Task 5, `scene/renderer.py`). *Done.*
   - Name tag "P1"/"P2" at `above_head` in the team colour, with a minimum size; "FACE THE CAMERA" takes its place while shown.
   - Dizzy stars below 25% HP; a crown above the tag for the player with more HP (none at a draw). Both fade with `alpha`.
   - As built: `Renderer._draw_status()` runs right after the face effects. The name tag is shown in every phase. The crown and stars show from the countdown on, when health counts.
     - `crowned()` gives the round player with strictly the most HP, or None at a draw or with one player.
     - `_draw_tag()` draws the tag upright, so it stays readable when the head tilts, and the crown above it.
     - A head near the top of the frame has no room above it under the top bar. On `session.mp4` the tags and crown were hidden under the bar. In that case the tag goes beside the hat, `tag_side_fw` face widths away, on the side towards the middle of the frame.
     - The tag is kept steady per player (`_Tag`; `Renderer.reset(pid)` on `reentered` and key `R`). Without this it jittered: on `session.mp4` it jumped between above the head and beside the hat 26–45 times, and changed size on 73–94% of frames.
       - It goes back above the head only with `tag_side_hyst` tag heights of room to spare (hysteresis).
       - Beside the hat, it changes sides only once the player is `tag_side_flip` of the frame width past the middle.
       - The text size changes only when the face width has changed by more than `tag_size_tol`.
       - Its position has its own One Euro filter (`tag_filter_min_cutoff` 0.5 Hz, smoother than the face's 1.5 Hz), because `above_head` sits far above the eyes and shakes most.
     - "FACE THE CAMERA" replaces the tag and crown while it is shown (`_hint_shown()`).
     - The stars ring the hat, `stars_rise` above the brim, instead of `above_head`, where the tag is. Stars behind the head are drawn first and smaller (`star()` gives the five-pointed polygon).
     - While a face fades out, its status is drawn on a copy of the frame and blended in with `alpha`.
   - New `render:` keys: `tag_face_h`, `tag_min_h`, `tag_side_fw`, `tag_side_hyst`, `tag_side_flip`, `tag_size_tol`, `tag_filter_min_cutoff`, `tag_filter_beta`, `crown_per_tag`, `dizzy_hp`, `stars_n`, `stars_hz`, `stars_rise`, `stars_orbit`, `stars_flat`, `star_size`. The stars are 0.2 face widths across; 0.12 was hard to see at ~2.3 m.
   - On `session.mp4` (frame 1400; P1 has 38 HP, P2 has 6): P1's tag and crown sit beside the hat, P2 has stars ringing the hat, and both tags are readable. Frame 400: tags and crown above the heads. Cost: 0.26 ms for two players, 1.4 ms while a face fades. Steadiness over the whole clip, before → after:
     - jumps between above and beside: 26 → 5 (P1) and 45 → 3 (P2);
     - size changes: 1789 → 36 and 1350 → 1;
     - while the head is nearly still, the tag moves 2.0 → 0.6 px per frame on average for P1 (1.3 → 0.6 for P2; 90th percentile 3.2 → 1.4 px). The jumps left over come with the hat itself jumping (~29 px).
5. **Cheek bulge** (Task 1, `face/effects.py`).
   - Radial bulge warp with `cv2.remap` around each cheek anchor, only in two small boxes; strength rises and falls over 1 s.
   - Started by `cheekPuff` above the threshold (once per puff) or by `FaceEffects.bulge(pid, t)`.
6. **Check and docs.** Rendered frames from the clips for each effect, timing per face, then the feature moves to `merge_log.md`.

**Check.**

- Both players on the crossing clip keep their own colour; the stripes stay on the cheeks when the head turns.
- One knock-off per hit, and the hat always comes back; two hits in a row don't stack two hats.
- The icons stay above the hat (also with the chin up) and fade out with the face.
- The bulge stays on the cheeks while the head moves and turns, and ends cleanly; ~1 ms extra per face.

### 2. Final smash: a paint orb in four parts, raining until the caster blinks

**What.** The major attack becomes a final smash, like the Smash Ball in Smash Bros. Players collect a paint orb in four parts. With all four, hands up starts the attack: the orb flies up, bursts, and paint rains down on the opponent. **The rain lasts only while the caster does not blink:** the first blink ends it, and the orb is used up, so the player has to collect four new parts. Attacking becomes a staring contest that the other player can try to break. The opponent has two defences: getting out from under the direct paint drop, which is possible but very hard, or the shield. It replaces the magic beam.

**Builds on** (state on 2026-10-28):

- **Attack now:** `hands_up` fires the magic beam: `Game.beam_attack()`, `PaintBeam`, the `beam_*` keys in `game:`, drawn by `scene/magic_beam.py` with the `beam_ball` / `beam_band` keys in `render:`. Texts that name it: `CONTROLS` in `scene/renderer.py` ("HANDS UP: MAGIC BEAM") and `MOVE_TEXT["hands_up"]` in `scene/instructions.py`.
- **Normal attack until feature 5 is built:** the fire bolt (`Projectile`). "Blob hits" below means bolt hits until then.
- **Rules a hands-up gesture already goes through:** one action at a time (`gestures.py`: back to a normal position between actions), and facing the camera, checked in `Game.update()` before the action (`Game.facing_camera()`, head and body; merge_log "Game start: Modifications").
- **Shield:** `activate_shield()`, `shield_s` (1 s) in `game:`, `Game.shielded(pid)`.
- **Face data:** `state.faces[pid]`: `blendshapes` (`eyeBlinkLeft`, `eyeBlinkRight`), `alpha`, `valid`, `anchors["above_head"]`; the face stage logs `face_p<pid>_blink` and `face_p<pid>_pitch_deg`. The face tracker is Task 1's code.
- **Test on clips:** combat with `--no-intro`; `session.mp4` never reaches the countdown with the intro.

**Owners.**

- Task 4: orb parts, the attack's timing, area and hits in `interaction/game.py` (replaces `PaintBeam`); the `blink` event in `interaction/gestures.py`.
- Task 1: the blink signal (already there: `blendshapes`) and the eye glow in `face/effects.py`.
- Task 5: part sprites, the orb icon in the HUD, and the rise, burst, drops and splashes in the renderer.

**Design.**

- **Orb parts.**
  - **Spawning:** one part at a time at a random free spot in the upper half of the frame (reachable with a raised hand), every ~5–8 s while fewer than four parts are in play; it slowly bobs.
  - **Collecting:** a wrist (smoothed keypoint, `state.motion`) within ~0.5 shoulder widths of the part, held for ~0.2 s, picks it up, as for the items in feature 3.
  - **Holding:** each player's parts show as a 4-slot orb icon by their health bar, in their team colour; the quarters fill in as they are collected. A player cannot hold more than four.
  - **Dropping:** 3 blob hits (feature 5) in a row knock one part out of the player; it flies off and lands as a part to collect again, so the other player can steal it.
- **Start.** Without all four parts, `hands_up` does nothing: the orb icon flashes "not ready", and the one-action-at-a-time reset still applies. With all four, the complete orb glows above the player's head, and `hands_up` starts the attack. The slots empty at the start: the orb is used up whether the rain lasts a second or much longer.
- **Duration: until the caster blinks.** Blinks count only during the attack. The rain stops at the first blink after the start, and also when the caster's face is not measured (empty `Face.blendshapes`, or `alpha` < 1), so turning away cannot hide a blink. A cap (~6 s, in config) keeps one attack from deciding the round alone. Damage per drop is fixed, so a longer stare means more damage.
- **Blink detection.** The raw score is not enough: on the test clips, `blink` = mean of `eyeBlinkLeft`/`Right` went over 0.5 in up to 26% of frames just because people looked down. Use:
  - a per-player baseline: the running median of the score over the last ~3 s, which follows lighting, glasses and eye shape;
  - a blink that starts at score > baseline + 0.35 (and > 0.5), and ends at score < baseline + 0.15 (hysteresis);
  - a duration of 60–500 ms (longer = eyes closed, not a blink; closed eyes also end the rain);
  - looking down (pitch < −20°, see `log_fields` `pitch_deg`) does not count as a blink.
- **Measurement rate.** MediaPipe measures each player at ~15 Hz (`face.landmarker_per_frame: 1`). Between two runs, `Face.blendshapes` repeats the last result for up to `face.landmarker_max_age_s` (0.25 s) and is `{}` after that; it has no flag for a fresh result, so time a blink with `Frame.t`. A 100 ms blink can fall between two measurements, and a missed blink makes the rain last longer. During an attack, measure the caster every frame (as `landmarker_per_frame: 2` does, +~8 ms); that needs a way for the game to ask the face stage for it (e.g. through the pipeline, like the game start's `_calibrate_identity()`).
- **The attack, in phases:**
  1. **Burst:** the orb flies up out of the top of the frame (~0.3 s). The frame dims for a moment with a flash in the caster's colour, as Smash Bros does for a final smash.
  2. **Warning:** a target mark on the floor under the opponent (~0.4 s).
  3. **Rain:** paint drops fall over the drop zone, densest in the middle, until the caster blinks.
  4. **Splashes:** splash marks fade on the floor.
- **Drop zone and dodging.** The zone is centred on the opponent's body box, ~2 box widths wide, and follows the opponent slowly: it moves towards their current position at a limited speed (~0.3 frame widths/s). A player standing still takes the whole attack. Getting out of the direct drop needs a quick, long step to the side, faster than the zone follows; the zone then catches up again, so dodging works only now and then. Zone width and follow speed go in config and are tuned on the webcam.
- **Hits.** Each drop tests its path since the last frame against the target's box, as the blob does. Every hit adds a splash on the body (feature 5) and a little damage (e.g. 1 per drop, ~5 per second of rain; the beam did 12 per attack). The shield blocks the drops while it is up. It lasts `shield_s` (1 s), so it covers part of a long rain, not all of it.
- **Look.**
  - **Eye glow:** while the rain lasts, the caster's eyes glow in their team colour, so both players can see what ends it. On the blink, the glow shatters into particles, with a short "blink!" text at `above_head`.
  - **Rain:** drops in the caster's team colour, falling with gravity, slightly random sizes; splash sprites where they hit the floor (the bottom of the body box, or the frame bottom) or the player.
- **Replaces the beam.** `hands_up` no longer fires the magic beam. The `beam_ball` / `beam_band` drawing goes; the magic ball frames can stay for the burst. The calibration panel (game start, `merge_log.md`) then shows hands up as the final smash, not "MAGIC BEAM".

**Check.**

- Orb: parts only disappear when collected or used; a player cannot hold more than four; a dropped part can be picked up by either player; `hands_up` does nothing without four parts.
- Blink: on the webcam, a deliberate blink always ends the rain; looking down at the floor for 2 s does not; turning away ends it. Log blinks and the rain's length per attack, and count false blinks on the clips.
- Attack: the rain follows the opponent at the set speed; standing still takes every drop, a quick long step out avoids some; the rain never lasts past the cap; the shield blocks the drops while it is up.

### 3. Consumables, Smash Bros style

**What.** Items drop from the sky during the round. A player can **eat** a falling item by catching it with their mouth (chomp), or, once it has landed, **grab** it from the floor with a hand. Each item has an effect tied to the existing mechanics. Power items raise the attack level, which decides who gets the shot when both players attack at once (feature 6).

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
  | Chili (power item) | +1 attack level for ~15 s, up to 3: more damage per blob, and the blob wins a clash (feature 6). Another chili adds a level and restarts the time |
  | Rotten item | splashes the eater with paint (−10 HP): a risk when grabbing blindly |

- **Attack level** (`GamePlayer.attack`, 1 without items): shown as small flames by the health bar, one per level, fading in the last 2 s.
- **Feedback.** Eating triggers the cheek bulge (feature 1); grabbing a short sparkle at the wrist.
- **Sprites (candidate):** `assets/Fancy Drinks/` (64 × 64 and 128 × 128 drinks and glasses, by Savannah; its Read Me states no licence). Not in git yet; add it with a README credit before using it.

**Check.** Items can be eaten and grabbed on the webcam; an item cannot be taken twice; mouth movements while talking do not eat items that are not near the mouth.

### 4. Aiming the normal attack: a pointing laser and paint blobs towards the opponent

**What.** The normal attack (a punch fires a paint blob, feature 5) no longer flies only sideways: it goes where the player aims, towards the opponent. A thin pointing laser from the hand shows the aim before firing. Only the normal attack is aimed; the final smash (feature 2) falls on the opponent and needs no aim.

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
  - The splash (feature 5) uses the 2D impact point.
- **Pointing laser.** While an arm is extended (straight but not yet a punch), a thin dashed line in the team colour runs from the wrist along the aim direction until it meets a body box or the frame edge. It fades with the gesture's state, and it is drawn only for the aiming player, to keep the frame readable.
- **Drawing.** The blob's trail follows the velocity vector, so the drawn blob needs no rotated sprite.

**Check.** Blobs hit an opponent who stands higher or lower (e.g. crouching) or in the same half of the frame; the pointing laser matches where the blob actually flies; aim assist does not snap to a player behind the shooter.

### 5. Normal attack as splashed paint that covers the body

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

### 6. Both players attacking at the same time: the stronger attack gets the shot

**What.** Make simultaneous attacks fair and visible, and link them to the consumables (feature 3): power items raise a player's attack level, and when both attack at once, the higher attack level gets the shot. Today (`Game.advance()` in `interaction/game.py`, as in `game.py`):

- bolts and beams fly through each other, and both hits land;
- when both hits would finish the round in the same frame, the attack fired first wins: `_apply_damage()` ignores everything once `winner_id` is set, so the order of the `projectiles` list decides.

**Owners.** Task 4: collisions, attack levels and the tie rule in `interaction/game.py`. Task 5: the clash effects in the renderer.

**Design.**

- **Attack level:** `GamePlayer.attack`, 1 without items, raised by power items (feature 3).
- **Blob meets blob:** two blobs (feature 5) of different players whose paths cross in a frame (x segments overlap, y within the blob's size):
  - the blob with the higher attack level flies on and hits; the other bursts at the meeting point, a splash in mid-air in its colour;
  - with equal levels, both burst, with a splash in both team colours.
- **Final smash** (feature 2): it falls from above, so blobs never meet it, and a blob fired at a player under the rain still hits. If both players fire one, both fall; each player's shield blocks the drops on them.
- **Final blows in the same frame:** apply all of a frame's damage first, then decide: if both are at 0, the higher attack level wins; with equal levels it is a draw ("DRAW", both colours). Otherwise the one with health left wins. Needs a `result` instead of only `winner_id` (also used by the game start's result screen, `merge_log.md`).

**Check.** Unit tests: two blobs with equal levels cancel in the middle; a blob with the higher level goes through and hits; two finishing hits in one frame give a draw with equal levels, and a win for the higher level otherwise. On the webcam: both punch at once, with and without a power item.

### 7. Dodge by stepping forward

**What.** A player dodges a normal attack by stepping forward, towards the camera, off the line both players stand on. The paint blob (feature 5) then flies past behind them. The game sees the step as a change in the player's depth (`Player.pos`). For the line to mean the same for both players, the calibration also asks them to stand level with each other, side by side at the same distance from the camera.

**Owners.**

- Task 4: the fight line, the dodge state and the hit rule in `interaction/game.py` (the line-up in the calibration is done).
- Task 3: the depth (`Player.pos`, done in the game start) and a faster depth signal for the dodge (below).
- Task 5: the dodge's look in the renderer.

**Design.**

- **Line up in the calibration:** done (`merge_log.md`, "Game start: Modifications"). The neutral pose is only measured while the two players stand level by depth (within 0.4 m); the nearer player is told to step back. Gestures also count only while the player faces the camera (same section).
- **The fight line:** each player's own depth when the neutral pose is measured (`depth_cal`), kept for new rounds like the other calibration values.
- **Dodge state,** per player, from the player's own change in depth. depth / depth_cal = torso_cal / torso (`Player.pos`, game start step 6). It is a ratio of the same person's own measurements, so the assumed field of view and shoulder width cancel out.
  - **Dodging** once the player is at least ~0.5 m closer than their line, i.e. depth < depth_cal − `dodge_step_m`. At 2.5 m that is 20% closer, about four times the depth noise.
  - **Back on the line** under ~0.3 m (hysteresis), so a dodge does not flicker on and off at the threshold.
  - **At most ~1 s per dodge;** then the player has to return to the line before the next one. Otherwise standing forward would make a player safe for good. While dodging, the player cannot attack (the one-action-at-a-time rule).
- **Fast enough to react.** A blob crosses the frame in ~1 s, and `identity.pos_tau_s` (0.3 s) delays the smoothed depth. The dodge reads its own depth signal with a shorter time constant (~0.1 s), or the torso ratio directly. Leaning forward shortens the torso in the image, which reads as further away, so it never fakes a dodge. If the hips leave the frame up close, the depth falls back to the shoulder width, which turning also changes.
- **Hit rule:** a normal blob passes a dodging player without hitting them, and flies on. The final smash is not dodged by stepping forward: its paint falls on every depth, and stepping sideways is its dodge (feature 2).
- **Look:** while dodging, the player's outline flashes white. A blob that passes them is drawn behind their body (the body mask of feature 5), with a short "MISS!" at `above_head`.

**Check.** On the webcam, at measured distances:

- a quick step forward of ~0.5 m during a blob's flight makes it miss;
- standing still, punching to the side, turning, or leaning forward never counts as a dodge;
- a dodge ends after ~1 s.

Log `p<pid>_depth_m` and the dodge state per player (`log_fields`), and count false dodges on the clips.

## Technical follow-ups from the game and renderer merges

| Item | Why | Where |
| --- | --- | --- |
| Balance on the webcam | Damage, speeds and the shield time are `game.py`'s values; check them over a few real rounds | `game:` config |
| `tools/eval_gestures.py` | Needs clips with real, labelled gestures; the test clips have none | `tools/` |
| Webcam check of the game start | The intro was checked on clips only: two players through the start places, calibration, countdown, result and replay; the P1-left swap with players who walked in the other way round; the line-up (whether 0.4 m by depth is too strict or too loose for two real players); the facing rule (whether 30° for head and body blocks players who glance at the opponent) | `--until game` |
| Depth on the webcam | `Player.pos` uses an assumed field of view and shoulder width; stand at a measured distance (e.g. 2 m), compare `p<pid>_depth_m` in the log, and set `camera_hfov_deg` from the camera's spec and `shoulder_m` to match | `identity:` config |
| Render cost at high resolution | The pixel look costs ~14 ms per frame on a 4K clip (~4.5 ms at 720p) | `render.pixel_size`, `pixelate()` |

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
