Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. Features move to [merge_log.md](merge_log.md) once built: so far the game start (["Game start"](merge_log.md#game-start-instructions-and-pose-calibration)), the face effects (["Face effects"](merge_log.md#face-effects-team-colour-hit-reaction-status-above-the-head-cheek-bulge)) and the consumables (["Consumables"](merge_log.md#consumables-drinks-caught-with-the-mouth)). Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

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
| Game rules: actions, hits, damage, round end | `interaction/game.py`: `Game.update()` → `_act()` (events to actions, `facing()`) and `_drink_with_mouths()` (mouths from `_track_mouth()`, buffs by `_apply_drink()`) → `advance()` (movement, drinks falling in `_advance_drinks()`, hits, `_apply_damage()` with the buffs, `_tick_buffs()`); state in `GamePlayer` (also `attack`, `defense`), `Projectile`, `PaintBeam`, `Shield`, `PaintSplat`, `Drink`; this frame's `hits` and `drunk` | `game:` | `tests/test_game.py` (`frame_state()` builds fake frames) |
| How the game looks | `scene/renderer.py`: `Renderer.draw()` (order: camera, paint, `draw_game()`, face effects, `_draw_status()` (name tag, crown, dizzy stars), health bars, `_draw_hud()`, `_draw_game_over()`); sprites in `scene/fire_bolt.py`, `magic_beam.py`, `fire_ring.py` (frames loaded by `scene/animation.py`, `load_frames()`); paint on the bodies in `scene/paint.py` (`BodyPaint`, `splash_shapes()`, `body_mask()`); helpers in `scene/draw.py` (`player_color()`, `blend_rgba()`, `shift_hue()`); intro screens in `_draw_start_place()`, `_draw_instructions()` (stick figures and move texts in `scene/instructions.py`); drinks in `_draw_drinks()` (sprites from `drink_sprites()`, `draw_badge()`), `_draw_popups()` (pick-up text), `_draw_buffs()` (icons above the health bars), colours in `DRINK_COLORS` | `render:` | `tests/test_renderer.py` |
| Face effects and face data | `face/effects.py`: `FaceEffects.update()` (game hits, players holding a drink) and `draw()` (war paint, cheek bulge, mustache, flash, team hat, flying hats), per-player `_Anim`, `bulge(pid, t)` on drinking; `face/tracker.py` (`Face`, anchors), `face/head_pose.py` (3D anchor points), `face/landmarker.py` (blendshapes) | `face:` | `tests/test_face.py`, `test_effects.py` |
| Players and identity | `identity/tracker.py`; `Player.status`, `reentered`, `pos` (`_position()`); P1 on the left at calibration (`order_left_to_right()`) | `identity:` | `tests/test_identity.py` |
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

Renumbered three times: twice on 2026-10-28 (the second time when the face effects were built), and on 2026-10-29 when the consumables were built. Old numbers in commits and `merge_log.md` map as follows:

| Now | Was after the face effects (2026-10-28) | Was after the first renumbering (2026-10-28) | Was before |
| --- | --- | --- | --- |
| Game start: built, recorded in `merge_log.md` ("Game start") | — | — | 12 |
| Face effects: built, recorded in `merge_log.md` ("Face effects") | — | 1 | 1 (team colour), 2 (hit reaction), 3 (status above the head), 7 (cheek bulge) |
| Consumables: built, recorded in `merge_log.md` ("Consumables") | 2 | 3 | 6 |
| 1. Final smash | 1 | 2 | 4 (super charge, reset by blinking), 15 (orb in four parts), 16 (paint from above) |
| 2. Aiming the normal attack | 3 | 4 | 11 |
| 3. Normal attack as splashed paint | 4 | 5 | 13 (paint stains where the hit landed) |
| 4. Both players attacking at once | 5 | 6 | 14 |
| 5. Dodge by stepping forward | 6 | 7 (added on 2026-10-28) | — |

Dropped (numbers from before the first renumbering): 5 (taunt), 8 (learned punch classifier: the gestures stay rule-based; its limits are in [limitations.md](limitations.md)), 9 (charge bar: the orb unlocks the final smash) and 10 (weapon pick-ups).

### 1. Final smash: a paint orb in four parts, raining until the caster blinks

**What.** The major attack becomes a final smash, like the Smash Ball in Smash Bros. Players collect a paint orb in four parts. With all four, hands up starts the attack: the orb flies up, bursts, and paint rains down on the opponent. **The rain lasts only while the caster does not blink:** the first blink ends it, and the orb is used up, so the player has to collect four new parts. Attacking becomes a staring contest that the other player can try to break. The opponent has two defences: getting out from under the direct paint drop, which is possible but very hard, or the shield. It replaces the magic beam.

**Builds on** (state on 2026-10-28):

- **Attack now:** `hands_up` fires the magic beam: `Game.beam_attack()`, `PaintBeam`, the `beam_*` keys in `game:`, drawn by `scene/magic_beam.py` with the `beam_ball` / `beam_band` keys in `render:`. Texts that name it: `CONTROLS` in `scene/renderer.py` ("HANDS UP: MAGIC BEAM") and `MOVE_TEXT["hands_up"]` in `scene/instructions.py`.
- **Normal attack until feature 3 is built:** the fire bolt (`Projectile`). "Blob hits" below means bolt hits until then.
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
  - **Collecting:** a wrist (smoothed keypoint, `state.motion`) within ~0.5 shoulder widths of the part, held for ~0.2 s, picks it up.
  - **Holding:** each player's parts show as a 4-slot orb icon by their health bar, in their team colour; the quarters fill in as they are collected. A player cannot hold more than four.
  - **Dropping:** 3 blob hits (feature 3) in a row knock one part out of the player; it flies off and lands as a part to collect again, so the other player can steal it.
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
- **Hits.** Each drop tests its path since the last frame against the target's box, as the blob does. Every hit adds a splash on the body (feature 3) and a little damage (e.g. 1 per drop, ~5 per second of rain; the beam did 12 per attack). The shield blocks the drops while it is up. It lasts `shield_s` (1 s), so it covers part of a long rain, not all of it.
- **Look.**
  - **Eye glow:** while the rain lasts, the caster's eyes glow in their team colour, so both players can see what ends it. On the blink, the glow shatters into particles, with a short "blink!" text at `above_head`.
  - **Rain:** drops in the caster's team colour, falling with gravity, slightly random sizes; splash sprites where they hit the floor (the bottom of the body box, or the frame bottom) or the player.
- **Replaces the beam.** `hands_up` no longer fires the magic beam. The `beam_ball` / `beam_band` drawing goes; the magic ball frames can stay for the burst. The calibration panel (game start, `merge_log.md`) then shows hands up as the final smash, not "MAGIC BEAM".

**Check.**

- Orb: parts only disappear when collected or used; a player cannot hold more than four; a dropped part can be picked up by either player; `hands_up` does nothing without four parts.
- Blink: on the webcam, a deliberate blink always ends the rain; looking down at the floor for 2 s does not; turning away ends it. Log blinks and the rain's length per attack, and count false blinks on the clips.
- Attack: the rain follows the opponent at the set speed; standing still takes every drop, a quick long step out avoids some; the rain never lasts past the cap; the shield blocks the drops while it is up.

### 2. Aiming the normal attack: a pointing laser and paint blobs towards the opponent

**What.** The normal attack (a punch fires a paint blob, feature 3) no longer flies only sideways: it goes where the player aims, towards the opponent. A thin pointing laser from the hand shows the aim before firing. Only the normal attack is aimed; the final smash (feature 1) falls on the opponent and needs no aim.

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
  - The splash (feature 3) uses the 2D impact point.
- **Pointing laser.** While an arm is extended (straight but not yet a punch), a thin dashed line in the team colour runs from the wrist along the aim direction until it meets a body box or the frame edge. It fades with the gesture's state, and it is drawn only for the aiming player, to keep the frame readable.
- **Drawing.** The blob's trail follows the velocity vector, so the drawn blob needs no rotated sprite.

**Check.** Blobs hit an opponent who stands higher or lower (e.g. crouching) or in the same half of the frame; the pointing laser matches where the blob actually flies; aim assist does not snap to a player behind the shooter.

### 3. Normal attack as splashed paint that covers the body

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
- **Cost.** The shapes are generated and drawn into a small sprite once per hit. Each frame only places each sprite on the body with one affine warp and blends the paint (step 2: one warped paint layer for the whole torso was tried and dropped; step 3: filling every polygon every frame was too slow). Target: under ~2 ms per player, measured in the run log.
- **Config.** Blob size and wobble, spike and droplet counts, and drip speed in `render:`; the hit band in `game:`.

**Decisions (Task 5 owner, 2026-10-29).**

- **Where the splash lands:** at the hit height, on the torso's side facing the attacker, with a random step inward. The blob flies sideways, so it always enters the body box at the near edge, often at an outstretched arm. The spikes fly on in the blob's direction, so paint builds up from the near side across the shirt.
- **Shading: partial.** The paint keeps its team colour and darkens only partly in the clothes' shadows, so it stays visible on a black shirt.
- **A blocked blob** bursts into falling drops on the shield (this replaces the fire bolt's break-up). At the frame edge it just leaves.
- **Beam hits splash too,** with the same generator, until the final smash replaces the beam: every damaging hit adds a splash.
- **Hits stay as they are:** the blob still flies sideways (aiming is feature 2) and hits the body box; only where the paint goes and how it looks change.
- **One player.** A blob that hits someone needs an opponent, so key `b` throws a test blob at P1 from the far side of the frame, in P2's colour. With `--solo` or `--no-intro` alone, punches fire blobs, and `b` shows the hit, the splash on the body, its drips and the body filling with paint. Each step also has a test with one player.

**Implementation steps.** One step at a time, each checked by the Task 5 owner before the next. Every new number goes in the module's config section with a reason.

1. **Splashes in torso coordinates** (Task 4, `interaction/game.py`; Task 5, `app/main.py`). *Done.*
   - `GamePlayer.torso`: the four corners (both shoulders, both hips) from the smoothed keypoints, each frame, like `bounds`.
   - `PaintSplat` becomes one splash per hit: (`u`, `v`) on the torso, the flight direction, a seed, the attacker and the hit time. `u` is the near side plus a random step inward, `v` the hit height between the shoulder line and the hips, both kept in the torso band `splat_band_u` / `splat_band_v`. These replace `splat_extra`, `splat_spread` and `splat_band`.
   - `Game.test_hit()` and key `b`: a test blob at P1 from the far side.
   - Drawn for now as the old pixel crosses, placed by the bilinear map over the four corners, so they follow the body. In `--until game`, the torso corners and the splashes as dots.
   - As built:
     - `core/geometry.py`, `bilinear(quad, u, v)`: the point at (u, v) in a four-corner quad.
     - `Game.torso_corners(m)`: `R_SHOULDER`, `L_SHOULDER`, `L_HIP`, `R_HIP`, so u = 0 is the screen-left side for a player facing the camera (the model's `L_*` is on the screen's right, `core/skeleton.py`). With the hips not both seen, the shoulders are moved down to `torso()`'s estimated hip level. `update()` stores it in `GamePlayer.torso`; None without a usable shoulder width.
     - `PaintSplat(u, v, owner_id, direction, seed, t)`. `_splash()` makes one per damaging hit (bolt or beam): u at the near band edge plus a random 0–`splat_inward_u` inward, v from the hit height between the shoulder line and the hips. The near side comes from the corners' x and the flight direction. Without a torso: v = 0.5. Seeded per target and splash count, as before.
     - `Game.test_hit(pid)`: a normal bolt from the far frame edge at `test_hit_v` down P1's torso, owned by the other team (P2's blue), action `test_blob`. Only while playing with the player in view. `Pipeline.test_hit()` and key `b` in `app/main.py`; README and the `main.py` docstring list the key.
     - Renderer: `_draw_player_paint(canvas, torso, splats, coverage)` puts each cross at `bilinear(torso, u, v)`, sized by the torso's width and height; `paint_area()` is gone. Paint is drawn only for players with a torso this frame.
   - New `game:` keys: `splat_band_u`, `splat_band_v`, `splat_inward_u`, `test_hit_v`; gone: `splat_extra`, `splat_spread`, `splat_band`.
   - On `session.mp4` (`--no-intro`): the same round as before (7 and 10 hits, P1 wins with 28 HP). Every splash is on the side facing the attacker (P1's at u 0.22–0.38, P2's at 0.59–0.86). 10 of 17 hits arrive at or above the shoulder line (all 7 on P1), so v is 0.05, the top of the band: punches arrive at shoulder height. Kept as it is (Task 5 owner, 2026-10-29): with the spikes and drips of step 3 it reads as paint hitting the upper chest and running down.
   - One player: on `shield.mp4` with a test blob every 2 s, 5 of the blobs hit (the rest met the shield in this clip of shield attempts); the paint sits on the near side of the coat and moves with it.
2. **Paint on the body and body mask** (Task 5, `scene/paint.py`, `scene/renderer.py`). *Done.*
   - Each player's splashes are kept as polygons in torso coordinates, made once when a hit lands. Every frame their points are mapped onto the body with the bilinear map over the four corners; points outside the torso are extrapolated, so paint can run onto the arms.
   - The paint is clipped to a body mask: the torso quad plus a capsule along each upper and lower arm and leg, with widths in shoulder widths.
   - Partial shading from the camera's brightness; render time per player in the log.
   - Plain discs as splashes until step 3.
   - As built:
     - Changed from the plan: no paint layer image warped onto the body. A warp through the four corners (`cv2.warpPerspective`) treats the torso's trapezoid (hips narrower than shoulders) as a slanted rectangle seen in perspective, and on a test body it put the middle of the torso 42 px too low. Mapping the polygons' points with `bilinear()`, the game's own map, is exact and costs less than 1 ms per player.
     - `scene/paint.py`, `BodyPaint` per player (`Renderer._paint`): `update()` makes the polygons again only when the splashes or the coverage change; `draw()` maps them, fills them in a region around them (colour without anti-aliasing plus a 2 px outline; coverage anti-aliased), multiplies the coverage by the body mask and `paint_opacity`, shades and blends. Nothing for a torso narrower than `paint_min_torso_px` (side-on) or folded.
     - `splash_polygons()`: for now a disc of radius `splash_r` = [base, growth] torso widths (growing as health drops, like the crosses did), its v divided by `paint_aspect` so it is about round on the body.
     - `body_mask()`: the torso quad stretched to `mask_torso_u` / `_v` (the keypoints are joints, inside the outline), and a line of `mask_arm_sw` / `mask_leg_sw` shoulder widths along every arm and leg bone with both keypoints valid; anti-aliased.
     - `_shade()`: factor (1 − `paint_shade`) + `paint_shade` × the brightness over its mean under the paint, the ratio capped at 2. A dark shirt keeps the paint bright; only folds and shadows relative to the shirt show.
     - `Renderer.log_fields()`: `render_p<pid>_paint_ms`; the pipeline now also asks the renderer for log columns. `Renderer.reset()` forgets the paint shapes.
   - New `render:` keys: `paint_aspect`, `splash_r`, `mask_torso_u`, `mask_torso_v`, `mask_arm_sw`, `mask_leg_sw`, `paint_min_torso_px`, `paint_shade`, `paint_opacity`; gone: `splat_cell_w`, `splat_cell_h`.
   - Torso height over width measured (median): 1.82 and 1.74 on `session.mp4`, 1.46 on `shield.mp4` (hips often estimated there), so `paint_aspect` is 1.7. Torsos are 102–236 px wide on `session.mp4` and 178–390 px on `shield.mp4` (5th–95th percentile).
   - On `session.mp4` (`--no-intro`): paint time per player median 0.52 and 0.66 ms, 95th percentile 0.82 ms, at most 2.3 ms; the render stage went from 5.07 to 5.68 ms per frame on average. The discs sit on the chests and stop at the body's outline; paint near a raised arm runs onto it. On `shield.mp4` alone with test blobs, the discs stay on the coat as it moves.
3. **Splash shapes** (Task 5, `scene/paint.py`). *Done.*
   - The generator, seeded per hit: a wobbly central blob, 5–10 tapering spikes mostly in the flight direction, power-law droplets, and 1–3 drips that run down over ~1 s.
   - Splashes grow as health drops, so a player near 0 HP is mostly paint.
   - As built:
     - `splash_shapes(seed, cfg)`, in units of the blob's radius with x along the flight and y down:
       - `blob()`: r(a) = R (1 + Σ aₖ sin(k a + φₖ)) for k = 2…5, with aₖ up to `splash_wobble` / k;
       - `splash_spikes` wedges, each from inside the blob out to 1 + `spike_len` radii with a drop at the tip. A share `spike_forward` aims within `spike_spread_rad` of the flight, the rest anywhere;
       - `splash_drops` droplets at `drop_dist`, aimed the same way, with Pareto radii (`drop_r`, exponent `drop_power`): many small, few large;
       - `drips` from the lower edge, `drip_len` long and `drip_w` wide.
     - The blob radius is `splash_r` = [base, growth] torso widths, base + growth × (1 − health / max_health). Every splash swells as the player loses, so earlier splashes grow too.
     - Changed from step 2: each splash is drawn once into a coverage sprite (`paint_sprite_px` per blob radius, cropped to its paint).
       - Every frame it is placed with one `cv2.warpAffine`, using the local affine map of `bilinear()` at the splash's centre (`BodyPaint._affine()`: the map's value and its slopes along u and v there). The centre lands exactly at its (u, v), and the shape follows the torso's local stretch and lean.
       - Mapping and filling every polygon every frame (~30 per splash) took 4–9 ms per player with 10 splashes on `session.mp4`.
       - While a splash's drips run (`drip_s`, eased out), they are filled as polygons with the same map, then baked into its sprite.
     - Fix of a step 2 bug: `cv2.fillPoly` (and `drawContours`) given several overlapping polygons fills them even-odd and leaves holes where they overlap; two overlapping discs of one colour had a hole. `fill_polygons()` now fills one polygon per call.
     - The flight's sign along u is fixed when a splash is first drawn, from the torso then.
     - Paint from several attackers: one coverage image per colour, later colours over earlier ones. With one attacker (the usual case) the colour is applied per channel with `cv2.convertScaleAbs`.
     - The per-pixel work (mask, shading, blend with `cv2.blendLinear`) runs only over the painted pixels.
     - `BodyPaint.draw()` takes `Frame.t` for the drips; splashes cleared by a new round are forgotten.
   - New `render:` keys: `splash_wobble`, `splash_spikes`, `spike_len`, `spike_width`, `spike_forward`, `spike_spread_rad`, `splash_drops`, `drop_dist`, `drop_r`, `drop_power`, `drips`, `drip_len`, `drip_w`, `drip_s`, `paint_sprite_px`, `paint_sprite_extent`; `splash_r` is now the blob's radius ([0.1, 0.15]).
   - On `session.mp4` (`--no-intro`):
     - Paint time per player, median by hits taken: 0.4 ms after 1 hit, 0.8 ms after 4 and 1.4–1.6 ms after 7–10; 95th percentile 1.7–1.8 ms. Single frames reach 7 ms when a new splash is generated.
     - The splashes spray onward from the attack, drip down the shirts and stop at the body's outline; at the end of the round P1's upper chest is mostly blue.
   - One player: on `shield.mp4` with test blobs, the spray leans the way the blob flew, and a splash at the coat's edge is cut by the mask.
4. **The blob in flight and the burst** (Task 5; Task 4 for the end reason). *Done.*
   - The wobbling blob with trailing droplets and a highlight replaces `FireBolt` for the normal attack; on a shield it bursts into falling drops.
   - Texts: "PUNCH: PAINT BLOB" in `CONTROLS` and `MOVE_TEXT["punch"]`.
   - `scene/fire_bolt.py` and its `bolt_sprite*` keys go, if nothing else uses them.
   - As built:
     - `scene/paint.py`, `draw_blob()`: `blob()` with wobble amplitudes `blob_wobble` / k (k = 2…4) whose phases turn at `blob_wobble_hz` × k turns a second, stretched `blob_stretch` along the flight. It has a darker rim, a highlight towards the front top, and `blob_trail` droplets behind it, each `blob_trail_gap` radii further back and sinking `blob_trail_fall` × i² radii. Drawn opaque, anti-aliased, `blob_r_w` frame widths in radius.
     - `draw_burst()`: `burst_drops` drops thrown back and up (within `burst_spread_rad` of 45° up-back) at `burst_speed` radii a second, falling with `burst_gravity` and shrinking to nothing over `game.bolt_end_s`. Seeded by where the blob ended, so the drops are the same in every frame.
     - `interaction/game.py`: `BoltEnd.reason` is "hit", "shield" or "edge". The renderer bursts only on "shield": at a hit the splash on the body is the effect, and at the frame edge the blob just leaves. `bolt_end_s` went from 0.25 s (the fire bolt's break-up frames) to 0.5 s, so the drops have time to rise and fall.
     - Bursts are drawn after the shields. Drawn before, the fire ring covered most drops on `shield.mp4`, and at 4–9 radii/s they stayed under it; at 8–16 they clear it.
     - The game logic keeps its names (`Projectile`, `bolt_*` keys, `game_bolts` in the log); only the look and the texts changed.
     - Removed: `scene/fire_bolt.py`, `tests/test_fire_bolt.py`, `Renderer.bolt_effect()` and the `bolt_sprite`, `bolt_sprite_launch`, `bolt_sprite_flight`, `bolt_sprite_fps`, `bolt_sprite_h`, `bolt_effect_owner` keys. `assets/beam/` (the fire blast frames) is no longer used; it and its README credit row can go (follow-ups).
   - New `render:` keys: `blob_r_w`, `blob_wobble`, `blob_wobble_hz`, `blob_stretch`, `blob_trail`, `blob_trail_gap`, `blob_trail_fall`, `burst_drops`, `burst_spread_rad`, `burst_speed`, `burst_gravity`, `burst_drop_r`.
   - On `session.mp4` (`--no-intro`): the same round (P1 wins with 28 HP). Render 4.4 ms per frame (median) with blobs in flight and 5.3 ms without. The blob is a blue or red wet ball about a fist across, with its droplets trailing.
   - One player: on `shield.mp4` with test blobs, the blobs fly in from the far side, and the ones stopped by the shield burst into drops thrown back off the ring.
5. **Check and docs.** Clips (`session.mp4` with `--no-intro` has real hits; `shield.mp4` alone with test blobs), the render time, then the feature moves to `merge_log.md`.

**Check.** A punch that hits at shoulder height leaves paint near the shoulder, and the paint stays there while the player moves their arms and leans; no paint outside the body mask; no two splashes look the same; the render cost per frame is logged.

### 4. Both players attacking at the same time: the stronger attack gets the shot

**What.** Make simultaneous attacks fair and visible, and link them to the consumables (built: [merge_log.md, "Consumables"](merge_log.md#consumables-drinks-caught-with-the-mouth)): an attack drink raises a player's attack multiplier, and when both attack at once, the higher attack multiplier gets the shot. Today (`Game.advance()` in `interaction/game.py`, as in `game.py`):

- bolts and beams fly through each other, and both hits land;
- when both hits would finish the round in the same frame, the attack fired first wins: `_apply_damage()` ignores everything once `winner_id` is set, so the order of the `projectiles` list decides.

**Owners.** Task 4: collisions, attack levels and the tie rule in `interaction/game.py`. Task 5: the clash effects in the renderer.

**Design.**

- **Attack level:** the attack multiplier from the consumables (`GamePlayer.attack`), 1 without an attack drink.
- **Blob meets blob:** two blobs (feature 3) of different players whose paths cross in a frame (x segments overlap, y within the blob's size):
  - the blob with the higher attack level flies on and hits; the other bursts at the meeting point, a splash in mid-air in its colour;
  - with equal levels, both burst, with a splash in both team colours.
- **Final smash** (feature 1): it falls from above, so blobs never meet it, and a blob fired at a player under the rain still hits. If both players fire one, both fall; each player's shield blocks the drops on them.
- **Final blows in the same frame:** apply all of a frame's damage first, then decide: if both are at 0, the higher attack level wins; with equal levels it is a draw ("DRAW", both colours). Otherwise the one with health left wins. Needs a `result` instead of only `winner_id` (also used by the game start's result screen, `merge_log.md`).

**Check.** Unit tests: two blobs with equal levels cancel in the middle; a blob with the higher level goes through and hits; two finishing hits in one frame give a draw with equal levels, and a win for the higher level otherwise. On the webcam: both punch at once, with and without a power item.

### 5. Dodge by stepping forward

**What.** A player dodges a normal attack by stepping forward, towards the camera, off the line both players stand on. The paint blob (feature 3) then flies past behind them. The game sees the step as a change in the player's depth (`Player.pos`). For the line to mean the same for both players, the calibration also asks them to stand level with each other, side by side at the same distance from the camera.

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
- **Hit rule:** a normal blob passes a dodging player without hitting them, and flies on. The final smash is not dodged by stepping forward: its paint falls on every depth, and stepping sideways is its dodge (feature 1).
- **Look:** while dodging, the player's outline flashes white. A blob that passes them is drawn behind their body (the body mask of feature 3), with a short "MISS!" at `above_head`.

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
| Render cost at high resolution | The pixel look costs ~14 ms per frame on a 4K clip (~4.5 ms at 720p); the face effects and status add ~0.6 ms at 720p (render 4.3 → 5.0 ms on `session.mp4`) | `render.pixel_size`, `pixelate()` |
| Webcam check of the drinks | Checked on clips (no drinking) and in tests; one webcam run so far (`logs/run-20261028-231534.csv`, which set `drink_hold_s` to 2.5 s). Check alone (`--solo`) and with two: catching and drinking, whether `jawOpen` 0.5 / 0.25 and the 2.5 s hold fit, whether players know to close the mouth (a "CLOSE YOUR MOUTH" text while holding is an option), the buff values and the spawn rate | `game:` `drink_*` keys |
| Shield render time | On `shield.mp4` (2026-10-29, TeamViewer running), the render took ~51 ms on frames with the shield up and ~3 ms without; the same with drinks turned off. Not checked why yet | `scene/fire_ring.py`, `FireRing` |
| Asset packs in the repo | All of `assets/Fancy Drinks/` (paid; reselling or redistributing the files as assets is not allowed), including `Misc/`, and `assets/Jester/` are committed. If the repo or the final ZIP is shared, keep only the nine sprites in use, or keep the packs out of git with a download note in the README. `assets/beam/` (the fire blast, purchased) is unused since the paint blob (feature 3, step 4): delete it and its README row | `assets/`, `.gitignore`, `README.md` |
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
