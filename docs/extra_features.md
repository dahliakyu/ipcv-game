Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. Features move to [merge_log.md](merge_log.md) once built: so far the game start (["Game start"](merge_log.md#game-start-instructions-and-pose-calibration)) and the face effects (["Face effects"](merge_log.md#face-effects-team-colour-hit-reaction-status-above-the-head-cheek-bulge)). Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

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
| How the game looks | `scene/renderer.py`: `Renderer.draw()` (order: camera, paint, `draw_game()`, face effects, `_draw_status()` (name tag, crown, dizzy stars), health bars, `_draw_hud()`, `_draw_game_over()`); sprites in `scene/fire_bolt.py`, `magic_beam.py`, `fire_ring.py` (frames loaded by `scene/animation.py`, `load_frames()`); helpers in `scene/draw.py` (`player_color()`, `blend_rgba()`, `shift_hue()`); intro screens in `_draw_start_place()`, `_draw_instructions()` (stick figures and move texts in `scene/instructions.py`) | `render:` | `tests/test_renderer.py` |
| Face effects and face data | `face/effects.py`: `FaceEffects.update()` (game hits) and `draw()` (war paint, cheek bulge, mustache, flash, team hat, flying hats), per-player `_Anim`, `bulge(pid, t)` for eating; `face/tracker.py` (`Face`, anchors), `face/head_pose.py` (3D anchor points), `face/landmarker.py` (blendshapes) | `face:` | `tests/test_face.py`, `test_effects.py` |
| Players and identity | `identity/tracker.py`; `Player.status`, `reentered`, `pos` (`_position()`); P1 on the left at calibration (`order_left_to_right()`) | `identity:` | `tests/test_identity.py` |
| Keys, flags, pipeline order | `app/main.py` (`q`, `p`, `r` new round from the countdown, `R` full reset to waiting; `--no-intro`, `--solo`), `app/pipeline.py` | | `tests/test_pipeline.py` |

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

Renumbered twice on 2026-10-28, the second time when the face effects were built. Old numbers in commits and `merge_log.md` map as follows:

| Now | Was on 2026-10-28 | Was before |
| --- | --- | --- |
| Game start: built, recorded in `merge_log.md` ("Game start") | — | 12 |
| Face effects: built, recorded in `merge_log.md` ("Face effects") | 1 | 1 (team colour), 2 (hit reaction), 3 (status above the head), 7 (cheek bulge) |
| 1. Final smash | 2 | 4 (super charge, reset by blinking), 15 (orb in four parts), 16 (paint from above) |
| 2. Consumables (drinks, redesigned 2026-10-28) | 3 | 6 |
| 3. Aiming the normal attack | 4 | 11 |
| 4. Normal attack as splashed paint | 5 | 13 (paint stains where the hit landed) |
| 5. Both players attacking at once | 6 | 14 |
| 6. Dodge by stepping forward | 7 (added on 2026-10-28) | — |

Dropped (numbers from before the first renumbering): 5 (taunt), 8 (learned punch classifier: the gestures stay rule-based; its limits are in [limitations.md](limitations.md)), 9 (charge bar: the orb unlocks the final smash) and 10 (weapon pick-ups).

### 1. Final smash: a paint orb in four parts, raining until the caster blinks

**What.** The major attack becomes a final smash, like the Smash Ball in Smash Bros. Players collect a paint orb in four parts. With all four, hands up starts the attack: the orb flies up, bursts, and paint rains down on the opponent. **The rain lasts only while the caster does not blink:** the first blink ends it, and the orb is used up, so the player has to collect four new parts. Attacking becomes a staring contest that the other player can try to break. The opponent has two defences: getting out from under the direct paint drop, which is possible but very hard, or the shield. It replaces the magic beam.

**Builds on** (state on 2026-10-28):

- **Attack now:** `hands_up` fires the magic beam: `Game.beam_attack()`, `PaintBeam`, the `beam_*` keys in `game:`, drawn by `scene/magic_beam.py` with the `beam_ball` / `beam_band` keys in `render:`. Texts that name it: `CONTROLS` in `scene/renderer.py` ("HANDS UP: MAGIC BEAM") and `MOVE_TEXT["hands_up"]` in `scene/instructions.py`.
- **Normal attack until feature 4 is built:** the fire bolt (`Projectile`). "Blob hits" below means bolt hits until then.
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
  - **Collecting:** a wrist (smoothed keypoint, `state.motion`) within ~0.5 shoulder widths of the part, held for ~0.2 s, picks it up, as for the items in feature 2.
  - **Holding:** each player's parts show as a 4-slot orb icon by their health bar, in their team colour; the quarters fill in as they are collected. A player cannot hold more than four.
  - **Dropping:** 3 blob hits (feature 4) in a row knock one part out of the player; it flies off and lands as a part to collect again, so the other player can steal it.
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
- **Hits.** Each drop tests its path since the last frame against the target's box, as the blob does. Every hit adds a splash on the body (feature 4) and a little damage (e.g. 1 per drop, ~5 per second of rain; the beam did 12 per attack). The shield blocks the drops while it is up. It lasts `shield_s` (1 s), so it covers part of a long rain, not all of it.
- **Look.**
  - **Eye glow:** while the rain lasts, the caster's eyes glow in their team colour, so both players can see what ends it. On the blink, the glow shatters into particles, with a short "blink!" text at `above_head`.
  - **Rain:** drops in the caster's team colour, falling with gravity, slightly random sizes; splash sprites where they hit the floor (the bottom of the body box, or the frame bottom) or the player.
- **Replaces the beam.** `hands_up` no longer fires the magic beam. The `beam_ball` / `beam_band` drawing goes; the magic ball frames can stay for the burst. The calibration panel (game start, `merge_log.md`) then shows hands up as the final smash, not "MAGIC BEAM".

**Check.**

- Orb: parts only disappear when collected or used; a player cannot hold more than four; a dropped part can be picked up by either player; `hands_up` does nothing without four parts.
- Blink: on the webcam, a deliberate blink always ends the rain; looking down at the floor for 2 s does not; turning away ends it. Log blinks and the rain's length per attack, and count false blinks on the clips.
- Attack: the rain follows the opponent at the set speed; standing still takes every drop, a quick long step out avoids some; the rain never lasts past the cap; the shield blocks the drops while it is up.

### 2. Consumables: drinks caught with the mouth

**What.** Drinks fall from the sky during the round. A player drinks one by catching it with their open mouth and closing it. Each drink gives one of three buffs: **HP**, **attack** or **defense**. Drinks fall in random sizes, and a bigger drink gives a bigger buff. The attack buff also decides who gets the shot when both players attack at once (feature 5).

**Owners.**

- Task 4: spawning, falling, the mouth rule (open, catch, close), the buffs and their effect on damage, in `interaction/game.py`.
- Task 1: mouth state (`jawOpen`, `anchors["mouth"]`); the cheek bulge on drinking (`FaceEffects.bulge()`).
- Task 5: drink sprites, category icons, buff display and pick-up text in the renderer.

**Design (decided 2026-10-28).**

- **Spawning.** Only while playing. Every 6–10 s (random), one drink appears at a random x over the play area, just above the top of the frame. At most two drinks are in the air at once.
- **Falling.** Slowly and at a constant speed (one frame height in ~4 s), so a player can get under it. A drink that falls out of the bottom of the frame is gone.
- **Size and value.** Each drink gets a random size factor (e.g. 0.7–1.4 × the base size, a fraction of the frame height). Its buff value scales linearly with the size, from the category's smallest to its largest value.
- **Drinking: only with the mouth.** There is no grabbing with the hands.
  - The mouth is open when `jawOpen` > 0.5, and closed again below 0.25 (hysteresis, per player).
  - A drink that touches the open mouth (within ~0.6 face widths of `anchors["mouth"]`, plus the drink's half size) sticks to the mouth and stops falling.
  - Closing the mouth drinks it. A drink not drunk within 1 s falls on again.
  - The mouth anchor follows head tilt, so players can lean to catch drinks. Talking does not drink anything that is not at the mouth.
  - Drinking puffs the drinker's cheeks: `FaceEffects.bulge(pid, t)` (built: [merge_log.md, "Face effects"](merge_log.md#face-effects-team-colour-hit-reaction-status-above-the-head-cheek-bulge)), its only trigger.
- **Buff categories.** Each uses its own drinks from `assets/Fancy Drinks/`, in colours that do not look like the team colours (red P1, blue P2), with an icon on the drink:

  | Category | Drinks | Icon | Effect (value from the size) |
  | --- | --- | --- | --- |
  | HP | green: Lime Lemonade, Cucumber Water, Power Up | green "+" | +10 to +30 HP at once, up to the maximum |
  | Attack | orange: Orange Juice, Tropic Cider, Poolside | orange lightning bolt | ×1.2 to ×1.6 damage dealt, for ~10 s |
  | Defense | purple: Midnight, Pomegranate, Neon Night | purple shield | 20% to 50% less damage taken, for ~10 s |

- **Stacking.** Another drink of the same category replaces the running buff's value and restarts its time.
- **One player.** Everything must be testable alone (asked 2026-10-28): with `--solo` (the intro with P1 only) or `--no-intro` with one player in view, drinks fall, can be drunk (cheek bulge, pick-up text) and their buffs show by the health bar and in the log (`game_p1_attack`, `_defense`). Only the attack and defense effect on damage needs an opponent; the tests cover it. An HP drink at full health heals nothing but still shows its text.
- **Damage with buffs:** damage = base damage × the attacker's attack multiplier × (1 − the target's defense), rounded.
- **Look.**
  - Falling drinks: the sprite at its size, with the category icon in a small badge.
  - On drinking: "+18 HP", "ATTACK ×1.4" or "DEFENSE 35%" rises from the drinker's head in the category colour.
  - Running attack and defense buffs: an icon with the value by the player's health bar, with a bar for the time left, fading in the last 2 s.
- **Sprites:** `assets/Fancy Drinks/` (64 × 64 drinks, by Savannah). Its Read Me states no licence: check the pack's download page, and credit it in the README before the final ZIP.

**Implementation steps.** One step at a time, each checked by the Task 5 owner before the next. Every new number goes in the module's config section with a reason.

1. **Falling drinks** (Task 4, `interaction/game.py`; plain shapes in `--until game`). *Done.*
   - `Drink` (position, category, size factor, value, sprite index) and spawning with a seedable random generator, only while playing, at most `drink_max` at once.
   - Constant-speed fall; gone below the frame; cleared on a new round and on reset.
   - Logged as `game_drinks`; drawn as circles with the category letter in `draw_debug()`.
   - As built: `Drink` has `x`, `y` (centre, px), `kind`, `size`, `value` and `variant` (0–1). With `variant`, the renderer picks one of the category's sprites, so the game needs no sprite list.
     - `_advance_drinks()` runs inside `advance()`, so only while playing and not after a win. It counts `_drink_wait` down and spawns a drink when it is due and fewer than `drink_max` are in the air; when it is full, the next drink comes as soon as there is room.
     - `_spawn_drink()`: a random category and size; the value is linear in the size; a random x between the `drink_margin_w` margins; the drink starts just above the frame. `drink_height()` gives its height on screen.
     - The random generator is seeded from `drink_seed` (null: different every game). `_clear_round()` clears the drinks and draws a new wait.
     - In `--until game`: a circle of the drink's size in the category colour (`DRINK_DEBUG_COLORS`: green, orange, purple), with its letter and value.
   - New `game:` keys, all with reasons: `drink_every_s`, `drink_max`, `drink_fall_s`, `drink_h`, `drink_size`, `drink_margin_w`, `drink_hp`, `drink_attack`, `drink_defense`, `drink_buff_s`, `drink_seed`. The buff values and time are used from step 3 on.
   - On `session.mp4` (`--no-intro`, seed 3): drinks at 7.0, 15.7, 25.7 and 32.3 s; at the base size a drink is about as tall as a player's head.
   - One player: drinks fall on `shield.mp4` (`--no-intro`, one player); a test runs the solo intro (`round_players: 1`) to playing and gets drinks, and one runs `--no-intro` with only P1.
2. **Drinking with the mouth** (Task 4, from Task 1's face data). *Done.*
   - Mouth open/closed per player from `jawOpen`, with hysteresis; a held drink follows the mouth anchor.
   - Catch, drink on closing, drop again after `drink_hold_s`.
   - `Game.drunk` (pid, category, value) for this frame, logged as the action `drink_<category>`. The renderer calls `FaceEffects.bulge()` for each.
   - As built:
     - `_track_mouth()` runs every frame for every player. The mouth opens above `drink_mouth_open` (0.5) and closes below `drink_mouth_closed` (0.25). A frame without a score (MediaPipe runs for each player in turn) keeps the state; a new face starts closed, and no visible face means closed.
     - `_drink_with_mouths()` runs only while playing, before `advance()`.
       - A drink touches the open mouth when its centre is within `drink_catch_fw` face widths of the mouth anchor plus half its height. Then it is held (`Drink.held_by`, action `catch`), it does not fall, and its rim (`drink_hold_rim` of its height above the centre) sits at the mouth.
       - Each player holds one drink at a time, and the nearer mouth wins.
       - Closing the mouth drinks it: `Drunk(pid, kind, value)` in `Game.drunk` for this frame, and the action `drink_<kind>`.
       - Held longer than `drink_hold_s`, or the face lost, the drink falls on, and that player cannot catch it again (`dropped_by`), so holding the mouth open does not keep a drink.
     - `reset(pid)` (re-entry) forgets the mouth and lets a held drink fall.
     - The renderer calls `FaceEffects.bulge()` for each drink drunk.
     - In `--until game`: each mouth's catch circle, filled while open; a held drink has a thick outline. Logged as `game_p<pid>_mouth_open` and `_holding` (the held drink's kind).
   - New `game:` keys: `drink_mouth_open`, `drink_mouth_closed`, `drink_catch_fw`, `drink_hold_rim`, `drink_hold_s`.
   - On `session.mp4` and `shield.mp4` (`--no-intro`, a drink every 2–3 s): no catch and nothing drunk; talking never reached a `jawOpen` of 0.5 (P1 at most 0.44, P2 0.38). Catching and drinking are covered by tests, also with one player; the clips have no drinking.
3. **Buffs** (Task 4). *Done.*
   - HP at once; attack and defense as timed multipliers in `_apply_damage()`; replace and restart.
   - Cleared on a new round. Logged as `game_p<pid>_attack` and `_defense`.
   - As built:
     - `GamePlayer.attack` / `attack_left` and `defense` / `defense_left`: 1 and 0 without a buff.
     - `_apply_drink()` runs as a drink is drunk. HP adds the value, rounded, up to `max_health`; attack and defense take the value and `drink_buff_s`, replacing a running buff of the same kind.
     - `_tick_buffs()` counts them down in `advance()`, so only while playing; an ended buff goes back to no effect.
     - `_apply_damage()`: damage = base × attacker's `attack` × (1 − target's `defense`), rounded, for bolts and beams alike. `Hit.damage` is the damage after the buffs.
     - `_refresh()` clears the buffs (new round, reset).
     - Logged as `game_p<pid>_attack`, `_attack_left`, `_defense`, `_defense_left`; `--until game` adds "atk x1.42 7s" and "def 35% 7s" to the player's label.
   - On `session.mp4` (`--no-intro`, nobody drinks): the round is the same as before the buffs (7 and 10 hits, P1 wins with 28 HP). The buffs are covered by tests, also for one player through the log.
4. **Look** (Task 5, `scene/renderer.py`). *Done.*
   - Drink sprites (loaded on first use, like the other sprites) with the category badge.
   - The held drink at the mouth, the pick-up text, and the buff icons by the health bars.
   - The README credit for the sprites.
   - As built:
     - `DRINK_COLORS` in `scene/draw.py` is the one table of buff colours (green, orange, purple), used by the badges, icons, text and `--until game`.
     - `drink_sprites(kind)` loads the config's sprites on first use, cropped to their visible part.
     - `_draw_drinks()`: each drink as one of its kind's sprites (`Drink.variant` picks it), scaled to its height with the nearest pixel. Without sprites (`drink_sprites: null`) it is a circle. `draw_badge()` puts the buff's icon (`ICONS`: a plus, a lightning bolt, a shield) on a light disc at the drink's upper right.
     - Drinks are drawn after the face effects, so a drink held at the mouth is in front of the mustache. They are not drawn on the result screen.
     - `_draw_popups()`: "+18 HP", "ATTACK x1.4" or "DEFENSE 35%" in the buff's colour, starting at the mouth and offset `popup_side_fw` face widths towards the middle of the frame, so it does not cover the face. It rises `popup_rise_h` over `popup_s`, fades over the last third and stays under the top bar.
     - `_draw_buffs()`: above each health bar (P1 from the left, P2 from the right), each running attack or defense buff as its badge, its value and a time bar, fading over the last `buff_fade_s`. `Game.drink_buff_s` gives the full time.
     - `faded()` blends a box drawn in a `with` block at an opacity.
     - README: a credit row for the drinks in "Third-party licences".
   - New `render:` keys: `drink_sprites` (three drinks per kind), `drink_badge`, `drink_badge_min_px`, `popup_s`, `popup_rise_h`, `popup_side_fw`, `popup_text_scale`, `buff_icon_h`, `buff_fade_s`.
   - On `session.mp4` (a drink every 2–3 s; an attack drink put at P1's mouth, drunk on the next frame because the mouth was closed): falling drinks with their badges, the drink at P1's lips in front of the mustache, then "ATTACK x1.5" beside the head and the x1.5 icon with its time bar above P1's health bar. Render time 4.98 → 5.24 ms per frame with drinks in the air.
5. **Check and docs.** Clips for falling, sizes and the look; eating and the buffs in tests (the clips have no drinking); the webcam check is the Task 5 owner's, also alone (`--solo`). Then the feature moves to `merge_log.md`.

Every step is checked with one player too (a test, and `data/clips/shield.mp4`, a solo clip, where it applies).

**Check.** A drink cannot be drunk twice or by both players; talking without a drink at the mouth drinks nothing; each buff changes the damage as designed and ends on time; bigger drinks give bigger values.

### 3. Aiming the normal attack: a pointing laser and paint blobs towards the opponent

**What.** The normal attack (a punch fires a paint blob, feature 4) no longer flies only sideways: it goes where the player aims, towards the opponent. A thin pointing laser from the hand shows the aim before firing. Only the normal attack is aimed; the final smash (feature 1) falls on the opponent and needs no aim.

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
  - The splash (feature 4) uses the 2D impact point.
- **Pointing laser.** While an arm is extended (straight but not yet a punch), a thin dashed line in the team colour runs from the wrist along the aim direction until it meets a body box or the frame edge. It fades with the gesture's state, and it is drawn only for the aiming player, to keep the frame readable.
- **Drawing.** The blob's trail follows the velocity vector, so the drawn blob needs no rotated sprite.

**Check.** Blobs hit an opponent who stands higher or lower (e.g. crouching) or in the same half of the frame; the pointing laser matches where the blob actually flies; aim assist does not snap to a player behind the shooter.

### 4. Normal attack as splashed paint that covers the body

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

### 5. Both players attacking at the same time: the stronger attack gets the shot

**What.** Make simultaneous attacks fair and visible, and link them to the consumables (feature 2): an attack drink raises a player's attack multiplier, and when both attack at once, the higher attack multiplier gets the shot. Today (`Game.advance()` in `interaction/game.py`, as in `game.py`):

- bolts and beams fly through each other, and both hits land;
- when both hits would finish the round in the same frame, the attack fired first wins: `_apply_damage()` ignores everything once `winner_id` is set, so the order of the `projectiles` list decides.

**Owners.** Task 4: collisions, attack levels and the tie rule in `interaction/game.py`. Task 5: the clash effects in the renderer.

**Design.**

- **Attack level:** the attack multiplier from the consumables (feature 2), 1 without an attack drink.
- **Blob meets blob:** two blobs (feature 4) of different players whose paths cross in a frame (x segments overlap, y within the blob's size):
  - the blob with the higher attack level flies on and hits; the other bursts at the meeting point, a splash in mid-air in its colour;
  - with equal levels, both burst, with a splash in both team colours.
- **Final smash** (feature 1): it falls from above, so blobs never meet it, and a blob fired at a player under the rain still hits. If both players fire one, both fall; each player's shield blocks the drops on them.
- **Final blows in the same frame:** apply all of a frame's damage first, then decide: if both are at 0, the higher attack level wins; with equal levels it is a draw ("DRAW", both colours). Otherwise the one with health left wins. Needs a `result` instead of only `winner_id` (also used by the game start's result screen, `merge_log.md`).

**Check.** Unit tests: two blobs with equal levels cancel in the middle; a blob with the higher level goes through and hits; two finishing hits in one frame give a draw with equal levels, and a win for the higher level otherwise. On the webcam: both punch at once, with and without a power item.

### 6. Dodge by stepping forward

**What.** A player dodges a normal attack by stepping forward, towards the camera, off the line both players stand on. The paint blob (feature 4) then flies past behind them. The game sees the step as a change in the player's depth (`Player.pos`). For the line to mean the same for both players, the calibration also asks them to stand level with each other, side by side at the same distance from the camera.

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
- **Look:** while dodging, the player's outline flashes white. A blob that passes them is drawn behind their body (the body mask of feature 4), with a short "MISS!" at `above_head`.

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
