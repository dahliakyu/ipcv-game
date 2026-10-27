Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. None of the features is built yet. Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

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

### 1. Team colour

**What.** Each player wears their team colour on the face, so it is always clear who is who: the hat's band in the team colour, and two short war-paint stripes on each cheek.

**Owners.** Task 1: cheek anchors and the drawing in `face/effects.py`. Task 5: one colour table.

**Design.**

- **One colour table.** Done in the game merge: `PLAYER_COLORS` in `scene/draw.py` (P1 red, P2 blue, the colours from `game.py`) is the only table; use `player_color(pid)`.
- **Hat band.** Recolour the band pixels of `hat.png` once at load, one copy per team: select the red band by hue, shift its hue to the team colour, keep the shading.
- **Cheek anchors.** Add `left_cheek` and `right_cheek` as 3D points on the head model, about (±45, 30, −5) mm, projected like the hat. They follow tilt and turn for free.
- **Stripes.** Draw at the cheek anchors, rotated by `roll`, squashed by cos(`yaw`) on the cheek turned away, faded by `alpha`.

**Check.** Both players on the crossing clip keep their own colour; the stripes stay on the cheeks when the head turns.

### 2. Hit reaction

**What.** When a player is hit, their hat flies off and the face flashes. The hat comes back after a moment.

**Owners.** Task 4: a `hit` event with the attacker's direction. Task 1: the animation in `face/effects.py`.

**Design.**

- **Hit event.** `Game` emits a `hit` event (pid, direction, damage) when `_apply_damage()` runs. The renderer passes events and `frame.t` to `FaceEffects.draw`, which then needs per-player animation state (it is stateless today).
- **Hat knocked off.** The hat detaches from its anchor with a velocity away from the attacker plus upwards (~1.5 face widths/s), spin (~360°/s) and gravity. It fades out over 0.6 s, then pops back on the head with a short scale-up (0.2 s). Use time, not frames.
- **Flash.** For 150 ms, blend white then the attacker's team colour over an ellipse at `center` (size from `scale`, rotated by `roll`).
- **Beam hits.** A beam hit (stronger) can also add a short shake of the hat anchor.

**Check.** One knock-off per hit; the hat always comes back. Two hits in a row don't stack two hats.

### 3. Status above the head

**What.** A name tag and status icons above each player's head, so the state is visible on the player. The health bars stay pinned to the bottom corners (Task 5 modifications: a bar that moved with the player was too busy).

**Owners.** Task 5: the renderer, reading game state. Task 1 provides `anchors["above_head"]`.

**Design.**

- **Name tag.** "P1" at `above_head`, small, in the team colour, faded by `alpha`; size from `scale` with a minimum so far players stay readable.
- **Dizzy stars.** Below 25% HP: 3 stars orbit `above_head` on an ellipse in the head's plane (flattened by cos(pitch), turned by `roll`).
- **Crown.** The player with more HP gets a small crown above the tag.

**Check.** The icons stay above the hat (also with the chin up) and fade out with the face.

### 4. Super charge, reset by blinking

**What.** A super move that has to be charged: the player holds the charge pose, and a meter fills. **A blink resets the charge to zero**, so charging becomes a staring contest. The other player can try to make them blink.

**Owners.**

- Task 4: the charge state machine and the `blink` event in `interaction/gestures.py`.
- Task 1: the blink signal (already there: `blendshapes`) and the charge aura in `face/effects.py`.

**Design.**

- **Charge.** Holding `hands_up` (the beam pose) for longer than the normal beam charges a super beam: full after ~3 s. Releasing it at full charge fires it (more damage and width than the normal beam, `game.beam_damage`).
- **Blink detection.** The raw score is not enough: on the test clips, `blink` = mean of `eyeBlinkLeft`/`Right` went over 0.5 in up to 26% of frames just because people looked down. Use:
  - a per-player baseline: the running median of the score over the last ~3 s, which follows lighting, glasses and eye shape;
  - a blink that starts at score > baseline + 0.35 (and > 0.5), and ends at score < baseline + 0.15 (hysteresis);
  - a duration of 60–500 ms (longer = eyes closed, not a blink);
  - no blinks counted while pitch < −20° (looking down, see `log_fields` `pitch_deg`), or while the face is `hold` or `alpha` < 1.
- **Measurement rate.** MediaPipe measures each player at ~15 Hz (`face.landmarker_per_frame: 1`). A 100 ms blink can fall between two measurements. If blinks are missed, set it to 2 (every player every frame, +~8 ms).
- **Effect.** A glowing aura around the face that grows with the charge; on a blink it shatters into particles, with a short "blink!" text at `above_head`.

**Check.** On the webcam: deliberate blinks always reset; looking down at the floor for 2 s does not. Log blinks and count false triggers on the clips.

### 5. Taunt

**What.** A big smile held for a moment shows a taunt emote above the player's head. Cosmetic; it could later give a small bonus, for example a slightly faster charge.

**Owners.** Task 4: the `taunt` event. Task 1 or 5: the emote.

**Design.**

- **Trigger.** Smile = mean of `mouthSmileLeft`/`Right` > 0.6 for 0.5 s; then a 5 s cooldown. Not while `jawOpen` > 0.4, so a laugh while eating does not count.
- **Emote.** A speech bubble or emoji sprite at `above_head` that pops in, wobbles for 1.5 s, and fades.

**Check.** A held smile triggers once per cooldown; talking and eating do not trigger it.

### 6. Consumables, Smash Bros style

**What.** Items drop from the sky during the round. A player can **eat** a falling item by catching it with their mouth (chomp), or, once it has landed, **grab** it from the floor with a hand. Each item has an effect tied to the existing mechanics.

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
  | Chili | the next bolt or beam does double damage |
  | Rotten item | splashes the eater with paint (−10 HP): a risk when grabbing blindly |

- **Feedback.** Eating triggers the cheek bulge (feature 7); grabbing a short sparkle at the wrist.
- **Sprites (candidate):** `assets/Fancy Drinks/` (64 × 64 and 128 × 128 drinks and glasses, by Savannah; its Read Me states no licence). Not in git yet; add it with a README credit before using it.

**Check.** Items can be eaten and grabbed on the webcam; an item cannot be taken twice; mouth movements while talking do not eat items that are not near the mouth.

### 7. Cheek bulge

**What.** After eating, the player's cheeks puff up for a second, chipmunk style.

**Owners.** Task 1: `face/effects.py`, triggered by the `bite` event.

**Design.**

- **Warp.** A real image warp, not a sticker: a radial "bulge" (magnifying) warp around each cheek anchor with `cv2.remap`. Inside radius r (~0.3 face widths), the source radius is ρ' = ρ · (ρ / r)^k with k rising from 0 to ~0.5 and back over 1 s. A good image-processing technique to explain in the report.
- **Cost.** Warp only the two small cheek boxes, not the whole frame.
- **Optional.** The player can also puff their own cheeks: MediaPipe's `cheekPuff` > 0.5 shows the same bulge.

**Check.** The bulge stays on the cheeks while the head moves and turns, and ends cleanly; ~1 ms extra per face.

### 8. Learned punch classifier

**Status.** Planned for after the merge; the rule with sideways reach (merge step 2b) works well enough for now. Decisions already made (Task 5 owner, 2026-10-25):

- **Goal:** one real sideways punch fires one bolt.
- **Labels:** the start (arm starts moving out) and end (arm fully extended) of each punch, marked frame by frame in a labelling tool.
- **Data:** the Task 5 owner records themselves in sessions with different clothing and rooms. Clip names: `punch_<session>_<n>`, `neg_<session>_<n>` (no labels needed), `game_<n>` (test only).
- **Split:** leave one session out; the report states that all data is from one person.
- **Model:** logistic regression first.
- **Steps:** `interaction/punch_features.py` → `tools/label_punches.py` → `tools/train_punch.py` → `gestures.punch_backend: rule | classifier` + `tools/eval_gestures.py`.

**What.** A small trained classifier decides "punch or not" per player and frame, instead of the hand-written rule.

**Why.** The punch rule from `bodypose.py` (`detect_punch()` in `interaction/gestures.py`) checks poses one frame at a time. It reads a straight hanging arm next to a bent arm as a punch: 33% of player-frames on the jump clip, none of them real punches (`merge_log.md`, Task 4, step 2). The rule also ignores speed. A punch is fast and a hanging arm is still, and a classifier can learn that from examples instead of us adding one rule after another.

**Owners.** Task 4: the classifier, labelling and evaluation. Task 2: the inputs from `Motion` (smoothed keypoints, `valid`, `vel`).

**Design.**

- **Features per player and frame**, all in shoulder widths so distance to the camera doesn't matter:
  - each arm's elbow and wrist position relative to its shoulder;
  - both elbow angles;
  - wrist speed (from `Motion.vel`, in sw/s), plus the same features averaged over the last ~0.2 s, so the movement counts and not just the pose.
- **Labels:** none / punch, per player-frame. Record clips with `tools/record_clip.py` and label them with a small script of our own. Include hard negatives: hanging arms, running, waving, hands up, arms crossed, stretching. Use the two current test clips as extra negatives.
- **Model:** start small and explainable with `scikit-learn` (logistic regression, k-NN or a random forest), as in `docs/candidate-modules-and-libraries.md`. Pin `scikit-learn` and `joblib` in `requirements.txt`; keep the trained model file in `models/` (gitignored) with a training command in the README.
- **Keep the rule as fallback:** `gestures.punch_backend: rule | classifier`. The event logic stays the same (hold time, rising-edge `punch` event), and only the per-frame "is this a punch" changes. A probability threshold goes in config.

**Check.** Split train and test by recording, not by frame, so frames from one clip never land on both sides. Report precision and recall per player-frame and missed or extra punch events, for the rule and the classifier on the same test clips. Target: no punches on the jump and fitness clips, real sideways punches still detected, under 1 ms per frame.

### 9. Charge bar for the major attack

**What.** The major attack (hands up: the magic ball and beam) can only be fired once the player's charge bar has reached a threshold. Firing it empties the bar, so the beam becomes a reward for playing well instead of something to repeat.

**Owners.** Task 4: the charge state and the rule in `interaction/game.py`. Task 5: the bar in the renderer, next to the health bar.

**Design.**

- **Charge per player,** 0–100, a new `GamePlayer.charge` field. It fills:
  - slowly over time, so a player who is being hit can still get there (~+5 per s: full after ~20 s);
  - by landing bolts (~+15 per hit);
  - a little by blocking an attack with the shield (~+10).

  All amounts go in the `game:` config.
- **Threshold.** At `charge` ≥ the threshold (100 to start with), `hands_up` fires the beam and the bar drops to 0. Below the threshold, `hands_up` fires nothing. The player gets "not charged" feedback (the bar flashes), and the reset rule still applies, so they have to lower their arms before trying again.
- **Display.** A thin bar just above the player's health bar in their bottom corner, in the team colour. When full it glows, and the magic ball can already show between the hands as a hint.
- **Link with feature 4.** The super charge (hold the pose, a blink resets it) can share this bar: a full bar plus holding the pose charges the super beam.

**Check.** On the webcam: `hands_up` with a part-filled bar fires nothing; after enough hits it fires once and the bar empties. Log `charge` per player (`log_fields`) to tune the fill rates so a beam comes every ~20–30 s of play.

### 10. Weapon pick-ups (e.g. a laser gun)

**What.** Weapons drop into the arena like the consumables of feature 6. A player who grabs one gets a stronger attack for a limited time or number of shots. First weapon: a laser gun that fires a fast laser on each punch instead of the normal fire bolt.

**Owners.** Task 4: weapon items, the equipped state and the weapon attacks in `interaction/game.py`. Task 5: weapon sprites and the laser drawing.

**Design.**

- **Spawning and grabbing:** as for the items in feature 6 (falling from the top, lying on the floor, grabbed with a wrist), but rarer (every ~20 s) and at most one weapon in the arena at a time.
- **Equipped state,** a new `GamePlayer.weapon`: the weapon's type and its ammo (e.g. 5 shots) or time left (e.g. 10 s). The weapon is drawn at the punching hand (`punching_hand()`). Out of ammo, it disappears.
- **Laser gun:**
  - a punch fires a laser instead of a fire bolt: faster (~3 frame widths/s or instant) and more damage (~15);
  - the reset rule and "one action at a time" still apply.
  - Sprites: "Laser Shoot Animation #1" by dinhaz (itch.io; no licence stated, credit it in the README). `LaserGun_Shoot.png` from that page for the shot. `LaserGun_Burn..png` (in git history, commit `62febc0`) for a held burn. The laser drawing tried during the merge (muzzle on the hand, the burn tiled to its length, hue-turned per team) can be rebuilt from that sheet.
- **More weapons later:** a paint bucket (a short-range splash that paints a large area), a shield bubble (blocks the next 3 hits).

**Check.** Only one player can take a weapon; the ammo counts down and the weapon disappears at 0; the normal bolt comes back afterwards.

### 11. Aiming: a pointing laser and attacks towards the opponent

**What.** Attacks no longer fly only sideways towards the middle of the frame: they go where the player aims, towards the opponent. A thin pointing laser from the hand (or from the magic ball) shows the aim before and while firing.

**Owners.**
- Task 4: aim direction and 2D movement and hits in `interaction/game.py`.
- Task 2: arm direction from the keypoints.
- Task 5: the pointing laser and rotated sprites.

**Design.**

- **Aim direction.** For a punch: the punching arm's direction, shoulder → wrist (`punching_hand()` already finds the arm). For the beam: from the ball (between the hands) towards the opponent's torso. Aim assist: if the arm points within ~20° of the opponent's torso (shoulder midpoint to hip level, `body_position()`), snap to it, so a roughly correct punch hits.
- **Towards the opponent, not the middle.** Done for sideways attacks (Task 4 modifications): `Game.facing()` fires towards the nearest other player in view. What is left is the 2D direction below.
- **2D game logic.**
  - `Projectile` gets a velocity vector instead of `direction`.
  - Hits test the path since the last frame (a segment) against the target's body box in 2D.
  - The beam becomes a segment from the ball to its tip. It hits a box within `beam_hit_radius_h` of that segment (point-to-segment distance).
  - Bolt ends and splats use the 2D impact point.
- **Pointing laser.**
  - **Before a punch:** while an arm is extended (straight but not yet a punch), a thin dashed line in the team colour runs from the wrist along the aim direction until it meets a body box or the frame edge.
  - **Beam charge:** the same line from the ball towards the target.
  - It fades with the gesture's state, and it is drawn only for the aiming player, to keep the frame readable.
- **Sprites.** Rotate the fire blast and the beam band along the direction (`cv2.warpAffine`, like `paste()` in `face/effects.py`); the magic ball needs no rotation.

**Check.** Bolts and beams hit an opponent who stands higher or lower (e.g. crouching) or in the same half of the frame; the pointing laser matches where the attack actually flies; aim assist does not snap to a player behind the shooter.

### 12. Game start: instructions and pose calibration

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
- **Waiting.** "Player 1: stand on the left / Player 2: on the right", with each player's box ticked in the team colour once identity has them (`status == "active"`), both faces visible.
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
     - `waiting` → `calibration`: P1 and P2 active with a valid face for ~1 s.
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
2. **Waiting, countdown and result screens** (Task 5, `scene/renderer.py`).
   - Waiting: the two start places, each ticked in the team colour.
   - Countdown: 3-2-1 in the middle.
   - Result: from `_draw_game_over()`, with final health and "hands up to play again".
   - The top bar says what the phase expects.
3. **Instruction panels** (Task 5, `scene/renderer.py`). Per player, on their half: a stick-figure picture of the current move (drawn with OpenCV lines from a few fixed joint positions, no image assets), its effect, ticks for the moves done, and the reminder flash.
4. **Pose calibration** (Task 2 measurements, Task 4 rules).
   - The neutral pose as the first calibration move: shoulders and hips seen, wrists below the shoulders, face towards the camera, for ~2 s.
   - Per player, the median shoulder width and arm length (shoulder–elbow–wrist), from a new `pose/measure.py`.
   - Kept on a new round; a warning when the arm is too short for `gestures.punch_min_reach`; logged in `log_fields`.
5. **Identity at calibration** (Task 3, `identity/tracker.py`, wired in `app/pipeline.py`).
   - P1 becomes the left player when calibration starts (the links are swapped if needed; follow-up "P1 / P2 order").
   - The OSNet vector taken during the neutral pose replaces the first-sight vector.
6. **Depth reference** (Task 3, `Player.pos`): depth relative to the calibration distance, from the calibrated shoulder width (pinhole model: depth ∝ 1 / shoulder width).
7. **Docs and webcam check.** Update the code map and the README keys; the Task 5 owner runs the full intro with two players on the webcam.

### 13. Paint stains where the hit landed

**What.** Since the renderer merge (step 4), the paint is drawn on the torso, so it no longer slides off the body when an arm goes out. But the game still keeps each splat as a fraction of the body box, inside a fixed band (`splat_band`, 0.38–0.62), and the renderer stretches that band over the torso. So a hit on the shoulder or the stomach lands near the middle of the shirt, and the splat's place on the torso changes when the box changes shape between the hit and later frames.

**Owners.** Task 4: splat coordinates in `interaction/game.py`. Task 5: drawing them in `scene/renderer.py` (`paint_area()`).

**Design.**

- **Torso coordinates at the hit.** When a hit lands, turn the impact point into (`u`, `v`) on the torso from the smoothed keypoints (`Game.torso()`): `u` from the left to the right shoulder, `v` from the shoulder line (0) to the hips (1). Store those in `PaintSplat` instead of box fractions; the renderer maps them back with `paint_area()`.
- **Torso quad instead of a rectangle (optional):** use the four corners (both shoulders, both hips) and a bilinear map, so the paint also leans and turns with the body.
- **Band.** `splat_band` becomes a band in torso coordinates (e.g. `u` 0.1–0.9, `v` 0.05–0.9), so hits outside the torso (an arm) still put the paint on the shirt.
- **Turned body (optional):** hide splats on a side turned away from the camera.

**Check.** A bolt that hits at shoulder height leaves paint near the shoulder; the paint stays there while the player moves their arms and leans.

### 14. Both players attacking at the same time

**What.** Make simultaneous attacks fair and visible. Today (`Game.advance()` in `interaction/game.py`, as in `game.py`):
- bolts and beams fly through each other, and both hits land;
- when both hits would finish the round in the same frame, the attack fired first wins: `_apply_damage()` ignores everything once `winner_id` is set, so the order of the `projectiles` list decides.

**Owners.** Task 4: collisions and the tie rule in `interaction/game.py`. Task 5: the clash effects in the renderer.

**Design.**

- **Bolt meets bolt:** two bolts of different players whose paths cross in a frame (x segments overlap, y within the bolt's height) cancel each other; draw both break-ups (`BoltEnd`) at the meeting point, plus a burst in both team colours.
- **Bolt meets beam:** the beam absorbs the bolt (the beam is the stronger attack); the bolt breaks up at the beam's tip.
- **Beam meets beam:** the tips stop where they meet and push against each other for the rest of their duration (a "beam struggle"), with a flash at the meeting point; neither player is hit. Optional: the player who keeps their hands up longer, or whose charge is higher (feature 9), pushes the meeting point towards the other.
- **Final blows in the same frame:** apply all of a frame's damage first, then decide: both at 0 → a draw ("DRAW", both colours), else the one with health left wins. Needs a `result` instead of only `winner_id` (also used by feature 12's result screen).

**Check.** Unit tests: two bolts fired at each other cancel in the middle; a bolt into a beam disappears; two beams stop at their meeting point; two finishing hits in one frame give a draw. On the webcam: both punch at once and see the clash.

### 15. Major attack orb in four parts (Smash Bros style)

**What.** The major attack (hands up) is unlocked by collecting its orb in four parts, like the assembled weapons in Smash Bros (the Dragoon or Daybreak parts). Parts appear in the arena one at a time; a player who holds all four can fire the major attack once.

**Owners.** Task 4: parts, collecting, dropping and the unlock rule in `interaction/game.py`. Task 5: part sprites (the magic ball cut into quarters) and the HUD.

**Design.**

- **Spawning:** one part at a time at a random free spot in the upper half of the frame (reachable with a raised hand), every ~5–8 s while fewer than four parts are in play; it slowly bobs.
- **Collecting:** a wrist (smoothed keypoint, `state.motion`) within ~0.5 shoulder widths of the part, held for ~0.2 s, picks it up, as for the items in feature 6.
- **Holding:** each player's parts are shown as a 4-slot orb icon by their health bar, in their team colour; the quarters fill in as they are collected.
- **Dropping:** a beam hit, or 3 bolt hits in a row, knocks one part out of the player; it flies off and lands as a part to collect again, so the other player can steal it.
- **Unlock:** with all four, `hands_up` fires the major attack and the slots empty. Without them, `hands_up` does nothing (with "not ready" feedback), and the one-action-at-a-time reset still applies.
- **Relation to feature 9:** this replaces the charge bar as the way to unlock the major attack; pick one of the two, or use the bar for a smaller attack.

**Check.** Parts only disappear when collected; a player cannot hold more than four; a dropped part can be picked up by either player; the major attack fires only with four parts.

### 16. Major attack as paint splashing down from above

**What.** Instead of the horizontal magic beam, the major attack throws paint up and lets it rain down on the other player: the magic ball rises above the caster, bursts, and paint splashes down over the opponent.

**Owners.** Task 4: the attack's timing, area and hits in `interaction/game.py` (replaces `PaintBeam`). Task 5: ball, burst, falling drops and floor splashes in the renderer.

**Design.**

- **Phases:** charge (the ball grows between the hands, as now, ~0.35 s) → rise (the ball flies up out of the top of the frame, ~0.3 s) → rain (paint drops fall over the opponent for ~1 s) → splash marks fade on the floor.
- **Where it falls:** a band centred on the opponent's body box at the moment of the burst, about 1.5 box widths wide, so the opponent can dodge by stepping out of it after the warning (a shadow or target mark on the floor during the rise).
- **Hits:** each drop tests its path since the last frame against the target's box (as the bolt does); every hit adds a splat on the box and a little damage (e.g. 2 per drop, at most the old beam's 12 per attack). The shield blocks drops while it is up.
- **Look:** drops in the caster's team colour, falling with gravity, slightly random sizes; splash sprites where they hit the floor (the bottom of the body box, or the frame bottom) or the player.
- **Replaces** `beam_ball` / `beam_band` drawing for the major attack; the magic ball frames can stay for the charge and rise. Feature 14's beam-meets-beam rule then becomes "drops falling on a raised shield".

**Check.** The rain falls where the opponent stood at the burst; stepping out of the band during the warning avoids most drops; damage per attack never exceeds the cap; the shield blocks it.

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
| P1 / P2 order | The first player confirmed is P1 (as in `bodypose.py`); with feature 12, the calibration can assign P1 to the left player | `identity/tracker.py` |

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
