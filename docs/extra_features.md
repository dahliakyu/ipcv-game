Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Extra features: to build after the merge

This is the backlog of game features agreed during the Colour Clash merge, plus the technical follow-ups the merge left open. None of it is built yet. Each feature lists what it should do, which task owns which part, the inputs it needs, a starting design with thresholds, and how to check it. The thresholds are starting points: tune them on the clips and put them in `configs/default.yaml` with a reason, like every other threshold.

## Before any of this: finish the merge

The features build on the Colour Clash game, which still lives in `garb/rest/` and is not yet in the shared codebase. In this order:

1. **Gestures (Task 4, with Task 2 signals).** Move `garb/rest/Project/bodypose.py`'s punch, hands-up and arms-crossed detection into `interaction/gestures.py` as `GestureEvent`s (`punch`, `hands_up`, `arms_crossed`).
2. **Game logic (Task 4).** Move `garb/rest/game.py` (health, bolts, beam, shield, paint splats, winner) into `interaction/game.py`, driven by those events.
3. **Rendering (Task 5).** Move `garb/rest/renderer.py` into `scene/renderer.py`, drawing at face anchors instead of its own `head_x`/`head_y` from the pose.
4. **Identity (Task 3).** See the open issues in [merge_log.md](merge_log.md#open-issues-found-during-the-merge).

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

- **One colour table.** `scene/draw.py` (`PLAYER_COLORS`) and the prototype's `game.py` (`PLAYER_COLORS`: P1 red, P2 blue) disagree today. Keep one table, in `scene/draw.py`, and use it everywhere.
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

**What.** Name, HP bar and status icons above each player's head, replacing the prototype's `draw_status()` and `renderer._draw_player_health()`.

**Owners.** Task 5: the renderer, reading game state. Task 1 provides `anchors["above_head"]`.

**Design.**

- **Name and HP bar.** "P1" and a bar at `above_head`, width ~1.2 × `scale` (with a minimum size so far players stay readable), in the team colour, faded by `alpha`.
- **Dizzy stars.** Below 25% HP: 3 stars orbit `above_head` on an ellipse in the head's plane (flattened by cos(pitch), turned by `roll`).
- **Crown.** The player with more HP gets a small crown just above the bar.

**Check.** The bar stays above the hat (also with the chin up), is readable at the back of the room, and fades out with the face.

### 4. Super charge, reset by blinking

**What.** A super move that has to be charged: the player holds the charge pose, and a meter fills. **A blink resets the charge to zero**, so charging becomes a staring contest. The other player can try to make them blink.

**Owners.**

- Task 4: the charge state machine and the `blink` event in `interaction/gestures.py`.
- Task 1: the blink signal (already there: `blendshapes`) and the charge aura in `face/effects.py`.

**Design.**

- **Charge.** Holding `hands_up` (the beam pose) for longer than the normal beam charges a super beam: full after ~3 s. Releasing it at full charge fires it (more damage and width than `BEAM_DAMAGE`).
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
  | Chili | the next bolt or beam does double damage, or the attack cooldown is reset |
  | Rotten item | splashes the eater with paint (−10 HP): a risk when grabbing blindly |

- **Feedback.** Eating triggers the cheek bulge (feature 7); grabbing a short sparkle at the wrist.

**Check.** Items can be eaten and grabbed on the webcam; an item cannot be taken twice; mouth movements while talking do not eat items that are not near the mouth.

### 7. Cheek bulge

**What.** After eating, the player's cheeks puff up for a second, chipmunk style.

**Owners.** Task 1: `face/effects.py`, triggered by the `bite` event.

**Design.**

- **Warp.** A real image warp, not a sticker: a radial "bulge" (magnifying) warp around each cheek anchor with `cv2.remap`. Inside radius r (~0.3 face widths), the source radius is ρ' = ρ · (ρ / r)^k with k rising from 0 to ~0.5 and back over 1 s. A good image-processing technique to explain in the report.
- **Cost.** Warp only the two small cheek boxes, not the whole frame.
- **Optional.** The player can also puff their own cheeks: MediaPipe's `cheekPuff` > 0.5 shows the same bulge.

**Check.** The bulge stays on the cheeks while the head moves and turns, and ends cleanly; ~1 ms extra per face.

## Technical follow-ups from the Task 1 merge

| Item | Why | Where |
| --- | --- | --- |
| Webcam check of the face stage | Steps 8–10 were tested on clips and photos; check chin up and down, turning, mouth and blink bars, and `hat_depth_mm` live | `face:` config |
| Jitter on a still clip | The test clips are all moving; shrinking crops (step 8) and the pose (steps 9–10) may add shake that only a still clip shows | `tools/eval_face.py` on the shared `normal` clip |
| MediaPipe GPU delegate | MediaPipe costs ~8 ms per frame on the CPU; the GPU delegate may cut that | `face/landmarker.py` |
| One player colour table | `scene/draw.py` and the prototype's `game.py` define different colours | Task 5 |
| Remove `garb/` | The raw prototype is ignored by git only while the merge runs; delete it and its `.gitignore` entry at the end | `.gitignore` |
| `core/types.py` heads-up | Two fields were added during the merge: `Face.alpha` and `Face.blendshapes`; mention them in the merge PR | merge PR description |

## Considered, not planned

Mentioned during the merge but not chosen. Kept here in case there is time left:

- Paint splats drawn on the face (in face coordinates) as damage, instead of on the body box.
- A "paint spit" attack: hold the mouth wide open to fire a paint blob from `anchors["mouth"]`.
- Real props as consumables: YOLO detection of an apple, banana, bottle or cup (COCO classes) instead of virtual items. Needs a second detection model per frame.
