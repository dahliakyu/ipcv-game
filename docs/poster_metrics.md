# Poster: Performance and Limitations (draft)

Text for the five "Metrics" boxes of the poster, one per task, from the
evaluation in `docs/project_report.md` (clips of two players replayed through
the final game; identities and actions hand-labelled).

## Metrics 1 · Face tracking & AR

- Both faces found in **100 %** of frames standing still at 1–2 m; **82–100 %** moving at 3 m, with optical flow bridging the misses.
- One Euro filter: **~60 %** less jitter on a still face; the hat stays within **1–2 px** of the face. **10–14 ms** per frame.
- Limit: small faces (3 m) and dark rooms (detection drops to ~53 %).

## Metrics 2 · Pose & motion

- Upper-body keypoints found in **98–100 %** of frames; dropouts up to **0.33 s** are bridged.
- One Euro smoothing cuts still jitter by **63 %** (the earlier EMA: 34 %) and trails fast moves by only **10–18 ms**; movement → signal in **21–29 ms**.
- Pose model: **10 ms** per frame on a GPU, **49–77 ms** on a CPU.
- Limit: keypoints drop out when players step partly out of the frame; keypoint accuracy not measured.

## Metrics 3 · Multiplayer identity

- **26 / 26** crossings and **14 / 14** re-entries kept the right ID: **0 switches**, 100 % of 681 labelled player-frames correct.
- Returning players recognised again in **0.3–0.7 s**.
- Limits: a newcomer stepping into a leaving player's place inherits the ID; a bystander can take a free player slot; depth reads **1.1–1.8 m** too far (uncalibrated camera).

## Metrics 4 · Interaction & game control

90 labelled actions by 3 players:

| | Precision | Recall |
| --- | --- | --- |
| Punch | 89 % | 88 % |
| Shield | 54 % | 81 % |
| Beam | 100 % | 78 % |

- Start of move → effect on screen: **0.2–0.4 s**; still **8 / 8** actions at the demo laptop's **~5 fps**. T-poses never fire a punch.
- Both players acting at once: **7 / 10** recognised.
- Limit: an arm swinging up to "hands up" passes the punch pose and fires a punch first, which blocks the beam; a fix tuned on 2 players failed on a 3rd.

## Metrics 5 · Integration & real time

| Machine | FPS |
| --- | --- |
| Desktop, GPU (RTX 3070 Ti) | **24–33** |
| Desktop, CPU only (Ryzen 7 5700G) | **11** |
| Demo laptop, CPU (Ryzen 7 4800H), live | **7.6** |

- On a CPU the pose model takes over half of every frame.
- **0** failed frames in ~13 min of play; start-up **4.3 s**.
- Limit: below real time without a GPU.

## Edits to the existing poster text

Changes needed in the other boxes of `Poster.pdf`, so they match the final game and the report.

| Box | Now | Change to | Why |
| --- | --- | --- | --- |
| Task 1 | "an OneEuro filter was used" | "a One Euro filter was used" | Spelling of the filter's name |
| Task 2 | "all keypoints were smoothed with an EMA filter" | "all keypoints were smoothed with a One Euro filter, which smooths still poses strongly and fast moves lightly" | The final game uses the One Euro filter (report, Task 2): it replaced the EMA after the evaluation |
| Task 4 | "Each gesture must hold briefly before it counts" | "Shield and beam must hold briefly before they count; a punch counts at once" | Only arms crossed and hands up have the confirm time (0.03 s) |
| Task 4 | "buffies are grabbed with a wrist keypoint" | "the orb is grabbed with a hand" | "buffies" is a typo; the grabbed item is the orb, while the drinks also give buffs |
| Game rules, rule 3 | "After grabbing a buff with your hand" | "After grabbing the orb with your hand" | Same: "buff" could mean a drink |
| Task 5 | Only the role ("make all the tasks come together …") | The description below | The other task boxes describe their method |

### Task 5 box: description

> **Responsible:** Ellen Zheng
> Pipeline integration: pose → identity → motion → face → gestures → game → render. The renderer gives the camera image a pixel look, keeps paint on each player's body, and splashes missed paint on the wall behind them, using a person mask (MediaPipe Selfie Segmenter). Every frame is timed and logged per stage.
