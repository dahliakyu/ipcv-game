"""Gesture rules from the Colour Clash bodypose.py, on synthetic skeletons.

The base pose: shoulders 80 px apart at y = 200 (sw = 80), hips at y = 360,
arms hanging down with the wrists below the hips. Image y grows downwards.
"""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.geometry import angle_at
from ipcvgame.core.types import FrameState, Motion
from ipcvgame.interaction.gestures import (
    GestureRecognizer,
    arm_bent,
    arm_extended,
    detect_crossed,
    detect_hands_up,
    detect_punch,
    lower_body_y,
)

SW = 80.0


@pytest.fixture
def cfg():
    return load_config()["gestures"]


def standing() -> tuple[np.ndarray, np.ndarray]:
    k = np.zeros((sk.K, 2))
    k[sk.NOSE] = (320, 140)
    k[sk.L_SHOULDER], k[sk.R_SHOULDER] = (360, 200), (280, 200)
    k[sk.L_ELBOW], k[sk.R_ELBOW] = (370, 300), (270, 300)
    k[sk.L_WRIST], k[sk.R_WRIST] = (375, 400), (265, 400)  # below the hips
    k[sk.L_HIP], k[sk.R_HIP] = (350, 360), (290, 360)
    return k, np.ones(sk.K, bool)


def test_angle_at():
    assert angle_at(np.array([0.0, 1]), np.zeros(2), np.array([1.0, 0])) == pytest.approx(90)
    assert angle_at(np.array([-1.0, 0]), np.zeros(2), np.array([1.0, 0])) == pytest.approx(180)
    assert angle_at(np.zeros(2), np.zeros(2), np.array([1.0, 0])) == 0.0


def test_lower_body_y_falls_back_below_shoulders(cfg):
    k, v = standing()
    assert lower_body_y(k, v, 200.0, SW, cfg) == 360.0
    v[sk.L_HIP] = False
    assert lower_body_y(k, v, 200.0, SW, cfg) == 200.0 + cfg["hip_below_shoulders"] * SW


def test_standing_is_no_gesture(cfg):
    k, v = standing()
    assert not detect_hands_up(k, v, SW, cfg)
    assert not detect_crossed(k, v, SW, cfg)
    assert detect_punch(k, v, SW, cfg) is None


def test_hanging_arm_with_guard_is_no_punch(cfg):
    k, v = standing()
    k[sk.R_ELBOW], k[sk.R_WRIST] = (260, 280), (300, 230)  # other arm bent
    assert detect_punch(k, v, SW, cfg) is None  # hanging wrist is below the hips


def test_hands_up(cfg):
    k, v = standing()
    k[sk.L_WRIST], k[sk.R_WRIST] = (370, 100), (270, 100)
    assert detect_hands_up(k, v, SW, cfg)
    k[sk.R_WRIST] = (270, 150)  # one hand below the nose
    assert not detect_hands_up(k, v, SW, cfg)


def test_hands_up_without_nose_uses_shoulders(cfg):
    k, v = standing()
    v[sk.NOSE] = False
    # Head estimate: 0.3 sw above the shoulders = y 176.
    k[sk.L_WRIST], k[sk.R_WRIST] = (370, 170), (270, 170)
    assert detect_hands_up(k, v, SW, cfg)
    k[sk.L_WRIST, 1] = 180
    assert not detect_hands_up(k, v, SW, cfg)


def test_hands_up_needs_both_wrists(cfg):
    k, v = standing()
    k[sk.L_WRIST], k[sk.R_WRIST] = (370, 100), (270, 100)
    v[sk.R_WRIST] = False
    assert not detect_hands_up(k, v, SW, cfg)


def crossed() -> tuple[np.ndarray, np.ndarray]:
    k, v = standing()
    k[sk.L_ELBOW], k[sk.R_ELBOW] = (380, 270), (260, 270)
    # Each wrist on the other side of the body from its own shoulder.
    k[sk.L_WRIST], k[sk.R_WRIST] = (300, 260), (340, 260)
    return k, v


def test_crossed(cfg):
    k, v = crossed()
    assert detect_crossed(k, v, SW, cfg)


def test_crossed_is_mirror_independent(cfg):
    k, v = crossed()
    k[:, 0] = 640 - k[:, 0]
    assert detect_crossed(k, v, SW, cfg)


def test_crossed_rejects_far_apart_or_out_of_chest_zone(cfg):
    k, v = crossed()
    k[sk.L_WRIST], k[sk.R_WRIST] = (200, 260), (440, 260)  # 3 sw apart
    assert not detect_crossed(k, v, SW, cfg)
    k, v = crossed()
    k[sk.L_WRIST, 1] = k[sk.R_WRIST, 1] = 100  # crossed overhead
    assert not detect_crossed(k, v, SW, cfg)


def punch_left() -> tuple[np.ndarray, np.ndarray]:
    k, v = standing()
    k[sk.L_ELBOW], k[sk.L_WRIST] = (420, 200), (480, 200)  # straight, 1.5 sw reach
    k[sk.R_ELBOW], k[sk.R_WRIST] = (260, 280), (300, 230)  # bent guard
    return k, v


def test_punch(cfg):
    k, v = punch_left()
    assert arm_extended(k, v, SW, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, cfg)
    assert arm_bent(k, v, sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST, cfg)
    assert detect_punch(k, v, SW, cfg) == "L"


def test_punch_right(cfg):
    k, v = punch_left()
    k[:, 0] = 640 - k[:, 0]
    k[[sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST]] = \
        k[[sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST]]
    assert detect_punch(k, v, SW, cfg) == "R"


def test_hanging_arm_above_hip_line_is_no_punch(cfg):
    # The false positive from the jump clip: a straight arm hanging with the
    # wrist just above the hip keypoint, the other arm bent. 1.9 sw from the
    # shoulder, but only 0.2 sw to the side.
    k, v = standing()
    k[sk.L_ELBOW], k[sk.L_WRIST] = (368, 270), (376, 350)
    k[sk.R_ELBOW], k[sk.R_WRIST] = (260, 280), (300, 230)
    assert not arm_extended(k, v, SW, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, cfg)
    assert detect_punch(k, v, SW, cfg) is None


def test_low_arm_is_no_punch(cfg):
    # Straight and reaching sideways, but at chest height (0.75 sw below the
    # shoulder): the webcam false positive behind this rule.
    k, v = punch_left()
    k[sk.L_ELBOW], k[sk.L_WRIST] = (420, 230), (480, 260)
    assert detect_punch(k, v, SW, cfg) is None


def test_punch_slightly_below_shoulder_still_counts(cfg):
    k, v = punch_left()
    k[sk.L_ELBOW], k[sk.L_WRIST] = (420, 205), (480, 210)  # 0.125 sw below
    assert detect_punch(k, v, SW, cfg) == "L"


def test_two_straight_arms_are_no_punch(cfg):
    k, v = punch_left()
    k[sk.R_ELBOW], k[sk.R_WRIST] = (220, 200), (160, 200)  # T-pose
    assert detect_punch(k, v, SW, cfg) is None


def test_unknown_guard_arm_counts_as_bent(cfg):
    k, v = punch_left()
    v[sk.R_WRIST] = False
    assert detect_punch(k, v, SW, cfg) == "L"


def test_punch_rejects_short_reach_and_raised_arm(cfg):
    k, v = punch_left()
    k[sk.L_ELBOW], k[sk.L_WRIST] = (390, 200), (420, 200)  # straight but 0.75 sw
    assert detect_punch(k, v, SW, cfg) is None
    k, v = punch_left()
    k[sk.L_ELBOW], k[sk.L_WRIST] = (360, 140), (360, 60)  # straight up
    assert detect_punch(k, v, SW, cfg) is None


# --- GestureRecognizer: hold times and one event per gesture ----------------

def recognizer() -> GestureRecognizer:
    return GestureRecognizer(load_config()["gestures"])


def motion(k: np.ndarray, v: np.ndarray) -> dict[int, Motion]:
    return {1: Motion(kpts=k, vel=np.zeros_like(k), valid=v, signals={})}


def hands_up() -> tuple[np.ndarray, np.ndarray]:
    k, v = standing()
    k[sk.L_WRIST], k[sk.R_WRIST] = (370, 100), (270, 100)
    return k, v


def names(events) -> list[str]:
    return [e.name for e in events]


def test_hands_up_fires_once_after_confirm_time():
    g = recognizer()
    assert g.update(motion(*hands_up()), 0.0) == []  # 1st frame: not confirmed yet
    ev = g.update(motion(*hands_up()), 1 / 30)
    assert names(ev) == ["hands_up"] and ev[0].pid == 1 and ev[0].t == 1 / 30
    assert g.update(motion(*hands_up()), 2 / 30) == []  # held: no repeat
    g.update(motion(*standing()), 3 / 30)               # lowered for 0.4 s,
    g.update(motion(*standing()), 3 / 30 + 0.4)         # longer than release_s
    g.update(motion(*hands_up()), 4 / 30 + 0.4)
    assert names(g.update(motion(*hands_up()), 5 / 30 + 0.4)) == ["hands_up"]


@pytest.mark.parametrize("pose", [hands_up, crossed])
def test_short_dropout_does_not_restart_a_gesture(pose):
    # The rule fails for 2 frames while the pose is held (keypoint noise):
    # still one gesture, so one beam / one shield.
    g = recognizer()
    events = []
    for i in range(20):
        k, v = pose() if i not in (8, 9) else standing()
        events += g.update(motion(k, v), i / 30)
    assert len(events) == 1


def test_gesture_ends_after_release_time():
    g, release = recognizer(), load_config()["gestures"]["release_s"]
    g.update(motion(*hands_up()), 0.0)
    g.update(motion(*hands_up()), 1 / 30)
    g.update(motion(*standing()), 2 / 30)
    assert g._state[1].active["hands_up"]                      # within release_s
    g.update(motion(*standing()), 1 / 30 + release + 0.01)
    assert not g._state[1].active["hands_up"]                  # released


def test_one_frame_glitch_is_no_gesture():
    g = recognizer()
    g.update(motion(*crossed()), 0.0)
    assert g.update(motion(*standing()), 1 / 30) == []
    assert g.update(motion(*crossed()), 2 / 30) == []


def test_arms_crossed_event_name():
    g = recognizer()
    g.update(motion(*crossed()), 0.0)
    assert names(g.update(motion(*crossed()), 1 / 30)) == ["arms_crossed"]


def test_one_punch_gives_one_event():
    g = recognizer()
    assert names(g.update(motion(*punch_left()), 0.0)) == ["punch"]  # no confirm time
    # Arm fails the rule for a frame, then passes again: same punch.
    assert g.update(motion(*standing()), 1 / 30) == []
    assert g.update(motion(*punch_left()), 2 / 30) == []
    # Off for longer than punch_hold_s: the next punch is a new event.
    hold = load_config()["gestures"]["punch_hold_s"]
    g.update(motion(*standing()), 2 / 30 + hold + 0.01)
    assert names(g.update(motion(*punch_left()), 2 / 30 + hold + 0.05)) == ["punch"]


def raise_arm_through_punch():
    """One arm straight out sideways at shoulder height, the other bent:
    the punch pose, as when one arm goes up before the other."""
    return punch_left()


def test_a_new_action_needs_a_normal_position_first():
    g, hold = recognizer(), load_config()["gestures"]["punch_hold_s"]
    assert names(g.update(motion(*raise_arm_through_punch()), 0.0)) == ["punch"]
    # The arms go on up: hands up turns on while the punch is still on.
    assert g.update(motion(*hands_up()), 0.05) == []
    assert g.update(motion(*hands_up()), 0.1) == []
    assert not g._state[1].ready
    # Back to normal (arms down) until every gesture is off: ready again.
    for t in (0.2, 0.2 + hold, 0.25 + hold):
        g.update(motion(*standing()), t)
    assert g._state[1].ready
    g.update(motion(*crossed()), 1.0)
    assert names(g.update(motion(*crossed()), 1.05)) == ["arms_crossed"]


def test_gestures_starting_together_send_only_the_major_attack(monkeypatch):
    import ipcvgame.interaction.gestures as gestures
    monkeypatch.setattr(gestures, "detect_all",
                        lambda k, v, cfg: {"hands_up": True, "arms_crossed": False, "punch": True})
    g = recognizer()
    events = g.update(motion(*standing()), 0.0) + g.update(motion(*standing()), 1 / 30)
    # Frame 1: only the punch is on yet (hands up needs confirm_s) -> punch.
    # Frame 2: hands up turns on, but the player has not reset -> nothing.
    assert names(events) == ["punch"]
    monkeypatch.setattr(gestures, "detect_all",
                        lambda k, v, cfg: {"hands_up": True, "arms_crossed": True, "punch": False})
    g = recognizer()
    g.update(motion(*standing()), 0.0)
    assert names(g.update(motion(*standing()), 1 / 30)) == ["hands_up"]   # same frame: priority


def test_log_fields_ready():
    g = recognizer()
    ev = g.update(motion(*punch_left()), 0.0)
    state = FrameState(frame=None, players={}, motion={}, faces={}, events=ev, timings={})
    assert g.log_fields(state)["gest_p1_ready"] == 0


def test_missing_player_starts_over():
    g = recognizer()
    g.update(motion(*hands_up()), 0.0)
    g.update(motion(*hands_up()), 1 / 30)
    g.update({}, 2 / 30)  # no keypoints this frame
    assert g.update(motion(*hands_up()), 3 / 30) == []
    assert names(g.update(motion(*hands_up()), 4 / 30)) == ["hands_up"]


def test_reset_clears_player():
    g = recognizer()
    g.update(motion(*punch_left()), 0.0)
    g.reset(1)
    assert names(g.update(motion(*punch_left()), 0.1)) == ["punch"]


def test_small_shoulders_give_no_gesture():
    g = recognizer()
    k, v = hands_up()
    k[sk.R_SHOULDER] = k[sk.L_SHOULDER] + (8, 0)  # 8 px apart
    g.update(motion(k, v), 0.0)
    assert g.update(motion(k, v), 1 / 30) == []


def test_log_fields():
    g = recognizer()
    ev = g.update(motion(*punch_left()), 0.0)
    state = FrameState(frame=None, players={}, motion={}, faces={}, events=ev, timings={})
    row = g.log_fields(state)
    assert row["gest_p1_punch"] == 1 and row["gest_p1_hands_up"] == 0
    assert row["gest_p1_events"] == "punch"


# --- Blinks (final smash, extra_features.md §1) ------------------------------

def eyes(score: float | None, pitch_deg: float = 0.0, valid: bool = True):
    """A face with both eye scores at `score` (None: no current MediaPipe result)."""
    from ipcvgame.core.types import Face
    b = {} if score is None else {"eyeBlinkLeft": score, "eyeBlinkRight": score}
    return Face(pid=1, valid=valid, landmarks=None, center=np.zeros(2), scale=60.0, roll=0.0,
                yaw=0.0, anchors={}, blendshapes=b, pitch=np.radians(pitch_deg))


def run(det, scores, dt=1 / 15, t0=0.0, pitch_deg=0.0):
    """Feed scores at dt apart; returns (t, event) for every event."""
    out = []
    for i, s in enumerate(scores):
        t = t0 + i * dt
        ev = det.update(eyes(s, pitch_deg), t)
        if ev:
            out.append((round(t, 3), ev))
    return out


def blink_detector(cfg, **over):
    from ipcvgame.interaction.gestures import BlinkDetector
    return BlinkDetector({**cfg, **over})


def test_blink_is_sent_when_the_eyes_open(cfg):
    det = blink_detector(cfg)
    events = run(det, [0.2] * 30 + [0.8, 0.8, 0.8, 0.2, 0.2])
    assert events == [(round(33 / 15, 3), "blink")]       # 3 frames closed = 200 ms


def test_baseline_follows_the_player(cfg):
    """A player whose eyes score 0.4 when open: 0.6 is not a blink (under
    baseline + blink_on), 0.8 is."""
    assert run(blink_detector(cfg), [0.4] * 30 + [0.6, 0.6, 0.4]) == []
    assert [e for _, e in run(blink_detector(cfg), [0.4] * 30 + [0.8, 0.8, 0.4])] == ["blink"]


def test_low_scores_never_blink(cfg):
    """A far face: a rise from 0.05 to 0.45 is over baseline + blink_on but
    under blink_min_score."""
    assert run(blink_detector(cfg), [0.05] * 30 + [0.45, 0.45, 0.05]) == []


def test_hysteresis_keeps_the_eyes_closed_between_the_thresholds(cfg):
    # 0.4 is under the start (0.2 + 0.35) but over the end (0.2 + 0.15): still closed.
    events = run(blink_detector(cfg), [0.2] * 30 + [0.8, 0.4, 0.4, 0.2])
    assert events == [(round(33 / 15, 3), "blink")]


def test_eyes_kept_closed_send_eyes_closed_once(cfg):
    events = run(blink_detector(cfg), [0.2] * 30 + [0.8] * 15 + [0.2] * 3)
    assert [e for _, e in events] == ["eyes_closed"]
    t_closed = events[0][0] - 30 / 15
    assert cfg["blink_max_s"] < t_closed <= cfg["blink_max_s"] + 1 / 15


def test_long_gap_while_closed_is_not_a_blink(cfg):
    det = blink_detector(cfg)
    run(det, [0.2] * 30)
    det.update(eyes(0.8), 2.0)
    assert det.update(eyes(0.2), 2.0 + cfg["blink_max_s"] + 0.1) == "eyes_closed"


def test_single_frame_glitch_is_not_a_blink(cfg):
    det = blink_detector(cfg)
    run(det, [0.2] * 30)
    det.update(eyes(0.8), 2.0)
    assert det.update(eyes(0.2), 2.0 + cfg["blink_min_s"] / 2) is None


def test_looking_down_is_not_a_blink(cfg):
    down = cfg["blink_down_deg"] - 5
    assert run(blink_detector(cfg), [0.2] * 30 + [0.8] * 20 + [0.2], pitch_deg=down) == []
    # Looking down during a closure cancels it.
    det = blink_detector(cfg)
    run(det, [0.2] * 30 + [0.8])
    assert det.update(eyes(0.8, down), 2.1) is None and det.closed_since is None
    assert det.update(eyes(0.2), 2.2) is None


def test_no_score_drops_the_closure(cfg):
    det = blink_detector(cfg)
    run(det, [0.2] * 30 + [0.8])
    assert det.update(eyes(None), 2.1) is None and det.closed_since is None
    assert det.update(eyes(0.2), 2.2) is None


def test_no_blink_before_the_baseline(cfg):
    assert run(blink_detector(cfg), [0.2] * (cfg["blink_min_samples"] - 1) + [0.9, 0.2]) == []


def test_recognizer_sends_blinks_per_face_outside_one_action_at_a_time(cfg):
    g = GestureRecognizer(cfg)
    t = 0.0
    for i in range(30):
        t = i / 15
        g.update({}, t, {1: eyes(0.2), 2: eyes(0.2)})
    g.update(motion(*hands_up()), t + 0.05, {1: eyes(0.8), 2: eyes(0.2)})
    ev = g.update(motion(*hands_up()), t + 0.1, {1: eyes(0.8), 2: eyes(0.2)})
    assert names(ev) == ["hands_up"]                      # gestures still work alongside
    ev = g.update(motion(*hands_up()), t + 0.2, {1: eyes(0.2), 2: eyes(0.2)})
    assert [(e.pid, e.name) for e in ev] == [(1, "blink")]
    s = FrameState(frame=None, players={}, motion={}, faces={}, events=ev, timings={})
    row = g.log_fields(s)
    assert row["gest_p1_events"] == "blink" and row["gest_p2_blink_base"] == 0.2
    assert row["gest_p1_eyes_closed"] == 0
    g.reset(1)
    assert 1 not in g._blinks
