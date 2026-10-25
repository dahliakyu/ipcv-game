"""Task 4 — gesture recognition: motion signals -> GestureEvent.

The rules are from the Colour Clash code (``Project/bodypose.py``), moved
here in the gesture merge (docs/merge_log.md, Task 4). They read the smoothed
keypoints and validity from ``Motion``; all distances are in shoulder widths
(``sw``), so they work at any distance from the camera. Every number comes
from the ``gestures:`` config.

``GestureRecognizer`` turns the rules into events: hands up and arms crossed
must hold for ``confirm_s`` (``Debounce`` in bodypose.py), and a punch stays on
for ``punch_hold_s`` after its last detection (``punch_until``). One event is
sent when a gesture turns on (the rising edge), so one punch gives one ``punch``
event; punches less than ``punch_hold_s`` apart count as one.
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.geometry import angle_at
from ipcvgame.core.types import FrameState, GestureEvent, Motion
from ipcvgame.scene.draw import draw_lines, draw_skeleton, draw_text, player_color

GESTURES = ("hands_up", "arms_crossed", "punch")


def lower_body_y(kpts: np.ndarray, valid: np.ndarray, shoulder_y: float, sw: float,
                 cfg: dict) -> float:
    """Hip level, or an estimate below the shoulders if hips are not visible."""
    if valid[sk.L_HIP] and valid[sk.R_HIP]:
        return (kpts[sk.L_HIP][1] + kpts[sk.R_HIP][1]) / 2.0
    return shoulder_y + cfg["hip_below_shoulders"] * sw


def detect_hands_up(kpts: np.ndarray, valid: np.ndarray, sw: float, cfg: dict) -> bool:
    """Both wrists above the head."""
    if not (valid[sk.L_WRIST] and valid[sk.R_WRIST]):
        return False

    if valid[sk.NOSE]:
        head_y = kpts[sk.NOSE][1]
    else:
        shoulder_y = (kpts[sk.L_SHOULDER][1] + kpts[sk.R_SHOULDER][1]) / 2.0
        head_y = shoulder_y - cfg["head_above_shoulders"] * sw

    return bool(kpts[sk.L_WRIST][1] < head_y and kpts[sk.R_WRIST][1] < head_y)


def detect_crossed(kpts: np.ndarray, valid: np.ndarray, sw: float, cfg: dict) -> bool:
    """Forearms crossed in front of the chest."""
    if not (valid[sk.L_WRIST] and valid[sk.R_WRIST]
            and valid[sk.L_SHOULDER] and valid[sk.R_SHOULDER]):
        return False

    lw, rw = kpts[sk.L_WRIST], kpts[sk.R_WRIST]
    ls, rs = kpts[sk.L_SHOULDER], kpts[sk.R_SHOULDER]

    # Wrists in the opposite left/right order to the shoulders
    # (independent of mirroring / label convention)
    swapped = (lw[0] - rw[0]) * (ls[0] - rs[0]) < 0

    shoulder_y = (ls[1] + rs[1]) / 2.0
    hip_y = lower_body_y(kpts, valid, shoulder_y, sw, cfg)

    top = shoulder_y - cfg["chest_above_shoulders"] * sw
    in_chest_zone = top < lw[1] < hip_y and top < rw[1] < hip_y

    close = np.linalg.norm(lw - rw) < cfg["cross_max_wrist_dist"] * sw

    return bool(swapped and in_chest_zone and close)


def arm_extended(kpts: np.ndarray, valid: np.ndarray, sw: float, shoulder: int,
                 elbow: int, wrist: int, cfg: dict) -> bool:
    """Arm is straight, reaching out sideways, and at roughly shoulder height."""
    if not (valid[shoulder] and valid[elbow] and valid[wrist]):
        return False

    straight = angle_at(kpts[shoulder], kpts[elbow], kpts[wrist]) > cfg["punch_elbow_angle"]

    # Sideways reach only (merge step 2b): a straight arm hanging down also
    # reaches 1 sw from the shoulder, but not to the side.
    reach = abs(kpts[wrist][0] - kpts[shoulder][0]) / sw > cfg["punch_min_reach"]

    # Wrist at about shoulder height: below the head (excludes hands up), and
    # at least level with its own shoulder (excludes low arms; was: above the
    # hips, merge step 2c). Image y grows downwards.
    shoulder_y = (kpts[sk.L_SHOULDER][1] + kpts[sk.R_SHOULDER][1]) / 2.0
    top = shoulder_y - cfg["punch_above_shoulders"] * sw
    bottom = kpts[shoulder][1] + cfg["punch_below_shoulder"] * sw
    at_height = top < kpts[wrist][1] < bottom

    return bool(straight and reach and at_height)


def arm_bent(kpts: np.ndarray, valid: np.ndarray, shoulder: int, elbow: int, wrist: int,
             cfg: dict) -> bool:
    """Arm is pulled in (guard). Unknown arms count as bent."""
    if not (valid[shoulder] and valid[elbow] and valid[wrist]):
        return True

    return angle_at(kpts[shoulder], kpts[elbow], kpts[wrist]) < cfg["punch_bent_angle"]


def detect_punch(kpts: np.ndarray, valid: np.ndarray, sw: float, cfg: dict) -> str | None:
    """Return 'L', 'R' or None: the model's side label of the punching arm.

    On our mirrored frame the model's 'L' is the player's own right arm (see
    core/skeleton.py). The event drops the side; it is kept here for debugging.
    """
    left = arm_extended(kpts, valid, sw, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, cfg)
    right = arm_extended(kpts, valid, sw, sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST, cfg)

    # One hand only: the other arm must be bent
    if left and arm_bent(kpts, valid, sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST, cfg):
        return "L"
    if right and arm_bent(kpts, valid, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, cfg):
        return "R"

    return None


def shoulder_width(kpts: np.ndarray, valid: np.ndarray, min_px: float) -> float | None:
    """Shoulder width in px, the unit of every rule; None if the shoulders are
    not both seen or are too close together to measure reliably."""
    if not (valid[sk.L_SHOULDER] and valid[sk.R_SHOULDER]):
        return None
    sw = float(np.linalg.norm(kpts[sk.L_SHOULDER] - kpts[sk.R_SHOULDER]))
    return sw if sw > min_px else None


def detect_all(kpts: np.ndarray, valid: np.ndarray, cfg: dict) -> dict[str, bool]:
    """All three rules on one frame, before any hold time."""
    sw = shoulder_width(kpts, valid, cfg["min_shoulder_px"])
    if sw is None:
        return dict.fromkeys(GESTURES, False)
    return {
        "hands_up": detect_hands_up(kpts, valid, sw, cfg),
        "arms_crossed": detect_crossed(kpts, valid, sw, cfg),
        "punch": detect_punch(kpts, valid, sw, cfg) is not None,
    }


class _PlayerState:
    """One player's timers (``Debounce`` and ``punch_until`` in bodypose.py)."""

    def __init__(self):
        # Start of the current run of frames where the rule holds; None = not holding.
        self.since: dict[str, float | None] = {"hands_up": None, "arms_crossed": None}
        self.punch_until = -np.inf  # the punch counts as on until this time
        self.active = dict.fromkeys(GESTURES, False)


class GestureRecognizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.confirm_s = float(cfg["confirm_s"])
        self.punch_hold_s = float(cfg["punch_hold_s"])
        self._state: dict[int, _PlayerState] = {}

    def update(self, motion: dict[int, Motion], t: float) -> list[GestureEvent]:
        # As in bodypose.py: a player without keypoints starts over.
        for pid in [pid for pid in self._state if pid not in motion]:
            del self._state[pid]

        events = []
        for pid, m in sorted(motion.items()):
            st = self._state.setdefault(pid, _PlayerState())
            raw = detect_all(m.kpts, m.valid, self.cfg)
            on = {}
            for name in ("hands_up", "arms_crossed"):
                if not raw[name]:
                    st.since[name] = None
                elif st.since[name] is None:
                    st.since[name] = t
                on[name] = st.since[name] is not None and t - st.since[name] >= self.confirm_s
            if raw["punch"]:
                st.punch_until = t + self.punch_hold_s
            on["punch"] = t < st.punch_until
            for name in GESTURES:
                if on[name] and not st.active[name]:
                    events.append(GestureEvent(pid=pid, name=name, t=t, conf=1.0))
                st.active[name] = on[name]
        return events

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._state.clear()
        else:
            self._state.pop(pid, None)

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Skeleton per player, the H/C/P line from bodypose.py under the body box
        (1 = gesture on), and this frame's events."""
        for pid, m in state.motion.items():
            draw_skeleton(canvas, m.kpts, m.valid, player_color(pid))
            st, p = self._state.get(pid), state.players.get(pid)
            if st is None or p is None:
                continue
            a = st.active
            text = f"H:{int(a['hands_up'])} C:{int(a['arms_crossed'])} P:{int(a['punch'])}"
            x1, y2 = int(p.bbox[0]), int(p.bbox[3])
            draw_text(canvas, text, (x1, min(y2 + 25, canvas.shape[0] - 5)), (255, 255, 0))
        lines = [f"EVENT P{e.pid} {e.name}" for e in state.events]
        draw_lines(canvas, lines, (10, canvas.shape[0] - 20 * len(lines) - 10))

    def log_fields(self, state: FrameState) -> dict:
        """Per-player CSV columns gest_p<pid>_<gesture> (1 while on) and
        gest_p<pid>_events (names of this frame's events, joined by '+')."""
        row: dict = {}
        for pid, st in sorted(self._state.items()):
            pre = f"gest_p{pid}_"
            for name in GESTURES:
                row[pre + name] = int(st.active[name])
            row[pre + "events"] = "+".join(e.name for e in state.events if e.pid == pid)
        return row
