"""Task 5 — wires the modules together, in the order from the setup guide:

    pose -> identity -> motion -> face -> gestures -> game -> render

Single-threaded; every module takes typed inputs and returns its own output
type (core/types.py). `until` stops after a given stage, so each owner can run
the pipeline up to their module and see only its debug overlay.

A failing module does not stop the game (brief §3.2.5, "module failures in a
controlled way"; config `pipeline:`). Every stage runs inside a guard: when it
raises, the error is printed (with its traceback the first time), the stage's
output for this frame is empty (no detections, players, faces or events; the
game does not advance; the raw frame instead of the game view), and the
next frame tries again. A stage that keeps failing for
`fallback_after_s` is swapped for its stub backend where it has one
(identity: zones, motion: stub, face: stub), as the config switch would do
at startup. The failed stages are logged (`pipeline_failed`,
`pipeline_fallback`) and named on screen.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import TypeVar

import cv2
import numpy as np

from ipcvgame.core.timing import Profiler
from ipcvgame.core.types import Frame, FrameState
from ipcvgame.face.effects import FaceEffects
from ipcvgame.face.tracker import FaceTracker
from ipcvgame.identity.tracker import IdentityTracker
from ipcvgame.interaction.game import Game
from ipcvgame.interaction.gestures import GestureRecognizer
from ipcvgame.pose.estimator import PoseEstimator
from ipcvgame.pose.motion import MotionFilter
from ipcvgame.scene.draw import draw_text
from ipcvgame.scene.renderer import Renderer, draw_timings

STAGES = ["pose", "identity", "motion", "face", "gestures", "game", "render"]
# The stub backend a stage falls back to when it keeps failing, and the
# class to rebuild it with.
FALLBACK = {
    "identity": ("zones", IdentityTracker),
    "motion": ("stub", MotionFilter),
    "face": ("stub", FaceTracker),
}
ERROR_COLOR = (60, 60, 255)    # BGR red, for the failure note

T = TypeVar("T")


class Pipeline:
    def __init__(self, cfg: dict, until: str | None = None):
        if until is not None and until not in STAGES:
            raise ValueError(f"until must be one of {STAGES}, got {until!r}")
        self.cfg = cfg
        self.until = until
        pcfg = cfg.get("pipeline", {})
        self.catch_errors = bool(pcfg.get("catch_errors", True))
        self.fallback_after_s = float(pcfg.get("fallback_after_s", 1.0))
        self.failed: list[str] = []                 # stages that failed this frame
        self.fallen_back: set[str] = set()          # stages swapped for their stub
        self._fail_since: dict[str, float] = {}     # stage -> Frame.t of its first failure in a row
        self._reported: set[str] = set()            # stages whose traceback was printed
        self._last_stage = STAGES.index(until) if until else len(STAGES) - 1
        self.prof = Profiler()

        self.pose = PoseEstimator(cfg["pose"])
        self.identity = IdentityTracker(cfg["identity"])
        self.motion = MotionFilter(cfg["motion"])
        self.face = FaceTracker(cfg["face"])
        self.gestures = GestureRecognizer(cfg["gestures"])
        self.game = Game(cfg["game"])
        self.renderer = Renderer(cfg["render"], FaceEffects(cfg["face"]))
        self._warmup(cfg["source"])

    @property
    def debug_module(self) -> dict:
        """Stage -> module with draw_debug(), the current one (a stage that
        fell back has a new module)."""
        return {
            "pose": self.pose, "identity": self.identity, "motion": self.motion,
            "face": self.face, "gestures": self.gestures, "game": self.game,
        }

    def _warmup(self, source_cfg: dict) -> None:
        """One dummy inference so CUDA/model init (seconds) happens before the
        first real frame, not between frames 0 and 1 where it would show up
        as a huge dt in the filters, the logs and recorded clips."""
        size = (source_cfg.get("height", 720), source_cfg.get("width", 1280), 3)
        self.pose.detect(Frame(np.zeros(size, np.uint8), -1, 0.0))

    def _runs(self, stage: str) -> bool:
        return STAGES.index(stage) <= self._last_stage

    def step(self, frame: Frame) -> tuple[FrameState, np.ndarray]:
        """Run one frame through the pipeline. Returns the state and the canvas
        (game view, or the `until` module's debug overlay)."""
        prof = self.prof
        prof.start_frame()
        self.failed = []
        players, motion, faces, events = {}, {}, {}, []
        t = frame.t

        with prof("pose"):
            dets = self._guard("pose", t, lambda: self.pose.detect(frame), [])
        if self._runs("identity"):
            with prof("identity"):
                players = self._guard("identity", t, lambda: self.identity.update(dets, frame), {})
            # The reentered flag tells Tasks 1, 2 and 4 to drop per-player state.
            for pid, p in players.items():
                if p.reentered:
                    self.face.reset(pid)
                    self.renderer.face_effects.reset(pid)
                    self.renderer.reset(pid)
                    self.gestures.reset(pid)
        if self._runs("motion"):
            with prof("motion"):
                motion = self._guard("motion", t, lambda: self.motion.update(players, t), {})
        if self._runs("face"):
            with prof("face"):
                faces = self._guard("face", t, lambda: self.face.update(frame, players), {})
        if self._runs("gestures"):
            with prof("gestures"):
                events = self._guard("gestures", t, lambda: self.gestures.update(motion, t, faces), [])

        # Timings of the previous full frame (this one is not finished yet).
        state = FrameState(frame, players, motion, faces, events, prof.last())

        if self._runs("game"):
            with prof("game"):
                phase = self.game.phase

                def game_step() -> None:
                    self.game.update(state)
                    self._calibrate_identity(state, phase)
                self._guard("game", t, game_step, None)

        with prof("render"):
            canvas = self._guard("render", t, lambda: self._render(state), None)
            if canvas is None:                     # the render stage failed: the plain frame
                canvas = frame.image.copy()
            self._draw_failures(canvas)
        prof.end_frame()
        return state, canvas

    def _render(self, state: FrameState) -> np.ndarray:
        if self.until is None or self.until == "render":
            return self.renderer.draw(state, self.game)
        canvas = state.frame.image.copy()
        if self.cfg["render"].get("debug_face_effects", True):
            self.renderer.face_effects.draw(canvas, state.faces)
        self.debug_module[self.until].draw_debug(canvas, state)
        draw_timings(canvas, state.timings)
        return canvas

    def _guard(self, stage: str, t: float, fn: Callable[[], T], empty: T) -> T:
        """fn() for one stage; on an exception (with pipeline.catch_errors) its
        empty output instead, and after fallback_after_s of failing in a row
        the stage's stub backend (FALLBACK). See the module doc."""
        if not self.catch_errors:
            return fn()
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - any module error must not stop the game
            self.failed.append(stage)
            since = self._fail_since.setdefault(stage, t)
            if stage not in self._reported:
                self._reported.add(stage)
                print(f"[pipeline] {stage} failed; the game goes on without it this frame:")
                traceback.print_exc()
            elif since == t:
                print(f"[pipeline] {stage} failed again: {exc!r}")
            if (stage in FALLBACK and stage not in self.fallen_back
                    and t - since >= self.fallback_after_s):
                self._fall_back(stage)
            return empty
        self._fail_since.pop(stage, None)
        return out

    def _fall_back(self, stage: str) -> None:
        """Swap `stage`'s module for its stub backend, with fresh state."""
        backend, cls = FALLBACK[stage]
        try:
            module = cls({**self.cfg[stage], "backend": backend})
        except Exception:  # noqa: BLE001 - keep the failing module rather than stop
            print(f"[pipeline] {stage}: the {backend!r} stub could not be built either")
            traceback.print_exc()
            return
        setattr(self, stage, module)
        self.fallen_back.add(stage)
        self._fail_since.pop(stage, None)
        print(f"[pipeline] {stage} kept failing for {self.fallback_after_s:g} s: "
              f"switched to the {backend!r} backend")

    def _draw_failures(self, canvas: np.ndarray) -> None:
        """Name the failing and fallen-back stages at the top right."""
        notes = [f"{s.upper()} FAILED" for s in self.failed if s not in self.fallen_back]
        notes += [f"{s.upper()}: STUB" for s in sorted(self.fallen_back)]
        if not notes:
            return
        text = "  ".join(notes)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        draw_text(canvas, text, (max(4, canvas.shape[1] - tw - 10), 60), ERROR_COLOR, 0.5)

    def _calibrate_identity(self, state: FrameState, phase_before: str) -> None:
        """Game start (docs/merge_log.md, steps 5-6): the game tells identity who is who.

        - Calibration has just started: P1 becomes the player in the left
          start place, P2 the one in the right, so the place colours and the
          player colours match (identity numbers players in the order it
          first sees them). With one round player there is nothing to order.
        - A player's neutral pose was just measured: their stored OSNet
          vector is replaced with one from this frame, and their shoulder
          width and torso length become the depth reference (Player.pos).
        """
        game = self.game
        if phase_before == "waiting" and game.phase == "calibration":
            xs = {pid: game.player_x(state, pid) for pid in game.round_players
                  if game.in_view(state, pid)}
            self.identity.order_left_to_right(xs)
        for pid, action in game.actions:
            if action == "learned_neutral":
                self.identity.refresh_vector(pid, state.frame)
                player = game.players[pid]
                self.identity.set_body_reference(pid, player.shoulder_px, player.torso_px)

    def log_fields(self, state: FrameState) -> dict:
        """Extra CSV columns from modules that define log_fields(state)."""
        row: dict = {}
        if self.failed:
            row["pipeline_failed"] = "|".join(self.failed)
        if self.fallen_back:
            row["pipeline_fallback"] = "|".join(sorted(self.fallen_back))
        for m in (*self.debug_module.values(), self.renderer):
            fn = getattr(m, "log_fields", None)
            if fn is not None:
                row.update(fn(state))
        return row

    def new_round(self) -> None:
        """A new round with the same players (key r): from the countdown,
        keeping their calibration."""
        self.game.new_round()

    def test_hit(self) -> bool:
        """Throw a test blob at P1 (key b), to try hits and paint alone."""
        return self.game.test_hit(1)

    def test_orb(self) -> bool:
        """Fill P1's orb (key n), to try the final smash alone."""
        return self.game.test_orb(1)

    def reset(self) -> None:
        """Clear every module's state; identity finds the players again."""
        for m in (self.pose, self.identity, self.motion, self.face, self.gestures, self.game,
                  self.renderer.face_effects, self.renderer):
            m.reset()
