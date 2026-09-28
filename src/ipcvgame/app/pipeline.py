"""Task 5 — wires the modules together, in the order from the setup guide:

    pose -> identity -> motion -> face -> gesture -> game -> render

Single-threaded; every module takes typed inputs and returns its own output
type (core/types.py). `until` stops after a given stage, so each owner can run
the pipeline up to their module and see only its debug overlay.
"""

from __future__ import annotations

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
from ipcvgame.scene.renderer import Renderer, draw_timings

STAGES = ["pose", "identity", "motion", "face", "gesture", "game", "render"]


class Pipeline:
    def __init__(self, cfg: dict, until: str | None = None):
        if until is not None and until not in STAGES:
            raise ValueError(f"until must be one of {STAGES}, got {until!r}")
        self.cfg = cfg
        self.until = until
        self._last_stage = STAGES.index(until) if until else len(STAGES) - 1
        self.prof = Profiler()

        self.pose = PoseEstimator(cfg["pose"])
        self.identity = IdentityTracker(cfg["identity"])
        self.motion = MotionFilter(cfg["motion"])
        self.face = FaceTracker(cfg["face"], cfg["motion"].get("kpt_conf_threshold", 0.5))
        self.gestures = GestureRecognizer(cfg["gestures"])
        self.game = Game(cfg["game"])
        self.renderer = Renderer(cfg["render"], FaceEffects(cfg["face"]))
        self._warmup(cfg["source"])
        self.debug_module = {
            "pose": self.pose, "identity": self.identity, "motion": self.motion,
            "face": self.face, "gesture": self.gestures, "game": self.game,
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
        players, motion, faces, events = {}, {}, {}, []

        with prof("pose"):
            dets = self.pose.detect(frame)
        if self._runs("identity"):
            with prof("identity"):
                players = self.identity.update(dets, frame)
            # The reentered flag tells Tasks 1 and 2 to drop filter state.
            for pid, p in players.items():
                if p.reentered:
                    self.face.reset(pid)
        if self._runs("motion"):
            with prof("motion"):
                motion = self.motion.update(players, frame.t)
        if self._runs("face"):
            with prof("face"):
                faces = self.face.update(frame, players)
        if self._runs("gesture"):
            with prof("gesture"):
                events = self.gestures.update(motion, frame.t)

        # Timings of the previous full frame (this one is not finished yet).
        state = FrameState(frame, players, motion, faces, events, prof.last())

        if self._runs("game"):
            with prof("game"):
                self.game.update(state)

        with prof("render"):
            if self.until is None or self.until == "render":
                canvas = self.renderer.draw(state, self.game)
            else:
                canvas = frame.image.copy()
                self.debug_module[self.until].draw_debug(canvas, state)
                draw_timings(canvas, state.timings)
        prof.end_frame()
        return state, canvas

    def log_fields(self, state: FrameState) -> dict:
        """Extra CSV columns from modules that define log_fields(state)."""
        row: dict = {}
        for m in self.debug_module.values():
            fn = getattr(m, "log_fields", None)
            if fn is not None:
                row.update(fn(state))
        return row

    def reset(self) -> None:
        for m in (self.pose, self.identity, self.motion, self.face, self.gestures, self.game):
            m.reset()
