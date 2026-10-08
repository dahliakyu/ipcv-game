"""Rally rules for the squash prototype: serve, hit, wall, miss.

State machine, one ball, one player:

    ready --serve--> incoming --hand touches ball--> outgoing --front wall--> incoming ...
                       |                                  |
           2nd floor bounce / passed         floor before wall / tin
                       +---------------> missed <---------+---> (pause) ---> ready

No rendering here; app.py draws whatever this state says. If the prototype
becomes the game, these rules move to interaction/game.py (Task 4).
"""

from __future__ import annotations

import numpy as np

from ipcvgame.scene.squash.physics import Ball, Court, aim, step, swept_hit


class Rally:
    def __init__(self, cfg: dict, seed: int | None = None):
        self.cfg = cfg
        self.court = Court.from_cfg(cfg)
        self.rng = np.random.default_rng(seed)
        self.ball = Ball(np.zeros(3), np.zeros(3))
        self.state = "ready"
        self.count = 0          # hits in the current rally
        self.best = 0
        self.message = "click or SPACE to serve"
        self._bounces = 0       # floor bounces since the ball last left a wall or the hand
        self._t_state = 0.0     # time spent in the current state
        self._park_ball()

    # --- control -------------------------------------------------------------

    def serve(self) -> None:
        if self.state not in ("ready", "missed"):
            return
        c = self.cfg
        x = self.rng.uniform(-c["serve_x_range"], c["serve_x_range"])
        self.ball.pos = np.array([x, self.court.wall_y - self.court.ball_radius, c["serve_height"]])
        self.ball.vel = aim(self.ball.pos, self._reach_target(), c["serve_speed"], self.court)
        self.count = 0
        self._bounces = 0  # the missed rally may have ended on a bounce
        self._set("incoming", "")

    def reset(self) -> None:
        self.best = 0
        self.count = 0
        self._park_ball()
        self._set("ready", "click or SPACE to serve")

    # --- per frame -------------------------------------------------------------

    def update(self, dt: float, hand0: np.ndarray, hand1: np.ndarray, hand_vel: np.ndarray) -> list[str]:
        """Advance one frame. hand0/hand1 are the hand positions at the start
        and end of the frame (for the swept hit test). Returns event names for
        HUD and sound: "serve", "hit", "wall", "bounce", "miss"."""
        self._t_state += dt
        events: list[str] = []
        if self.state == "ready":
            if self._t_state >= self.cfg["auto_serve_delay"]:
                self.serve()
                events.append("serve")
            return events
        if self.state == "missed":
            if self._t_state >= self.cfg["miss_pause"]:
                self._park_ball()
                self._set("ready", "click or SPACE to serve")
            return events

        ball0 = self.ball.pos.copy()
        hits = step(self.ball, self.court, dt)
        if self.state == "incoming" and swept_hit(ball0, self.ball.pos, hand0, hand1, self.cfg["hit_radius"]):
            self._return_ball(hand_vel)
            events.append("hit")
            return events

        for h in hits:
            if h == "floor":
                self._bounces += 1
                events.append("bounce")
                if self.state == "outgoing":
                    return events + self._miss("down: floor before the front wall")
                if self._bounces >= 2:
                    return events + self._miss("not up: second bounce")
            elif h == "front_wall" and self.state == "outgoing":
                if self.ball.pos[2] < self.cfg["tin_height"]:
                    return events + self._miss("tin")
                self._wall_rebound()
                events.append("wall")
        if self.state == "incoming" and self.ball.pos[1] < self.cfg["miss_y"]:
            return events + self._miss("missed")
        return events

    # --- internals ---------------------------------------------------------------

    def _return_ball(self, hand_vel: np.ndarray) -> None:
        """Send the ball back to the front wall. Speed grows with swing speed;
        sideways and vertical hand motion steer the ball."""
        c = self.cfg
        swing = float(np.linalg.norm(hand_vel))
        vy = min(c["hit_speed_min"] + c["hit_speed_gain"] * swing, c["hit_speed_max"])
        vx = c["hit_aim_gain"] * hand_vel[0]
        vz = c["hit_lift"] + c["hit_aim_gain"] * hand_vel[2]
        self.ball.vel = np.array([vx, vy, vz])
        self.count += 1
        self.best = max(self.best, self.count)
        self._bounces = 0
        self._set("outgoing", "")

    def _wall_rebound(self) -> None:
        """Ball leaves the front wall. With assist on, it is re-aimed so it
        arrives within reach after one bounce. The rebound speed still comes
        from how hard it was hit."""
        speed = abs(self.ball.vel[1])
        if self.cfg["assist"]:
            speed = float(np.clip(speed, self.cfg["rebound_speed_min"], self.cfg["rebound_speed_max"]))
            self.ball.vel = aim(self.ball.pos, self._reach_target(), speed, self.court)
        self._bounces = 0
        self._set("incoming", "")

    def _reach_target(self) -> np.ndarray:
        c = self.cfg
        x = self.rng.uniform(-c["target_x_range"], c["target_x_range"])
        z = self.rng.uniform(*c["target_z_range"])
        return np.array([x, c["hit_y"], z])

    def _miss(self, why: str) -> list[str]:
        self._set("missed", f"{why}  -  rally {self.count}")
        return ["miss"]

    def _park_ball(self) -> None:
        self.ball.pos = np.array([0.0, self.court.wall_y - 0.5, self.cfg["serve_height"]])
        self.ball.vel = np.zeros(3)

    def _set(self, state: str, message: str) -> None:
        self.state = state
        self.message = message
        self._t_state = 0.0
