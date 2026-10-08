"""Rigged avatar (resource/avatar.glb) with code-driven arms.

The model is rigged in T-pose. We take over the upper-arm and forearm joints
and set them each frame from a desired direction per bone:

1. Two-bone IK (law of cosines) turns a hand target into an elbow position.
   A pole vector picks which way the elbow bends.
2. For each bone, the rotation that turns its rest direction into the new
   direction (shortest arc) is applied on top of its rest orientation.
3. The joint's local transform is that new orientation relative to its
   parent joint, which is what Panda3D's controlJoint expects.

Later, pose keypoints can skip step 1: shoulder->elbow and elbow->wrist from
the camera give the two bone directions directly.
"""

from __future__ import annotations

import numpy as np
from direct.actor.Actor import Actor
from panda3d.core import Filename, LQuaternionf, LVector3f, NodePath, Point3


def _v(a) -> np.ndarray:
    return np.array([a[0], a[1], a[2]], dtype=float)


def shortest_arc(a: np.ndarray, b: np.ndarray) -> LQuaternionf:
    """Quaternion rotating direction a onto direction b by the smallest angle."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    angle = np.arctan2(s, float(a @ b))
    if s < 1e-8:  # parallel or opposite: any perpendicular axis works
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    q = LQuaternionf()
    q.setFromAxisAngleRad(float(angle), LVector3f(*(axis / np.linalg.norm(axis))))
    return q


def two_bone_ik(shoulder: np.ndarray, target: np.ndarray, l1: float, l2: float,
                pole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elbow and hand positions that put the hand as close to target as the
    arm allows. The elbow lies in the plane of (shoulder->target, pole)."""
    d = target - shoulder
    dist = float(np.linalg.norm(d))
    u = d / dist if dist > 1e-9 else np.array([0.0, 1.0, 0.0])
    dist = float(np.clip(dist, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)   # shoulder -> elbow foot point
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))                # elbow distance off the line
    p = pole - (pole @ u) * u
    p = p / np.linalg.norm(p) if np.linalg.norm(p) > 1e-9 else np.array([0.0, 0.0, -1.0])
    elbow = shoulder + a * u + h * p
    return elbow, shoulder + dist * u


class Arm:
    """One arm: shoulder.X -> upper_arm.X -> forearm.X -> hand.X (X = L or R)."""

    def __init__(self, actor: Actor, side: str, palm_frac: float):
        names = [f"upper_arm.{side}", f"forearm.{side}", f"hand.{side}", f"shoulder.{side}"]
        exposed = {n: actor.exposeJoint(None, "modelRoot", n) for n in names}
        rest = {n: exposed[n].getTransform(actor) for n in names}
        # hand tail is not a joint; approximate the palm a fraction along the
        # forearm direction past the wrist (hand bone ~0.25 of the arm here)
        p_sh = _v(rest[names[0]].getPos())
        p_el = _v(rest[names[1]].getPos())
        p_wr = _v(rest[names[2]].getPos())
        p_palm = p_wr + palm_frac * (p_wr - p_el)
        self.actor = actor
        self.rest_parent = rest[names[3]]
        self.rest_upper, self.rest_fore = rest[names[0]], rest[names[1]]
        self.shoulder = p_sh
        self.dir_upper = p_el - p_sh
        self.dir_fore = p_palm - p_el
        self.l1 = float(np.linalg.norm(self.dir_upper))
        self.l2 = float(np.linalg.norm(self.dir_fore))
        self.ctrl_upper = actor.controlJoint(None, "modelRoot", names[0])
        self.ctrl_fore = actor.controlJoint(None, "modelRoot", names[1])

    def set_directions(self, upper: np.ndarray, fore: np.ndarray) -> np.ndarray:
        """Point both bones along the given actor-space directions. Returns
        the palm position in actor space."""
        q_up = self.rest_upper.getQuat() * shortest_arc(self.dir_upper, upper)
        net_up = self.rest_upper.setQuat(q_up)
        elbow = self.shoulder + self.l1 * upper / np.linalg.norm(upper)
        q_fo = self.rest_fore.getQuat() * shortest_arc(self.dir_fore, fore)
        net_fo = self.rest_fore.setQuat(q_fo).setPos(Point3(*elbow))
        self.ctrl_upper.setTransform(self.rest_parent.invertCompose(net_up))
        self.ctrl_fore.setTransform(net_up.invertCompose(net_fo))
        return elbow + self.l2 * fore / np.linalg.norm(fore)

    def reach(self, target: np.ndarray, pole: np.ndarray) -> np.ndarray:
        """IK to an actor-space target; returns the palm in actor space."""
        elbow, hand = two_bone_ik(self.shoulder, target, self.l1, self.l2, pole)
        return self.set_directions(elbow - self.shoulder, hand - elbow)


class Avatar:
    """The model faces -y in its own frame, so its right side is -x.

    `heading` is Panda3D's H in degrees: 180 faces +y (squash), 90 faces +x.
    """

    def __init__(self, parent: NodePath, model_path: str | Filename, cfg: dict,
                 heading: float = 180.0):
        self.actor = Actor(model_path)
        self.actor.reparentTo(parent)
        self.actor.setScale(cfg["avatar_scale"])
        self.actor.setH(heading)
        self.arms = {"R": Arm(self.actor, "R", cfg["palm_frac"]),
                     "L": Arm(self.actor, "L", cfg["palm_frac"])}
        # idle arm: down and slightly out
        self._idle_dir = {s: self.body_dir(s, cfg["idle_arm_dir"]) for s in ("R", "L")}

    @staticmethod
    def body_dir(side: str, d) -> np.ndarray:
        """(out, forward, up) for one arm -> direction in the model's frame,
        where right = -x and forward = -y; "out" is away from the body."""
        out, fwd, up = d
        return np.array([-out if side == "R" else out, -fwd, up], dtype=float)

    def set_x(self, x: float) -> None:
        self.actor.setX(x)

    def set_pos(self, x: float, y: float) -> None:
        self.actor.setPos(x, y, 0.0)

    def reach_right(self, target_world: np.ndarray, pole_world: np.ndarray) -> np.ndarray:
        """Move the right hand towards a world target; returns the palm in world."""
        return self.reach("R", target_world, pole_world)

    def reach(self, side: str, target_world: np.ndarray, pole_world: np.ndarray) -> np.ndarray:
        """Move one hand ("R" or "L") towards a world target and let the other
        arm hang. Returns the palm in world coordinates."""
        render = self.actor.getTop()
        t = _v(self.actor.getRelativePoint(render, Point3(*target_world)))
        pole = _v(self.actor.getRelativeVector(render, LVector3f(*pole_world)))
        palm = self.arms[side].reach(t, pole)
        other = "L" if side == "R" else "R"
        self.arms[other].set_directions(self._idle_dir[other], self._idle_dir[other])
        return _v(render.getRelativePoint(self.actor, Point3(*palm)))

    def to_world(self, local_dir) -> np.ndarray:
        """Direction in the model's own frame -> world direction."""
        render = self.actor.getTop()
        return _v(render.getRelativeVector(self.actor, LVector3f(*local_dir)))
