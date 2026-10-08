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


class Limb:
    """Two bones: parent -> upper -> lower -> end joint. The arm is
    shoulder.X -> upper_arm.X -> forearm.X -> hand.X, the leg
    spine -> thigh.X -> shin.X -> foot.X. `parent` must be the upper joint's
    real parent in the rig, or the upper bone is posed in the wrong frame.

    The tip is `tip_frac` of the lower bone past the end joint: the palm for
    an arm (the hand tail is not a joint), the ankle itself (0) for a leg.
    With `hold_end`, the end joint keeps its rest orientation in actor space,
    so a foot stays flat on the floor however the shin turns.
    """

    def __init__(self, actor: Actor, parent: str, upper: str, lower: str, end: str,
                 tip_frac: float, hold_end: bool = False):
        names = [upper, lower, end, parent]
        exposed = {n: actor.exposeJoint(None, "modelRoot", n) for n in names}
        rest = {n: exposed[n].getTransform(actor) for n in names}
        p_base = _v(rest[upper].getPos())
        p_mid = _v(rest[lower].getPos())
        p_end = _v(rest[end].getPos())
        p_tip = p_end + tip_frac * (p_end - p_mid)
        self.actor = actor
        self.rest_parent = rest[parent]
        self.rest_upper, self.rest_fore, self.rest_end = rest[upper], rest[lower], rest[end]
        self.base = p_base
        self.rest_tip = p_tip
        self.dir_upper = p_mid - p_base
        self.dir_fore = p_tip - p_mid
        self.l1 = float(np.linalg.norm(self.dir_upper))
        self.l2 = float(np.linalg.norm(self.dir_fore))
        self.ctrl_upper = actor.controlJoint(None, "modelRoot", upper)
        self.ctrl_fore = actor.controlJoint(None, "modelRoot", lower)
        self.ctrl_end = actor.controlJoint(None, "modelRoot", end) if hold_end else None

    def set_directions(self, upper: np.ndarray, fore: np.ndarray) -> np.ndarray:
        """Point both bones along the given actor-space directions. Returns
        the tip position in actor space."""
        q_up = self.rest_upper.getQuat() * shortest_arc(self.dir_upper, upper)
        net_up = self.rest_upper.setQuat(q_up)
        mid = self.base + self.l1 * upper / np.linalg.norm(upper)
        q_fo = self.rest_fore.getQuat() * shortest_arc(self.dir_fore, fore)
        net_fo = self.rest_fore.setQuat(q_fo).setPos(Point3(*mid))
        self.ctrl_upper.setTransform(self.rest_parent.invertCompose(net_up))
        self.ctrl_fore.setTransform(net_up.invertCompose(net_fo))
        tip = mid + self.l2 * fore / np.linalg.norm(fore)
        if self.ctrl_end is not None:
            net_end = self.rest_end.setPos(Point3(*tip))
            self.ctrl_end.setTransform(net_fo.invertCompose(net_end))
        return tip

    def reach(self, target: np.ndarray, pole: np.ndarray) -> np.ndarray:
        """IK to an actor-space target; returns the tip in actor space."""
        mid, tip = two_bone_ik(self.base, target, self.l1, self.l2, pole)
        return self.set_directions(mid - self.base, tip - mid)


class Arm(Limb):
    """One arm: shoulder.X -> upper_arm.X -> forearm.X -> hand.X (X = L or R)."""

    def __init__(self, actor: Actor, side: str, palm_frac: float):
        super().__init__(actor, f"shoulder.{side}", f"upper_arm.{side}", f"forearm.{side}",
                         f"hand.{side}", palm_frac)


class Leg(Limb):
    """One leg: spine -> thigh.X -> shin.X -> foot.X; the foot stays flat.

    The thigh's parent joint is the root `spine` (the hips), not pelvis.X:
    in this rig pelvis.X is a sibling (check with `actor.listJoints()`).
    """

    def __init__(self, actor: Actor, side: str):
        super().__init__(actor, "spine", f"thigh.{side}", f"shin.{side}",
                         f"foot.{side}", 0.0, hold_end=True)


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
        self._legs: dict[str, Leg] | None = None  # taken over on the first crouch only

    def crouch(self, drop: float) -> None:
        """Lower the body by `drop` metres; the knees bend forward so the
        feet stay where they were on the floor."""
        if self._legs is None:
            if drop <= 0:
                return
            self._legs = {s: Leg(self.actor, s) for s in ("R", "L")}
        self.actor.setZ(-drop)
        up = drop / self.actor.getSz()   # world metres -> model units
        for s, leg in self._legs.items():
            leg.reach(leg.rest_tip + np.array([0.0, 0.0, up]), self.body_dir(s, (0.0, 1.0, 0.0)))

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
