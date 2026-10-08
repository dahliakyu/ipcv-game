"""Color Clash bolt prototype: two avatars on a field, Magic Bolts as balls.

    python -m ipcvgame.scene.clash.app [--set clash.bolt_speed=12]

Players move in 1D, up and down the field depth. P1 (blue, left) follows the
mouse cursor and fires with a left click. P2 (red, right) walks with W/S and
fires with SPACE. A hit tints the victim
towards the attacker's colour; the first player fully covered loses. R
restarts the round, ESC quits.

Rendering only: rules live in combat.py, input in control.py, the avatar and
its arm IK in scene/squash/avatar.py.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    Filename,
    KeyboardButton,
    NodePath,
    OrthographicLens,
    Plane,
    Point3,
    TextNode,
    TransparencyAttrib,
    Vec3,
    loadPrcFileData,
)

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.scene.clash.combat import PIDS, Combat, Event
from ipcvgame.scene.clash.control import KeyboardControl, MouseFloorControl
from ipcvgame.scene.squash.app import make_sphere, rect
from ipcvgame.scene.squash.avatar import Avatar

FLOOR = np.array([0.55, 0.56, 0.60])
FLOOR_TINT = 0.35           # how much of a player's colour shows on their half of the floor
MAX_SPLATS = 80             # floor splats kept per round (oldest removed first)


def lerp(a, b, k: float) -> np.ndarray:
    return (1 - k) * np.asarray(a, dtype=float) + k * np.asarray(b, dtype=float)


def flat(parent: NodePath, color, alpha: float, radius: float = 1.0) -> NodePath:
    """Unlit, see-through sphere: bolts, glows, splats and shadows."""
    n = parent.attachNewNode(make_sphere(radius))
    n.setColor(*color, alpha)
    n.setTransparency(TransparencyAttrib.MAlpha)
    n.setLightOff()
    n.setDepthWrite(False)
    return n


class BoltView:
    """Nodes for one bolt: core, glow, trail and a floor shadow. Pooled."""

    def __init__(self, parent: NodePath, r: float, trail_len: int):
        self.core = flat(parent, (1, 1, 1), 1.0, r)
        self.glow = flat(parent, (1, 1, 1), 0.3, 1.8 * r)
        self.trail = [flat(parent, (1, 1, 1), 0.0, r) for _ in range(trail_len)]
        self.shadow = flat(parent, (0, 0, 0), 0.4, r)
        self.shadow.setScale(1.2, 1.2, 0.02)  # flattened: the depth cue in an ortho view
        self.nodes = [self.core, self.glow, self.shadow, *self.trail]

    def show(self, pos: np.ndarray, trail: list[np.ndarray], color) -> None:
        self.core.setPos(*pos)
        self.core.setColor(*lerp(color, (1, 1, 1), 0.5), 1)  # bright centre
        self.glow.setPos(*pos)
        self.glow.setColor(*color, 0.35)
        self.shadow.setPos(pos[0], pos[1], 0.005)
        for i, node in enumerate(self.trail):
            j = len(trail) - 1 - i              # newest trail point first
            if j < 0:
                node.hide()
                continue
            k = 1 - (i + 1) / (len(self.trail) + 1)
            node.setPos(*trail[j])
            node.setScale(k)
            node.setColor(*color, 0.5 * k)
            node.show()
        for n in (self.core, self.glow, self.shadow):
            n.show()

    def hide(self) -> None:
        for n in self.nodes:
            n.hide()


class Fx:
    """Short effect: a sphere that grows and fades out."""

    def __init__(self, node: NodePath, color, t0: float, dur: float, r0: float, r1: float):
        self.node, self.color, self.t0, self.dur, self.r0, self.r1 = node, color, t0, dur, r0, r1

    def update(self, t: float) -> bool:
        k = (t - self.t0) / self.dur
        if k >= 1:
            self.node.removeNode()
            return False
        self.node.setScale(self.r0 + (self.r1 - self.r0) * k)
        self.node.setColor(*self.color, 0.8 * (1 - k))
        return True


class ClashApp(ShowBase):
    def __init__(self, cfg: dict):
        w, h = cfg["window_size"]
        loadPrcFileData("", f"win-size {w} {h}\nwindow-title IPCV Color Clash prototype\n"
                            "framebuffer-multisample 1\nmultisamples 4\nsync-video 1")
        super().__init__()
        self.cfg = cfg
        self.disableMouse()
        self.setBackgroundColor(0.10, 0.11, 0.14, 1)
        self.render.setShaderAuto()
        self._lights()
        self._camera(w, h)

        self.combat = Combat(cfg)
        self.arena = self.combat.arena
        self.colors = {pid: np.array(c, dtype=float)
                       for pid, c in zip(PIDS, cfg["player_colors"], strict=True)}
        self.fire_arm = dict(zip(PIDS, cfg["fire_arm"], strict=True))
        self._field()

        model = Filename.fromOsSpecific(str(REPO_ROOT / cfg["model"]))
        cheat = cfg["facing_cheat"]
        # model faces -y; H = 90 faces +x. Turn both towards the camera (-y).
        self.avatars = {1: Avatar(self.render, model, cfg, heading=90 - cheat),
                        2: Avatar(self.render, model, cfg, heading=-90 + cheat)}
        self.rings, self.lanes = {}, {}
        for pid in PIDS:
            ring = flat(self.render, self.colors[pid], 0.55)
            ring.setScale(0.45, 0.45, 0.01)
            self.rings[pid] = ring
            # thin line on the floor along the bolt's path: helps aim in depth
            lane = rect(self.render, 0, 1, -0.5, 0.5, (*self.colors[pid], 0.25), (0, 0, 0.004), (0, -90, 0))
            lane.setTransparency(TransparencyAttrib.MAlpha)
            lane.setLightOff()
            lane.setDepthWrite(False)
            self.lanes[pid] = lane

        self.bolt_views: list[BoltView] = []
        self.fx: list[Fx] = []
        self.splats: list[NodePath] = []
        self.rng = np.random.default_rng()
        self.palms = {pid: np.zeros(3) for pid in PIDS}
        self.want_fire = {pid: False for pid in PIDS}

        self.mouse = MouseFloorControl(cfg, self._floor_point, self.arena.start(1))
        self.keys = KeyboardControl(cfg, self._key_down, self.arena.start(2),
                                    lambda p: self.arena.clamp(2, p))
        self._hud()

        self.accept("escape", self.userExit)
        self.accept("mouse1", self._request_fire, [1])
        self.accept("space", self._request_fire, [2])
        self.accept("r", self._restart)
        self.taskMgr.add(self._tick, "clash-tick")

    # --- scene ------------------------------------------------------------------

    def _lights(self) -> None:
        sun = self.render.attachNewNode(DirectionalLight("sun"))
        sun.setHpr(15, -45, 0)  # from behind the camera: faces are lit
        self.render.setLight(sun)
        amb = self.render.attachNewNode(AmbientLight("amb"))
        amb.node().setColor((0.5, 0.5, 0.53, 1))
        self.render.setLight(amb)

    def _camera(self, w: int, h: int) -> None:
        c = self.cfg
        el = math.radians(c["camera_elevation"])
        look = np.array(c["camera_look_at"], dtype=float)
        pos = look + c["camera_distance"] * np.array([0.0, -math.cos(el), math.sin(el)])
        self.lens = OrthographicLens()
        self.lens.setFilmSize(c["film_width"], c["film_width"] * h / w)
        self.lens.setNearFar(1.0, 2 * c["camera_distance"])
        self.cam.node().setLens(self.lens)
        self.camera.setPos(*pos)
        self.camera.lookAt(*look)

    def _field(self) -> None:
        a = self.arena
        L, D = a.half_length, a.half_depth
        field = self.render.attachNewNode("field")
        self.halves = {1: rect(field, -L, 0, -D, D, (*FLOOR, 1), (0, 0, 0), (0, -90, 0)),
                       2: rect(field, 0, L, -D, D, (*FLOOR, 1), (0, 0, 0), (0, -90, 0))}
        white = (0.95, 0.95, 0.95, 1)
        rect(field, -0.03, 0.03, -D, D, white, (0, 0, 0.002), (0, -90, 0))           # centre line
        for pid in PIDS:                                                              # movement tracks
            x, m = a.start(pid)[0], D - a.edge_margin
            rect(field, x - 0.02, x + 0.02, -m, m, (0.85, 0.85, 0.85, 1), (0, 0, 0.002), (0, -90, 0))
        for y in (-D, D):                                                             # edges
            rect(field, -L, L, y - 0.03, y + 0.03, white, (0, 0, 0.002), (0, -90, 0))
        for x in (-L, L):
            rect(field, x - 0.03, x + 0.03, -D, D, white, (0, 0, 0.002), (0, -90, 0))
        self.splat_root = field.attachNewNode("splats")

    def _hud(self) -> None:
        ar = self.getAspectRatio()
        self.bars = {}
        for pid in PIDS:
            # P1's bar grows rightwards from the left screen edge, P2's leftwards
            # from the right edge: grow = +1 / -1 is the sign of the x scale
            grow = 1 if pid == 1 else -1
            x_out = -grow * (ar - 0.08)
            bg = self._card(0, 0.9, -0.03, 0.03, (*self.colors[pid], 1))
            fill = self._card(0, 0.9, -0.03, 0.03, (*self.colors[3 - pid], 1))
            cool = self._card(0, 0.9, -0.008, 0.008, (1, 1, 1, 0.8))
            for n, z in ((bg, 0.88), (fill, 0.88), (cool, 0.83)):
                n.setPos(x_out, 0, z)
                n.setSx(grow)
            label = OnscreenText(text="", pos=(x_out, 0.93), scale=0.05, fg=(1, 1, 1, 1),
                                 align=TextNode.ALeft if pid == 1 else TextNode.ARight, mayChange=True)
            self.bars[pid] = (grow, fill, cool, label)
        self.msg_text = OnscreenText(text="", pos=(0, 0.55), scale=0.12, fg=(1, 0.95, 0.4, 1),
                                     shadow=(0, 0, 0, 0.8), mayChange=True)
        OnscreenText(text="P1: mouse moves, click fires     P2: W/S moves, SPACE fires"
                          "     R: restart   ESC: quit", pos=(0, -0.95), scale=0.045, fg=(0.8, 0.8, 0.8, 1))

    def _card(self, x0: float, x1: float, z0: float, z1: float, color) -> NodePath:
        cm = CardMaker("bar")
        cm.setFrame(x0, x1, z0, z1)
        n = self.aspect2d.attachNewNode(cm.generate())
        n.setColor(*color)
        n.setTransparency(TransparencyAttrib.MAlpha)
        return n

    # --- input ------------------------------------------------------------------

    def _floor_point(self) -> np.ndarray | None:
        """Mouse cursor cast onto the floor (z = 0)."""
        if not self.mouseWatcherNode.hasMouse():
            return None
        near, far = Point3(), Point3()
        self.lens.extrude(self.mouseWatcherNode.getMouse(), near, far)
        hit = Point3()
        floor = Plane(Vec3(0, 0, 1), Point3(0, 0, 0))
        if not floor.intersectsLine(hit, self.render.getRelativePoint(self.cam, near),
                                    self.render.getRelativePoint(self.cam, far)):
            return None
        return np.array([hit.x, hit.y])

    def _key_down(self, key: str) -> bool:
        return self.mouseWatcherNode.isButtonDown(KeyboardButton.asciiKey(key))

    def _request_fire(self, pid: int) -> None:
        self.want_fire[pid] = True

    def _restart(self) -> None:
        self.combat.reset()
        self.mouse.reset(self.arena.start(1))
        self.keys.reset(self.arena.start(2))
        for s in self.splats:
            s.removeNode()
        self.splats.clear()

    # --- frame ------------------------------------------------------------------

    def _tick(self, task):
        t = task.time
        dt = min(ClockObject.getGlobalClock().getDt(), 0.05)  # a stalled frame must not teleport a bolt
        self.combat.move(1, self.mouse.read(t))
        self.combat.move(2, self.keys.read(dt))
        for pid in PIDS:
            self.palms[pid] = self._pose(pid)
            if self.want_fire[pid]:
                self.combat.fire(pid, self.palms[pid])
                self.want_fire[pid] = False
        events = self.combat.update(dt)
        self._effects(events, t)
        self._draw(t)
        return task.cont

    def _pose(self, pid: int) -> np.ndarray:
        """Place the avatar and its firing arm; returns the palm (world)."""
        c, f, av = self.cfg, self.combat.fighters[pid], self.avatars[pid]
        av.set_pos(*f.pos)
        # jab: hand goes out and back once over jab_time (half a sine)
        k = math.sin(math.pi * (1 - f.jab / c["jab_time"])) if f.jab > 0 else 0.0
        fwd, depth, height = lerp(c["hand_ready"], c["hand_jab"], k)
        toward = -self.arena.side(pid)                # +1: opponent is at +x
        target = np.array([f.pos[0] + toward * fwd, f.pos[1] + depth, height])
        side = self.fire_arm[pid]
        pole = av.to_world(Avatar.body_dir(side, c["elbow_pole"]))
        return av.reach(side, target, pole)

    def _effects(self, events: list[Event], t: float) -> None:
        r = self.cfg["bolt_radius"]
        for e in events:
            if e.name == "hit":
                color = self.colors[3 - e.pid]          # the attacker's colour
                self._burst(e.pos, color, t, 0.35, 5 * r)
                self._splat(self.combat.fighters[e.pid].pos, color)
            elif e.name == "fizzle":
                self._burst(e.pos, self.colors[e.pid], t, 0.25, 3 * r)
        self.fx = [fx for fx in self.fx if fx.update(t)]

    def _burst(self, pos: np.ndarray, color, t: float, dur: float, r1: float) -> None:
        node = flat(self.render, color, 0.8)
        node.setPos(*pos)
        self.fx.append(Fx(node, color, t, dur, self.cfg["bolt_radius"], r1))

    def _splat(self, at: np.ndarray, color) -> None:
        """Paint stays on the floor around the victim (design doc 3: the
        area around a player is increasingly covered in the opponent's colour)."""
        off = self.rng.normal(0, 0.35, 2)
        s = flat(self.splat_root, color, 0.55)
        s.setPos(at[0] + off[0], at[1] + off[1], 0.003 + 0.0001 * len(self.splats))
        rad = self.rng.uniform(0.15, 0.4)
        s.setScale(rad * self.rng.uniform(0.8, 1.25), rad, 0.005)
        s.setH(self.rng.uniform(0, 180))
        self.splats.append(s)
        if len(self.splats) > MAX_SPLATS:
            self.splats.pop(0).removeNode()

    def _draw(self, t: float) -> None:
        c, cb = self.cfg, self.combat
        for pid in PIDS:
            f = cb.fighters[pid]
            mix = lerp(self.colors[pid], self.colors[3 - pid], f.damage)
            self.avatars[pid].actor.setColorScale(*lerp((1, 1, 1), mix, c["tint_strength"]), 1)
            self.halves[pid].setColor(*lerp(FLOOR, mix, FLOOR_TINT), 1)
            self.rings[pid].setPos(f.pos[0], f.pos[1], 0.006)
            self.rings[pid].setColor(*mix, 0.55)
            palm = self.palms[pid]
            edge = -self.arena.side(pid) * self.arena.half_length
            self.lanes[pid].setPos(palm[0], palm[1], 0.004)
            self.lanes[pid].setScale(edge - palm[0], 1, 0.04)
            grow, fill, cool, label = self.bars[pid]
            fill.setSx(grow * max(f.damage, 1e-4))     # zero scale would make the matrix singular
            cool.setSx(grow * max(1 - f.cooldown / c["bolt_cooldown"], 1e-4))
            label.setText(f"P{pid}  {round(100 * f.damage)} % covered")

        while len(self.bolt_views) < len(cb.bolts):
            self.bolt_views.append(BoltView(self.render, c["bolt_radius"], c["trail_len"]))
        for i, view in enumerate(self.bolt_views):
            if i < len(cb.bolts):
                b = cb.bolts[i]
                view.show(b.pos, b.trail, self.colors[b.owner])
            else:
                view.hide()

        if cb.state == "countdown":
            msg = str(math.ceil(cb.countdown_left()))
        elif cb.state == "fight":
            msg = "FIGHT!" if cb.t_state < 0.7 else ""
        else:
            msg = f"P{cb.winner} wins!   R to restart"
        self.msg_text.setText(msg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. clash.bolt_speed=12")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)["clash"]
    ClashApp(cfg).run()


if __name__ == "__main__":
    main()
