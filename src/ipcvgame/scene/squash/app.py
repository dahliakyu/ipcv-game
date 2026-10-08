"""Squash game-play prototype: a Panda3D window with the rigged avatar.

    python -m ipcvgame.scene.squash.app [--set squash.serve_speed=6]

Mouse moves the avatar's right hand (and the avatar follows sideways); touch
the incoming ball to return it to the front wall. SPACE/click serves, R
resets, ESC quits. Rendering only: rules live in rally.py, input in
control.py, the arm IK in avatar.py.
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
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    TextNode,
    TransparencyAttrib,
    loadPrcFileData,
)

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.core.filters import EMA
from ipcvgame.scene.squash.avatar import Avatar
from ipcvgame.scene.squash.control import Control, MouseControl
from ipcvgame.scene.squash.rally import Rally

WALL = (0.86, 0.87, 0.85, 1)
FLOOR = (0.80, 0.66, 0.46, 1)
LINE = (0.75, 0.10, 0.10, 1)


def make_sphere(radius: float = 1.0, rings: int = 12, segments: int = 20) -> GeomNode:
    """UV sphere (Panda3D ships no sphere primitive)."""
    vdata = GeomVertexData("sphere", GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vw, nw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "normal")
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments + 1):
            th = 2 * math.pi * j / segments
            n = (math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), math.cos(phi))
            nw.addData3(*n)
            vw.addData3(*(radius * c for c in n))
    tris = GeomTriangles(Geom.UHStatic)
    for i in range(rings):
        for j in range(segments):
            a, b = i * (segments + 1) + j, (i + 1) * (segments + 1) + j
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("sphere")
    node.addGeom(geom)
    return node


def rect(parent: NodePath, x0: float, x1: float, z0: float, z1: float, color, pos, hpr=(0, 0, 0)) -> NodePath:
    """Flat rectangle in its own XZ plane, then placed with pos/hpr."""
    cm = CardMaker("rect")
    cm.setFrame(x0, x1, z0, z1)
    np_ = parent.attachNewNode(cm.generate())
    np_.setPos(*pos)
    np_.setHpr(*hpr)
    np_.setColor(*color)
    np_.setTwoSided(True)
    return np_


class SquashApp(ShowBase):
    def __init__(self, cfg: dict):
        w, h = cfg["window_size"]
        loadPrcFileData("", f"win-size {w} {h}\nwindow-title IPCV squash prototype\n"
                            "framebuffer-multisample 1\nmultisamples 4\nsync-video 1")
        super().__init__()
        self.cfg = cfg
        self.disableMouse()
        self.setBackgroundColor(0.12, 0.13, 0.16, 1)
        self.render.setShaderAuto()
        self._lights()
        self._court()

        model = Filename.fromOsSpecific(str(REPO_ROOT / cfg["model"]))
        self.avatar = Avatar(self.render, model, cfg)

        r = cfg["ball_radius"]
        self.ball_np = self.render.attachNewNode(make_sphere(r))
        self.ball_np.setColor(0.05, 0.05, 0.05, 1)
        # faint copy drawn last without depth test: the ball stays visible
        # when it passes behind the avatar
        self.ghost_np = self.render.attachNewNode(make_sphere(r))
        self.ghost_np.setColor(0.05, 0.05, 0.05, 0.35)
        self.ghost_np.setTransparency(TransparencyAttrib.MAlpha)
        self.ghost_np.setDepthTest(False)
        self.ghost_np.setDepthWrite(False)
        self.ghost_np.setBin("fixed", 10)
        self.ghost_np.setLightOff()
        # flattened dark disc on the floor: the main depth cue for the player
        self.shadow_np = self.render.attachNewNode(make_sphere(r))
        self.shadow_np.setScale(1.3, 1.3, 0.02)
        self.shadow_np.setColor(0, 0, 0, 0.45)
        self.shadow_np.setTransparency(TransparencyAttrib.MAlpha)
        self.shadow_np.setLightOff()
        # hit zone around the palm, so a mouse player can see the reach
        self.zone_np = self.render.attachNewNode(make_sphere(cfg["hit_radius"]))
        self.zone_np.setTransparency(TransparencyAttrib.MAlpha)
        self.zone_np.setLightOff()
        self.zone_np.setDepthWrite(False)

        self.rally = Rally(cfg)
        self.control = MouseControl(cfg, self._mouse_xy)
        self.hand_vel = EMA(tau=cfg["hand_vel_tau"])
        self.palm: np.ndarray | None = None

        self.score_text = OnscreenText(text="", pos=(-1.7, 0.9), scale=0.06, fg=(1, 1, 1, 1),
                                       align=TextNode.ALeft, mayChange=True)
        self.msg_text = OnscreenText(text="", pos=(0, 0.75), scale=0.08, fg=(1, 0.9, 0.3, 1),
                                     mayChange=True)
        OnscreenText(text="mouse: hand   SPACE/click: serve   R: reset   ESC: quit",
                     pos=(0, -0.95), scale=0.045, fg=(0.8, 0.8, 0.8, 1))

        self.camera.setPos(*cfg["camera_pos"])
        self.camera.lookAt(*cfg["camera_look_at"])
        self.accept("escape", self.userExit)
        self.accept("space", self.rally.serve)
        self.accept("mouse1", self.rally.serve)
        self.accept("r", self.rally.reset)
        self.taskMgr.add(self._tick, "squash-tick")

    # --- scene ------------------------------------------------------------------

    def _lights(self) -> None:
        sun = self.render.attachNewNode(DirectionalLight("sun"))
        sun.setHpr(20, -50, 0)
        self.render.setLight(sun)
        amb = self.render.attachNewNode(AmbientLight("amb"))
        amb.node().setColor((0.45, 0.45, 0.48, 1))
        self.render.setLight(amb)

    def _court(self) -> None:
        c = self.cfg
        hw, wy, by, wh = c["court_width"] / 2, c["front_wall_y"], c["back_wall_y"], c["wall_height"]
        court = self.render.attachNewNode("court")
        rect(court, -hw, hw, by, wy, FLOOR, (0, 0, 0), (0, -90, 0))     # floor
        rect(court, -hw, hw, 0, wh, WALL, (0, wy, 0))                   # front wall
        for x in (-hw, hw):                                             # side walls
            side = rect(court, by, wy, 0, wh, WALL, (x, 0, 0), (90, 0, 0))
            side.setColorScale(0.92, 0.92, 0.92, 1)
        # front wall markings, just in front of the wall to avoid z-fighting
        y = wy - 0.005
        rect(court, -hw, hw, 0, c["tin_height"], (0.55, 0.55, 0.58, 1), (0, y, 0))
        for z in (c["tin_height"], c["service_line"], c["out_line"]):
            rect(court, -hw, hw, z - 0.025, z + 0.025, LINE, (0, y - 0.001, 0))
        # floor: short line through the T and the half-court line behind it
        rect(court, -hw, hw, -0.025, 0.025, LINE, (0, 0, 0.003), (0, -90, 0))
        rect(court, -0.025, 0.025, by, 0, LINE, (0, 0, 0.003), (0, -90, 0))

    # --- input ------------------------------------------------------------------

    def _mouse_xy(self) -> tuple[float, float] | None:
        if not self.mouseWatcherNode.hasMouse():
            return None
        m = self.mouseWatcherNode.getMouse()
        return float(m.getX()), float(m.getY())

    def hand_target(self, ctrl: Control) -> tuple[float, np.ndarray]:
        """Normalised Control -> body x and hand target in world metres."""
        c = self.cfg
        body_x = ctrl.body_x * c["body_x_max"]
        z0, z1 = c["hand_z_range"]
        u, v = ctrl.hand
        target = np.array([body_x + u * c["hand_reach_x"], c["hit_y"], z0 + (v + 1) / 2 * (z1 - z0)])
        return body_x, target

    # --- frame ------------------------------------------------------------------

    def _tick(self, task):
        t = task.time
        dt = min(ClockObject.getGlobalClock().getDt(), 0.05)  # a stalled frame must not teleport the ball
        ctrl = self.control.read(t)
        if ctrl is not None:
            body_x, target = self.hand_target(ctrl)
            self.avatar.set_x(body_x)
            palm = self.avatar.reach_right(target, np.array(self.cfg["elbow_pole"], dtype=float))
        else:
            palm = self.palm if self.palm is not None else np.array([0.3, self.cfg["hit_y"], 1.3])
        prev = self.palm if self.palm is not None else palm
        vel = self.hand_vel((palm - prev) / max(dt, 1e-3), t)
        self.palm = palm

        events = self.rally.update(dt, prev, palm, vel)
        self._draw(events)
        return task.cont

    def _draw(self, events: list[str]) -> None:
        b = self.rally.ball.pos
        self.ball_np.setPos(*b)
        self.ghost_np.setPos(*b)
        self.shadow_np.setPos(b[0], b[1], 0.004)
        live = self.rally.state in ("incoming", "outgoing")
        for n in (self.ball_np, self.ghost_np, self.shadow_np):
            if live:
                n.show()
            else:
                n.hide()

        self.zone_np.setPos(*self.palm)
        near = self.rally.state == "incoming" and np.linalg.norm(b - self.palm) < 2 * self.cfg["hit_radius"]
        if "hit" in events:
            self.zone_np.setColor(1.0, 0.8, 0.2, 0.6)
        elif near:
            self.zone_np.setColor(0.3, 1.0, 0.4, 0.35)
        else:
            self.zone_np.setColor(1, 1, 1, 0.12)

        r = self.rally
        self.score_text.setText(f"rally {r.count}    best {r.best}")
        self.msg_text.setText(r.message)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. squash.serve_speed=6")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)["squash"]
    SquashApp(cfg).run()


if __name__ == "__main__":
    main()
