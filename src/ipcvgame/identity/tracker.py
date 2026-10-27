"""Task 3 — multiplayer identity tracking.

backend "reid" is Colour Clash `PlayerIdentityTracker` (Project/bodypose.py)
with its tracking loop from main.py:

- BoT-SORT (identity/tracks.py) gives every person a short-lived track id
  while it can follow them from frame to frame, on the pose model's boxes.
- A track already linked to a player keeps that player; OSNet does not run.
- A new track's person crop goes through OSNet (identity/embedding.py). Its
  vector is compared with each stored player's vector by cosine similarity
  (`_find_player`); the best player at >= reid_threshold gets the track.
  Players already in use this frame are skipped.
- No match: once the track has been seen for new_player_s it becomes a new
  player, up to max_players. Its vector is stored.

Changed from bodypose.py (merge_log, Task 3):
- Occluded, then lost. While the tracker still holds a player's track (not
  seen for at most its track_buffer), the player is "occluded" with the
  tracker's predicted box, and keeps the link. Only when the tracker drops the
  track is the player "lost"; when OSNet links a new track to them,
  `reentered` is True for that frame. bodypose.py dropped the link on every
  frame without the track.
- Stored vectors are updated: one active player per frame, in turn, at most
  every update_every_s, when their box overlaps no other detection and the
  fresh vector is still similar (update_min_sim). The stored vector moves
  update_weight of the way towards it.
- Linked tracks are handled before new ones, so a new track cannot take a
  player whose own track is visible this frame.
- A track that found no player while every player slot is taken retries only
  every retry_s, so a bystander does not cost an OSNet run every frame.
- New players take the lowest free id, so reset(pid) frees that id.
- Seconds instead of frames (NEW_PLAYER_CONFIRMATION = 5 frames).

backend "zones" is the phase-1 stub: Player 1 = best detection in the left
half of the screen, Player 2 = best in the right half; no model needed (tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ipcvgame.core.geometry import bbox_center, bbox_iou, crop
from ipcvgame.core.types import Frame, FrameState, Player, PoseDetection
from ipcvgame.scene.draw import draw_bbox, draw_text, player_color


@dataclass
class Tracks:
    """One frame of tracker output."""
    active: list[tuple[int, int]]                  # (track_id, index into dets), seen this frame
    lost: dict[int, np.ndarray] = field(default_factory=dict)   # track_id -> predicted box, still held


@dataclass
class _Pending:
    """A track not linked to a player yet (bodypose.py: pending_tracks)."""
    first_t: float
    last_try_t: float
    embedding: np.ndarray | None = None


class IdentityTracker:
    def __init__(self, cfg: dict, embedder=None, tracker=None):
        """embedder (OSNet) and tracker (BoT-SORT) are built from the config;
        tests pass fakes with the same methods."""
        self.cfg = cfg
        self.backend = cfg.get("backend", "reid")
        self.max_players = int(cfg["max_players"])
        self.players: dict[int, Player] = {}
        if self.backend == "reid":
            if embedder is None:
                from ipcvgame.identity.embedding import OSNetEmbeddingExtractor
                embedder = OSNetEmbeddingExtractor(cfg["reid_weights"], cfg.get("reid_device", "auto"))
                # The first run sets up the GPU (~85 ms); do it now, not when
                # the first player walks in.
                embedder.extract(np.zeros((256, 128, 3), np.uint8))
            self.embedding_extractor = embedder
            if tracker is None:
                from ipcvgame.identity.tracks import BotSortTracks
                tracker = BotSortTracks(cfg["botsort"])
            self.tracker = tracker
        elif self.backend != "zones":
            raise ValueError(f"Unknown identity backend {self.backend!r}")
        self._clear()

    def _clear(self) -> None:
        self.players.clear()
        self.embeddings: dict[int, np.ndarray] = {}        # bodypose.py: players
        self.track_to_player: dict[int, int] = {}
        self.pending_tracks: dict[int, _Pending] = {}
        self.last_update_t: dict[int, float] = {}           # vector update, per player
        self._update_turn = 0
        self._tracks = Tracks([])
        self._dets: list[PoseDetection] = []
        self._sim: dict[int, float] = {}                    # this frame's OSNet similarity per pid
        self._new_best_sim: float | None = None             # best similarity of a new track this frame
        self._osnet_runs = 0                                # OSNet crops this frame

    # ------------------------------------------------------------------ reid

    def update(self, dets: list[PoseDetection], frame: Frame) -> dict[int, Player]:
        if self.backend == "zones":
            return self._update_zones(dets, frame)
        self._dets = dets
        self._sim = {}
        self._new_best_sim = None
        self._osnet_runs = 0
        tracks = self.tracker.update(dets, frame)
        self._tracks = tracks
        self.forget_missing_tracks(tracks)

        seen: dict[int, PoseDetection] = {}
        new_tracks = []
        for track_id, i in tracks.active:
            pid = self.track_to_player.get(track_id)
            if pid is None or pid in seen:
                new_tracks.append((track_id, i))
            else:
                seen[pid] = dets[i]
        for track_id, i in new_tracks:
            pid = self._identify(track_id, dets[i], frame, excluded=set(seen))
            if pid is not None:
                seen[pid] = dets[i]

        occluded = {pid: tracks.lost[tid] for tid, pid in self.track_to_player.items()
                    if tid in tracks.lost and pid not in seen}
        for pid in sorted(set(self.players) | set(seen)):
            prev = self.players.get(pid)
            if pid in seen:
                det = seen[pid]
                self.players[pid] = Player(
                    pid=pid, status="active", det=det, bbox=det.bbox.copy(),
                    pos=np.zeros(2),  # (x, depth) in metres not estimated yet
                    reentered=prev is not None and prev.status == "lost",
                )
            else:
                status = "occluded" if pid in occluded else "lost"
                bbox = occluded[pid].copy() if pid in occluded else prev.bbox.copy()
                self.players[pid] = Player(pid=pid, status=status, det=None, bbox=bbox,
                                           pos=prev.pos.copy(), reentered=False)

        self._update_embedding(frame)
        return dict(self.players)

    def forget_missing_tracks(self, tracks: Tracks) -> None:
        """Drop links and pending tracks the tracker no longer holds. Pending
        tracks must be seen every frame to count towards new_player_s."""
        held = {tid for tid, _ in tracks.active} | set(tracks.lost)
        self.track_to_player = {tid: pid for tid, pid in self.track_to_player.items() if tid in held}
        visible = {tid for tid, _ in tracks.active}
        self.pending_tracks = {tid: p for tid, p in self.pending_tracks.items() if tid in visible}

    def _crop(self, frame: Frame, det: PoseDetection) -> np.ndarray | None:
        """Person crop inside the frame, or None when it is too small for
        OSNet (bodypose.py MIN_PERSON_HEIGHT / MIN_PERSON_WIDTH)."""
        image, _ = crop(frame.image, det.bbox)
        if image.shape[0] < self.cfg["min_crop_h"] or image.shape[1] < self.cfg["min_crop_w"]:
            return None
        return image

    def _embed(self, person: np.ndarray) -> np.ndarray:
        self._osnet_runs += 1
        return self.embedding_extractor.extract(person)

    def _find_player(self, embedding: np.ndarray, excluded_player_ids: set[int]) -> int | None:
        best_player, best_similarity = None, -1.0
        for player_id, player_embedding in self.embeddings.items():
            if player_id in excluded_player_ids:
                continue
            similarity = float(np.dot(embedding, player_embedding))
            if similarity > best_similarity:
                best_player, best_similarity = player_id, similarity
        if best_player is not None:
            self._new_best_sim = max(best_similarity, self._new_best_sim or -1.0)
        if best_player is not None and best_similarity >= self.cfg["reid_threshold"]:
            self._sim[best_player] = best_similarity
            return best_player
        return None

    def _free_id(self) -> int | None:
        for pid in range(1, self.max_players + 1):
            if pid not in self.embeddings:
                return pid
        return None

    def _identify(self, track_id: int, det: PoseDetection, frame: Frame, excluded: set[int]) -> int | None:
        """bodypose.py PlayerIdentityTracker.update() for a track without a player."""
        t = frame.t
        pending = self.pending_tracks.get(track_id)
        slots_full = self._free_id() is None
        if pending is not None and slots_full and t - pending.last_try_t < self.cfg["retry_s"]:
            return None
        person = self._crop(frame, det)
        if person is None:
            return None
        embedding = self._embed(person)
        matched_player = self._find_player(embedding, excluded)
        if matched_player is not None:
            # The player may still be linked to an older track the tracker holds as lost.
            self.track_to_player = {tid: pid for tid, pid in self.track_to_player.items()
                                    if pid != matched_player}
            self.track_to_player[track_id] = matched_player
            self.pending_tracks.pop(track_id, None)
            return matched_player

        if pending is None:
            pending = self.pending_tracks[track_id] = _Pending(first_t=t, last_try_t=t)
        pending.last_try_t = t
        pending.embedding = embedding
        if t - pending.first_t < self.cfg["new_player_s"] or slots_full:
            return None

        player_id = self._free_id()
        self.pending_tracks.pop(track_id)
        self.embeddings[player_id] = pending.embedding.copy()
        self.track_to_player[track_id] = player_id
        self.last_update_t[player_id] = t
        return player_id

    def _update_embedding(self, frame: Frame) -> None:
        """Blend one active player's stored vector with a fresh one (new).
        Players take turns, so OSNet adds at most one crop per frame."""
        active = sorted(pid for pid, p in self.players.items() if p.status == "active")
        if not active:
            return
        for k in range(len(active)):
            pid = active[(self._update_turn + k) % len(active)]
            if frame.t - self.last_update_t.get(pid, -np.inf) < self.cfg["update_every_s"]:
                continue
            det = self.players[pid].det
            others = [d for d in self._dets if d is not det]
            if any(bbox_iou(det.bbox, d.bbox) > self.cfg["update_max_iou"] for d in others):
                continue        # another person in the crop would blend into the vector
            person = self._crop(frame, det)
            if person is None:
                continue
            self._update_turn = (self._update_turn + k + 1) % len(active)
            self.last_update_t[pid] = frame.t
            fresh = self._embed(person)
            sim = float(fresh @ self.embeddings[pid])
            self._sim[pid] = sim
            if sim < self.cfg["update_min_sim"]:
                return          # too different: maybe not this player after all
            w = self.cfg["update_weight"]
            v = (1 - w) * self.embeddings[pid] + w * fresh
            self.embeddings[pid] = v / np.linalg.norm(v)
            return

    # ----------------------------------------------------------------- zones

    def _update_zones(self, dets: list[PoseDetection], frame: Frame) -> dict[int, Player]:
        width = frame.image.shape[1]
        zone_w = width / self.max_players
        best: dict[int, PoseDetection] = {}
        for d in dets:
            zone = min(int(bbox_center(d.bbox)[0] // zone_w), self.max_players - 1)
            pid = zone + 1
            if pid not in best or d.score > best[pid].score:
                best[pid] = d

        for pid in range(1, self.max_players + 1):
            prev = self.players.get(pid)
            det = best.get(pid)
            if det is not None:
                self.players[pid] = Player(
                    pid=pid, status="active", det=det, bbox=det.bbox.copy(),
                    pos=np.zeros(2),  # stub: (x, depth) in metres not estimated yet
                    reentered=prev is not None and prev.status == "lost",
                )
            elif prev is not None:
                self.players[pid] = Player(
                    pid=pid, status="lost", det=None, bbox=prev.bbox.copy(),
                    pos=prev.pos.copy(), reentered=False,
                )
        return dict(self.players)

    # ------------------------------------------------------------- interface

    def reset(self, pid: int | None = None) -> None:
        """None: forget every player and track (a new round re-identifies
        everyone). A pid: forget that player's vector and links; the id is
        free for the next new player."""
        if pid is None:
            self._clear()
            if self.backend == "reid" and self.tracker is not None:
                self.tracker.reset()
            return
        self.players.pop(pid, None)
        self.embeddings.pop(pid, None)
        self.last_update_t.pop(pid, None)
        self.track_to_player = {tid: p for tid, p in self.track_to_player.items() if p != pid}

    def _track_of(self) -> dict[int, int]:
        return {pid: tid for tid, pid in self.track_to_player.items()}

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Player boxes (dashed when not active) with status, BoT-SORT track
        id and this frame's OSNet similarity; green "Identifying..." boxes for
        tracks without a player (main.py's label); zones: the split line."""
        if self.backend == "zones":
            w = canvas.shape[1]
            for z in range(1, self.max_players):
                x = int(w * z / self.max_players)
                canvas[:, max(0, x - 1):x + 1] = (255, 255, 255)
        track_of = self._track_of() if self.backend == "reid" else {}
        for p in state.players.values():
            label = f"P{p.pid} {p.status}"
            if p.pid in track_of:
                label += f" t{track_of[p.pid]}"
            if p.pid in self._sim:
                label += f" sim {self._sim[p.pid]:.2f}"
            if p.reentered:
                label += " REENTER"
            draw_bbox(canvas, p.bbox, player_color(p.pid), label, dashed=p.status != "active")
        if self.backend == "reid":
            for tid, i in self._tracks.active:
                if tid not in self.track_to_player:
                    draw_bbox(canvas, self._dets[i].bbox, (0, 255, 0), f"Identifying... t{tid}")
            draw_text(canvas, f"stored players: {sorted(self.embeddings)}  OSNet runs: {self._osnet_runs}",
                      (12, canvas.shape[0] - 12))

    def log_fields(self, state: FrameState) -> dict:
        """CSV columns (status, box and reentered are in the base log):
        identity_p<pid>_track: BoT-SORT track id linked to the player, empty
        when none; identity_p<pid>_sim: OSNet similarity this frame (a match
        or a vector update), empty when OSNet did not run for the player;
        identity_new_best_sim: best similarity of a track without a player to
        any stored player (shows how close a returning player came to
        reid_threshold); identity_pending: tracks without a player;
        identity_osnet_runs: OSNet crops this frame (cost)."""
        if self.backend != "reid":
            return {}
        track_of = self._track_of()
        row: dict = {}
        for pid in sorted(self.players):
            row[f"identity_p{pid}_track"] = track_of.get(pid, "")
            row[f"identity_p{pid}_sim"] = self._sim.get(pid, "")
        row["identity_new_best_sim"] = "" if self._new_best_sim is None else self._new_best_sim
        row["identity_pending"] = sum(tid not in self.track_to_player for tid, _ in self._tracks.active)
        row["identity_osnet_runs"] = self._osnet_runs
        return row
