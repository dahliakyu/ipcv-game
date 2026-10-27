import numpy as np
import pytest
import torch

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.core.types import Frame, FrameState, PoseDetection
from ipcvgame.identity.osnet import load_weights, osnet_x1_0
from ipcvgame.identity.tracker import IdentityTracker, Tracks

WEIGHTS = REPO_ROOT / load_config()["identity"]["reid_weights"]
needs_weights = pytest.mark.skipif(not WEIGHTS.is_file(), reason="OSNet weights not downloaded (README)")


def person_crop(seed: int, h: int = 200, w: int = 90) -> np.ndarray:
    """A made-up 'person': random blocks of colour, the same for one seed."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 3, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.kron(blocks, np.ones((h // 8, w // 3, 1), np.uint8)))


def test_osnet_gives_512_numbers():
    m = osnet_x1_0(num_classes=10).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 256, 128)).shape == (1, 512)


def test_load_weights_raises_when_nothing_matches(tmp_path):
    path = tmp_path / "other.pth"
    torch.save({"state_dict": {"not_a_layer.weight": torch.zeros(3)}}, path)
    with pytest.raises(RuntimeError):
        load_weights(osnet_x1_0(num_classes=10), str(path))


@needs_weights
def test_extractor_loads_and_compares():
    """Replaces Project/test_osnet.py: loads the real weights (no webcam).
    Vectors have length 1; a crop matches itself best."""
    from ipcvgame.identity.embedding import OSNetEmbeddingExtractor

    ex = OSNetEmbeddingExtractor(WEIGHTS, device="cpu")
    a, a2, b = ex.extract(person_crop(1)), ex.extract(person_crop(1)), ex.extract(person_crop(2))
    assert a.shape == (512,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert a @ a2 == pytest.approx(1.0, abs=1e-5)
    assert a @ b < a @ a2


def test_extractor_missing_weights(tmp_path):
    from ipcvgame.identity.embedding import OSNetEmbeddingExtractor

    with pytest.raises(FileNotFoundError):
        OSNetEmbeddingExtractor(tmp_path / "missing.pth")


# --------------------------------------------------------------------------
# IdentityTracker logic, with a fake OSNet and a scripted tracker.

H, W = 200, 400
BLUE, GREEN, RED = (255, 0, 0), (0, 255, 0), (0, 0, 255)


class FakeOSNet:
    """Vector = the crop's mean colour, length 1: crops of one colour match
    (similarity 1), different pure colours do not (0)."""

    def __init__(self):
        self.calls = 0

    def extract(self, image):
        self.calls += 1
        v = image.reshape(-1, 3).mean(0)
        return v / np.linalg.norm(v)


class ScriptedTracker:
    def __init__(self):
        self.next: Tracks = Tracks([])
        self.resets = 0

    def update(self, dets, frame):
        return self.next

    def reset(self):
        self.resets += 1


def det(x1, x2=None) -> PoseDetection:
    x2 = x1 + 60 if x2 is None else x2
    return PoseDetection(np.array([x1, 20, x2, 180.0]), np.zeros((sk.K, 2)), np.ones(sk.K), 0.9)


def scene(t, people) -> Frame:
    """people: list of (det, colour); each box is filled with its colour."""
    img = np.full((H, W, 3), 1, np.uint8)
    for d, c in people:
        x1, y1, x2, y2 = d.bbox.astype(int)
        img[y1:y2, x1:x2] = c
    return Frame(img, int(t * 100), t)


CFG = {"backend": "reid", "max_players": 2, "reid_threshold": 0.6, "new_player_s": 0.13, "min_face_conf": 0.5,
       "min_crop_h": 40, "min_crop_w": 20, "retry_s": 0.5, "update_every_s": 0.5,
       "update_weight": 0.1, "update_min_sim": 0.5, "update_max_iou": 0.0}


def make(**over):
    tr = ScriptedTracker()
    ident = IdentityTracker({**CFG, **over}, embedder=FakeOSNet(), tracker=tr)
    return ident, tr


def step(ident, tr, t, people, active, lost=None):
    """active: track ids for people, in order; lost: {track_id: box}."""
    dets = [d for d, _ in people]
    tr.next = Tracks(list(zip(active, range(len(dets)), strict=True)), lost or {})
    return ident.update(dets, scene(t, people))


def test_new_player_after_confirmation():
    ident, tr = make()
    a = det(20)
    assert step(ident, tr, 0.0, [(a, BLUE)], [1]) == {}
    assert step(ident, tr, 0.1, [(a, BLUE)], [1]) == {}
    players = step(ident, tr, 0.15, [(a, BLUE)], [1])
    assert players[1].status == "active" and players[1].det is a and not players[1].reentered


def test_linked_track_skips_osnet():
    ident, tr = make(update_every_s=100.0)
    a = det(20)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [1])
    calls = ident.embedding_extractor.calls
    for t in (0.2, 0.25, 0.3):
        assert step(ident, tr, t, [(a, BLUE)], [1])[1].status == "active"
    assert ident.embedding_extractor.calls == calls


def test_occluded_then_lost_then_reentered():
    ident, tr = make()
    a = det(20)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [1])
    predicted = np.array([30, 20, 90, 180.0])
    p = step(ident, tr, 0.2, [], [], lost={1: predicted})[1]
    assert p.status == "occluded" and p.det is None
    np.testing.assert_allclose(p.bbox, predicted)
    p = step(ident, tr, 0.3, [], [])[1]                      # the tracker dropped track 1
    assert p.status == "lost" and not p.reentered
    p = step(ident, tr, 0.4, [(det(200), BLUE)], [7])[1]     # new track, same look
    assert p.status == "active" and p.reentered
    assert not step(ident, tr, 0.45, [(det(200), BLUE)], [7])[1].reentered


def test_same_track_back_from_occluded_is_not_reentered():
    ident, tr = make()
    a = det(20)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [1])
    step(ident, tr, 0.2, [], [], lost={1: a.bbox})
    p = step(ident, tr, 0.25, [(a, BLUE)], [1])[1]
    assert p.status == "active" and not p.reentered


def test_new_track_takes_over_from_held_track():
    """A player found on a new track while the tracker still holds the old
    one (occluded): the link moves, and it is not a re-entry."""
    ident, tr = make()
    a = det(20)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [1])
    step(ident, tr, 0.2, [], [], lost={1: a.bbox})
    p = step(ident, tr, 0.25, [(det(200), BLUE)], [2], lost={1: a.bbox})[1]
    assert p.status == "active" and not p.reentered
    assert ident.track_to_player == {2: 1}


def test_two_players_and_a_bystander_who_is_retried_slowly():
    ident, tr = make(update_every_s=100.0)
    a, b, c = det(20), det(150), det(300)
    people = [(a, BLUE), (b, GREEN)]
    for t in (0.0, 0.15):
        players = step(ident, tr, t, people, [1, 2])
    assert set(players) == {1, 2}
    people.append((c, RED))
    calls = ident.embedding_extractor.calls
    for t in np.arange(0.2, 0.65, 0.05):
        players = step(ident, tr, t, people, [1, 2, 3])
    assert set(players) == {1, 2}                       # max_players: no P3
    assert ident.embedding_extractor.calls - calls == 1   # first try, retry_s not reached yet
    step(ident, tr, 0.75, people, [1, 2, 3])
    assert ident.embedding_extractor.calls - calls == 2


def test_player_in_use_is_not_given_to_another_track():
    """bodypose.py excluded_player_ids: a new track that looks like P1 does
    not get P1 while P1's own track is visible."""
    ident, tr = make()
    a = det(20)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [1])
    players = step(ident, tr, 0.2, [(a, BLUE), (det(200), BLUE)], [1, 5])
    assert players[1].det is a and ident.track_to_player == {1: 1}


def test_small_crop_is_not_identified():
    ident, tr = make()
    tiny = PoseDetection(np.array([20, 20, 30, 50.0]), np.zeros((sk.K, 2)), np.ones(sk.K), 0.9)
    for t in (0.0, 0.15, 0.3):
        assert step(ident, tr, t, [(tiny, BLUE)], [1]) == {}
    assert ident.embedding_extractor.calls == 0


def test_vector_update_blends_and_takes_turns():
    ident, tr = make()
    a, b = det(20), det(150)
    people = [(a, BLUE), (b, GREEN)]
    for t in (0.0, 0.15):
        step(ident, tr, t, people, [1, 2])
    stored = {pid: v.copy() for pid, v in ident.embeddings.items()}
    bluish = (255, 60, 0)                               # P1's look changes a little
    people = [(a, bluish), (b, GREEN)]
    step(ident, tr, 0.7, people, [1, 2])                # P1's turn
    assert not np.allclose(ident.embeddings[1], stored[1])
    np.testing.assert_allclose(ident.embeddings[2], stored[2])
    fresh = FakeOSNet().extract(np.full((4, 4, 3), bluish, np.uint8))
    expected = 0.9 * stored[1] + 0.1 * fresh
    np.testing.assert_allclose(ident.embeddings[1], expected / np.linalg.norm(expected), atol=1e-6)
    step(ident, tr, 0.75, people, [1, 2])               # P2's turn
    assert ident.last_update_t[2] == 0.75


def test_vector_update_skipped_when_overlapping_or_too_different():
    ident, tr = make()
    a = det(20)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [1])
    stored = ident.embeddings[1].copy()
    step(ident, tr, 0.7, [(a, BLUE), (det(60), GREEN)], [1, 9])     # boxes overlap
    np.testing.assert_allclose(ident.embeddings[1], stored)
    assert ident.last_update_t[1] == 0.15
    step(ident, tr, 0.8, [(a, RED)], [1])                            # similarity 0 < 0.5
    np.testing.assert_allclose(ident.embeddings[1], stored)
    assert ident.last_update_t[1] == 0.8


def test_reset():
    ident, tr = make()
    a, b = det(20), det(150)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE), (b, GREEN)], [1, 2])
    ident.reset(1)
    assert set(ident.embeddings) == {2} and 1 not in ident.players
    assert ident._free_id() == 1
    ident.reset()
    assert ident.players == {} and ident.embeddings == {} and tr.resets == 1


def test_unknown_backend():
    with pytest.raises(ValueError):
        IdentityTracker({**CFG, "backend": "nope"})


# --------------------------------------------------------------------------
# BoT-SORT on PoseDetection boxes (identity/tracks.py), synthetic boxes.

def botsort(**over):
    from ipcvgame.identity.tracks import BotSortTracks

    return BotSortTracks({**load_config()["identity"]["botsort"], **over})


def moving(x, score=0.9) -> PoseDetection:
    d = det(x)
    d.score = score
    return d


def test_botsort_keeps_ids_on_moving_boxes():
    bot = botsort()
    frame = scene(0.0, [])
    ids = []
    for k in range(10):
        tr = bot.update([moving(20 + 3 * k), moving(250 - 3 * k)], frame)
        ids.append(sorted(tr.active, key=lambda a: a[1]))
    assert len(ids[-1]) == 2
    assert all(a == ids[-1] for a in ids[2:])           # same id per box once confirmed
    assert ids[-1][0][0] != ids[-1][1][0]


def test_botsort_lost_track_predicted_then_removed():
    bot = botsort(track_buffer=5)
    frame = scene(0.0, [])
    for k in range(6):
        tr = bot.update([moving(20 + 5 * k)], frame)
    (tid, _), = tr.active
    last_x = 20 + 5 * 5
    tr = bot.update([], frame)
    assert tr.active == [] and tid in tr.lost
    assert tr.lost[tid][0] > last_x                    # predicted onwards (constant velocity)
    for _ in range(6):
        tr = bot.update([], frame)
    assert tid not in tr.lost                           # removed after track_buffer frames


def test_botsort_weak_box_continues_but_does_not_start_a_track():
    bot = botsort()
    frame = scene(0.0, [])
    for _ in range(3):
        assert bot.update([moving(300, score=0.4)], frame).active == []
    for k in range(3):
        tr = bot.update([moving(20 + k)], frame)
    (tid, _), = tr.active
    tr = bot.update([moving(23, score=0.4)], frame)
    assert [a[0] for a in tr.active] == [tid]


def test_identity_runs_on_botsort():
    ident = IdentityTracker({**CFG, "botsort": load_config()["identity"]["botsort"]}, embedder=FakeOSNet())
    for k in range(8):
        a, b = moving(20 + 2 * k), moving(250)
        players = ident.update([a, b], scene(k / 30, [(a, BLUE), (b, GREEN)]))
    assert set(players) == {1, 2} and players[1].det is a and players[2].det is b


def test_log_fields_and_debug_view():
    ident, tr = make(update_every_s=100.0)
    a, c = det(20), det(250)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE)], [4])
    players = step(ident, tr, 0.2, [(a, BLUE), (c, RED)], [4, 6])
    state = FrameState(scene(0.2, []), players, {}, {}, [], {})
    row = ident.log_fields(state)
    assert row["identity_p1_track"] == 4
    assert row["identity_p1_sim"] == ""                   # linked track: OSNet did not run for P1
    assert row["identity_pending"] == 1 and row["identity_osnet_runs"] == 1
    assert row["identity_new_best_sim"] == ""             # P1 is in use, so not compared
    canvas = np.zeros((H, W, 3), np.uint8)
    ident.draw_debug(canvas, state)
    assert canvas.any()

    players = step(ident, tr, 0.8, [(c, RED)], [6])      # P1's track dropped: P1 lost
    row = ident.log_fields(FrameState(scene(0.8, []), players, {}, {}, [], {}))
    assert row["identity_p1_track"] == ""
    assert row["identity_new_best_sim"] == pytest.approx(0.0, abs=1e-6)   # red vs P1's blue


def test_no_face_never_identified():
    """A coat: the pose model finds a 'person' without a face."""
    ident, tr = make()
    coat = det(20)
    coat.conf[[sk.NOSE, sk.L_EYE, sk.R_EYE]] = 0.1
    for t in (0.0, 0.15, 0.3, 1.0):
        assert step(ident, tr, t, [(coat, BLUE)], [1]) == {}
    assert ident.embedding_extractor.calls == 0


def test_check_moves_link_to_occluded_player():
    """The tracker carried P1's track onto P2 (P2's own track is held as
    lost): the link moves to P2, P1 is lost, and P1's own person is found
    again on a new track."""
    ident, tr = make()
    a, b = det(20), det(250)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE), (b, GREEN)], [1, 2])
    players = step(ident, tr, 0.7, [(det(250), GREEN), (det(20), BLUE)], [1, 3], lost={2: b.bbox})
    assert players[2].status == "active" and players[2].reentered and players[2].det.bbox[0] == 250
    assert players[1].status in ("active", "lost")
    assert ident.track_to_player[1] == 2
    players = step(ident, tr, 0.75, [(det(250), GREEN), (det(20), BLUE)], [1, 3], lost={2: b.bbox})
    assert players[1].status == "active" and players[1].det.bbox[0] == 20
    assert ident.track_to_player == {1: 2, 3: 1}


def test_check_swaps_two_active_players():
    """Both tracks carried onto the other player (e.g. at a crossing): swap."""
    ident, tr = make()
    a, b = det(20), det(250)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE), (b, GREEN)], [1, 2])
    players = step(ident, tr, 0.7, [(det(20), GREEN), (det(250), BLUE)], [1, 2])
    assert players[1].det.bbox[0] == 250 and players[2].det.bbox[0] == 20
    assert players[1].reentered and players[2].reentered
    assert ident.track_to_player == {1: 2, 2: 1}


def test_check_does_not_swap_on_one_doubtful_crop():
    """P1's crop looks like P2, but P2's crop still looks like P2: no swap."""
    ident, tr = make()
    a, b = det(20), det(250)
    for t in (0.0, 0.15):
        step(ident, tr, t, [(a, BLUE), (b, GREEN)], [1, 2])
    players = step(ident, tr, 0.7, [(det(20), GREEN), (det(250), GREEN)], [1, 2])
    assert ident.track_to_player == {1: 1, 2: 2}
    assert not players[1].reentered and not players[2].reentered
