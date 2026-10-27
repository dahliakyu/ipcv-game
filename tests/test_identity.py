import numpy as np
import pytest
import torch

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.identity.osnet import load_weights, osnet_x1_0

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
