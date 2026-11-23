"""Task 3 — OSNet appearance vectors for person crops.

From Colour Clash `Project/osnet_embedding.py` (OSNetEmbeddingExtractor),
same steps: the crop goes from BGR to RGB, is resized to OSNet's 256 x 128
input and normalised with the ImageNet mean and std; the network returns a
512-number vector, scaled to length 1 so the dot product of two vectors is
their cosine similarity.

Changed: the device and the weights path come from the identity config
instead of being fixed in the code, and nothing is printed on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from ipcvgame.core.config import REPO_ROOT
from ipcvgame.identity.osnet import load_weights, osnet_x1_0

# The weights were trained on MSMT17 with all of its 4,101 identities
# ("combineall"). The classifier is not used at inference; the size only has
# to match so load_weights() keeps every other layer.
MSMT17_IDS = 4101


class OSNetEmbeddingExtractor:
    def __init__(self, model_path: str | Path, device: str = "auto"):
        path = Path(model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"OSNet weights not found: {path} (see README for the download)")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = osnet_x1_0(num_classes=MSMT17_IDS)
        load_weights(self.model, str(path))
        self.model.eval()
        self.model.to(self.device)

        # OSNet uses 256 x 128 input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image: np.ndarray) -> np.ndarray:
        """BGR person crop -> (512,) vector of length 1."""
        image = image[:, :, ::-1].copy()        # OpenCV BGR -> RGB
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(x)
        embedding = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
