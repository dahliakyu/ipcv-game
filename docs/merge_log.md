Oct 25, 2026 · @el created with help of Claude Opus 5.5

# Merge log: Colour Clash prototype into the shared codebase

This log records every change made while merging the Colour Clash prototype (the `Colour Clash/` folder) into the shared codebase. The merge is done by the Task 5 owner on branch `face-tracking-eline` and goes to `main` in a single PR when it is complete.

Each entry names the files it touched, so owners can check what changed in their own code.

## 2026-10-25 · Clean-up of the `Colour Clash/` folder

The folder arrived as a Windows zip of 1.4 GB and ~33,800 files; after the clean-up it holds 24 MB and 15 files. No source code was changed in this step.

**Permissions.** Every directory came out of the zip read-only (`dr-xr-xr-x`), so nothing in it could be deleted. All directories were made writable for their owner (`chmod u+w`).

**Deleted:**

| Path | What it was | Why it can go |
| --- | --- | --- |
| `Colour Clash/Project/Lib/`, `Scripts/`, `Include/`, `share/`, `pyvenv.cfg` | A Windows Python 3.11 virtual environment (torch, scipy, ultralytics, …) | Packages come from `requirements.txt` and the `ipcvgame` conda env; ~99% of the folder's size |
| `Colour Clash/Project/personreid/` | A git clone of [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) (torchreid) | Unmodified upstream code; the 154 "modified" files only had Windows line endings |
| `Colour Clash/__pycache__/`, `Colour Clash/Project/__pycache__/` | Python bytecode caches | Regenerated automatically |
| `Colour Clash/.vscode/`, `Colour Clash/Project/.vscode/` | Personal editor settings | Not shared project config |
| `Colour Clash/Project/Project description.pdf` | The project brief | Byte-identical to `docs/phase-1/IPCV_2026-1A_Project_Description_v1-1.pdf` |
| `Colour Clash/yolo26n-pose.pt` | Pose model weights | Byte-identical to `Colour Clash/Project/yolo26n-pose.pt`, which is the copy `main.py` loads |

**Kept:** all `.py` files, `botsort.yaml`, `notes.txt`, the palette `.jpg` (loaded by `renderer.py`), and the three model weight files: `Project/yolo26n-pose.pt`, `Project/models/yolo11n.pt` and `Project/models/osnet_x1_0_msmt17.pth`. The weights are ignored by git (`*.pt`, `models/`) and must stay out of the repo.

## 2026-10-25 · New dependency: `lap`

**Changed:** `requirements.txt`

- Added `lap==0.5.13`. Ultralytics' BoT-SORT tracker (`person_detector.track(...)` in `Colour Clash/main.py`) imports it, and if it is missing Ultralytics pip-installs it silently at runtime.
- Every other import in the prototype (`ultralytics`, `torch`, `torchvision`, `cv2`, `numpy`) was already pinned.

**Action for everyone:** run `pip install -r requirements.txt` in the `ipcvgame` env.

## 2026-10-25 · OSNet copied into the code; `torchreid` dropped

The identity code (`PlayerIdentityTracker` in `Colour Clash/Project/bodypose.py`) uses OSNet to recognise a player who comes back after the tracker lost them. It crops the person, turns the crop into a 512-number appearance vector, and compares that vector with each known player's (cosine similarity, threshold 0.6). OSNet came from the `torchreid` package, but the code used only two of its functions: building the network and loading the weights. Importing `torchreid` also needs `gdown` and `tensorboard`. The one file that defines the network is now copied into the project, so none of the three packages is needed.

**Added:** `Colour Clash/Project/osnet.py`

- Copied from `torchreid/models/osnet.py` at upstream commit `f8cd150f` (MIT licence; the licence text and source are in the file header).
- Trimmed to `osnet_x1_0` and the layers it uses. The other model sizes and the Google Drive weight download were removed.
- New `load_weights(model, path)` replaces `torchreid.utils.load_pretrained_weights`. It loads with `torch.load(..., weights_only=True)` and raises an error if no layer matches. The upstream version only printed a warning and kept running with random weights.
- Layer names are unchanged, so `osnet_x1_0_msmt17.pth` loads as before. Only the classifier layer is skipped, as before; it isn't used at inference.

**Changed:** `Colour Clash/Project/osnet_embedding.py`

- `import torchreid` replaced by `from osnet import load_weights, osnet_x1_0`, with the same `if __package__:` pattern as `bodypose.py`. Both `python main.py` (from `Colour Clash/`) and running scripts inside `Project/` still work.
- `torchreid.models.build_model(name="osnet_x1_0", num_classes=4101, pretrained=False)` became `osnet_x1_0(num_classes=4101)`.
- `torchreid.utils.load_pretrained_weights(self.model, model_path)` became `load_weights(self.model, model_path)`.
- Nothing else in the file changed; `OSNetEmbeddingExtractor` behaves as before.

**Changed:** `requirements.txt`: no `torchreid`, `gdown` or `tensorboard` entries.

**Checked:**

- On the same input with the same weights, the copied network gives the same output as upstream torchreid (max difference 0.0).
- `OSNetEmbeddingExtractor` returns a unit-length 512-d vector on CUDA.
- `Colour Clash/main.py` imports without `torchreid`.
- `ruff` finds no errors in the two files.

**Action for anyone who installed it:** `torchreid`, `gdown` and `tensorboard` are no longer needed and can be removed with `pip uninstall torchreid gdown tensorboard tensorboard-data-server`.

## 2026-10-25 · Prototype moved to `garb/rest/` and ignored by git

**Moved:** the contents of `Colour Clash/` now live in `garb/rest/`, and `Colour Clash/` no longer exists. The earlier entries in this log use the old `Colour Clash/...` paths, so `Colour Clash/Project/osnet.py` is now `garb/rest/Project/osnet.py`, and so on.

**Changed:** `.gitignore`

- Added `garb/`, so the raw prototype stays out of commits while its code is being moved into `src/ipcvgame/`. This is temporary: the entry gets removed once the merge is done.
- Removed the old `Colour Clash/Project/Lib` entry, because that folder was deleted in the clean-up.

## Open issues found during the merge

These are not fixed yet; each belongs to the owner of the code.

- **Two detectors per frame (Task 3, Task 2).** `Colour Clash/main.py` runs `yolo11n.pt` as a separate person detector for tracking and then `yolo26n-pose.pt` for keypoints. The architecture rule is one pose model per frame. YOLO-pose already returns person boxes, so tracking can run on those and `yolo11n.pt` can go.
- **Stored appearance is never updated (Task 3).** In `PlayerIdentityTracker`, each player's OSNet vector is stored once, when the player is created. Lighting changes or a jacket coming off can then break re-identification. Matching is also greedy per track, not a joint assignment (Hungarian).
- **`Colour Clash/Project/test_osnet.py` fails as written.** It calls `OSNetEmbeddingExtractor()` without the required `model_path` argument, and it opens the webcam.
- **Working directory.** `Colour Clash/` has a space in its name and `Project/` has no `__init__.py`, so `main.py` only runs with `Colour Clash/` as the working directory. This goes away when the code moves into `src/ipcvgame/`.
