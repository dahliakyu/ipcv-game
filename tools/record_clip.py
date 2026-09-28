"""Record a raw test clip from the webcam, without running the pipeline.

    python tools/record_clip.py crossing            # -> data/clips/crossing.mp4
    python tools/record_clip.py occlusion --seconds 30

Space starts/stops recording (get into position first), q / Esc quits.
Writes <name>.mp4 plus <name>.timestamps.csv with real capture times, which
VideoFile replays so filters see the true dt. The preview is mirrored like the
game; the clip itself is stored raw (see io/recorder.py).
"""

import argparse
import sys
from pathlib import Path

import cv2

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.io.recorder import ClipRecorder
from ipcvgame.io.source import Webcam
from ipcvgame.scene.draw import draw_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="clip name, e.g. crossing, occlusion, reentry")
    ap.add_argument("--camera", type=int, default=None, help="webcam index (default: config)")
    ap.add_argument("--seconds", type=float, default=None, help="stop automatically after this long")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "clips"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    full = load_config()
    cfg, rec_cfg = full["source"], full["recorder"]
    path = f"{args.out_dir}/{args.name}.mp4"
    if not args.overwrite and Path(path).exists():
        sys.exit(f"{path} exists; use --overwrite")

    cam = args.camera if args.camera is not None else int(cfg.get("uri", 0))
    clip = None
    t_start = None
    with Webcam(cam, cfg["width"], cfg["height"], cfg["fps"], cfg.get("mirror", True)) as src:
        print(f"camera {cam} {src.size[0]}x{src.size[1]}; space = start/stop, q = quit")
        for frame in src:
            if clip is not None:
                clip.write(frame)
                if args.seconds and frame.t - t_start >= args.seconds:
                    break
            preview = frame.image.copy()
            if clip is not None:
                cv2.circle(preview, (24, 24), 10, (0, 0, 255), -1)
                draw_text(preview, f"REC {frame.t - t_start:5.1f}s  {clip.n} frames", (42, 30), (0, 0, 255))
            else:
                draw_text(preview, f"{args.name}: space to start recording", (12, 30))
            cv2.imshow("record clip", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if clip is None:
                    clip = ClipRecorder(path, cfg["fps"], src.size, mirrored=src.mirror,
                                        fourcc=rec_cfg.get("fourcc", "mp4v"))
                    t_start = frame.t
                else:
                    break
    cv2.destroyAllWindows()
    if clip is not None:
        clip.close()
        dur = frame.t - t_start
        print(f"saved {path}: {clip.n} frames, {dur:.1f} s ({clip.n / max(dur, 1e-6):.1f} fps)")
    else:
        print("nothing recorded")


if __name__ == "__main__":
    main()
