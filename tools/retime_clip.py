"""Make recorded clips play at their real speed in a video player.

    python tools/retime_clip.py data/clips/clips/*.mp4            # show what would change
    python tools/retime_clip.py data/clips/clips/*.mp4 --apply    # change the clips in place

Clips recorded before the recorder fix (io/recorder.py, ClipRecorder.close)
say 30 fps in the container while the frames came in slower (e.g. 15 fps
from the camera, or the game's own rate while recording), so players show
them sped up. This sets the container rate to the mean rate of the clip's
timestamps sidecar with ffmpeg, without re-encoding: the frames stay
identical, and the pipeline (which replays the sidecar times) gives the
same results as before. Clips without a sidecar are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ipcvgame.io.recorder import retime_clip, sidecar_fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true", help="change the clips (default: only list)")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="leave clips whose container rate is within this fraction of the real one")
    args = ap.parse_args()
    for clip in args.clips:
        real = sidecar_fps(clip)
        cap = cv2.VideoCapture(str(clip))
        old = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if real is None:
            print(f"{clip}: no timestamps sidecar, skipped")
            continue
        if abs(real / old - 1.0) <= args.tolerance:
            print(f"{clip}: {old:.1f} fps, already right")
            continue
        if not args.apply:
            print(f"{clip}: container {old:.1f} fps, real {real:.1f} fps (plays {old / real:.1f}x too fast)")
            continue
        ok = retime_clip(clip, real)
        print(f"{clip}: {old:.1f} -> {real:.1f} fps" if ok else f"{clip}: failed (is ffmpeg installed?)")


if __name__ == "__main__":
    main()
