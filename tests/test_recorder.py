import csv

import cv2
import numpy as np

from ipcvgame.core.types import Frame, FrameState, Player
from ipcvgame.io.recorder import ClipRecorder, RunLogger, timestamps_path
from ipcvgame.io.source import VideoFile

W, H = 320, 240


def gradient_frame(i: int, t: float) -> Frame:
    # Left half dark, right half bright: mirroring is easy to detect.
    img = np.zeros((H, W, 3), np.uint8)
    img[:, W // 2:] = 200
    return Frame(img, i, t)


def test_clip_roundtrip_mirror_and_timestamps(tmp_path):
    path = tmp_path / "clip.mp4"
    times = [10.0, 10.03, 10.07, 10.2, 10.25]  # irregular, like a real webcam
    with ClipRecorder(path, 30, (W, H), mirrored=True) as rec:
        for i, t in enumerate(times):
            rec.write(gradient_frame(i, t))
    assert timestamps_path(path).exists()

    # Stored raw: the bright half is on the LEFT in the file.
    cap = cv2.VideoCapture(str(path))
    ok, raw = cap.read()
    cap.release()
    assert ok and raw[:, : W // 2].mean() > raw[:, W // 2:].mean()

    # Replayed with mirror=True: identical to what was recorded, true dt kept.
    frames = list(VideoFile(path, mirror=True, video_time="file"))
    assert len(frames) == len(times)
    assert frames[0].image[:, W // 2:].mean() > frames[0].image[:, : W // 2].mean()
    dts = np.diff([f.t for f in frames])
    np.testing.assert_allclose(dts, np.diff(times), atol=1e-5)


def player(pid: int, status: str = "active") -> Player:
    return Player(pid, status, None, np.array([0, 0, 10, 20.0]), np.zeros(2), False)


def state(i: int, players: dict) -> FrameState:
    return FrameState(Frame(np.zeros((1, 1, 3), np.uint8), i, 5.0 + i / 30), players, {}, {}, [],
                      {"pose": 10.0, "total": 11.0})


def test_run_logger_late_columns_and_flush(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(path, flush_every=2) as log:
        log.log(state(0, {1: player(1)}))
        log.log(state(1, {1: player(1)}))            # flush: appended rows
        log.log(state(2, {1: player(1), 2: player(2)}), {"face_p1_roll": 0.5})  # new columns
        log.log(state(3, {1: player(1, "lost")}))    # flush: rewrite with new header
        log.log(state(4, {}))                        # written on close
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5
    assert rows[0]["t"] == "0.000000" and rows[0]["ms_pose"] == "10.000"
    assert rows[0]["p2_status"] == "" and rows[2]["p2_status"] == "active"
    assert rows[2]["face_p1_roll"] == "0.500"
    assert rows[3]["p1_status"] == "lost"
    assert open(path).read().count("frame_id") == 1


def test_run_logger_column_order(tmp_path):
    path = tmp_path / "run.csv"
    with RunLogger(path) as log:
        log.log(state(0, {2: player(2)}), {"face_p2_roll": 0.1})
        log.log(state(1, {1: player(1), 2: player(2)}))
    header = open(path).readline().strip().split(",")
    assert header[:2] == ["frame_id", "t"]
    assert header.index("ms_pose") < header.index("ms_total") < header.index("p1_status")
    assert header.index("p1_score") < header.index("p2_status")
    assert header[-1] == "face_p2_roll"
