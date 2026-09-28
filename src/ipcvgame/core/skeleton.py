"""Canonical skeleton: COCO-17 keypoint names and indices.

Everyone indexes keypoints by name (``kpts[L_WRIST]``), never by raw number.
Pose adapters convert model-specific layouts into this order, so Task 2 can
swap models without breaking Tasks 1, 3 and 4.
"""

NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
K = len(NAMES)
INDEX = {name: i for i, name in enumerate(NAMES)}

# Limb connections for drawing.
EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]

# Source-model layouts: canonical index i takes source index MAP[i].
# YOLO-pose outputs COCO-17 directly.
FROM_COCO17 = list(range(K))
# MediaPipe Pose (33 landmarks) -> COCO-17.
FROM_MEDIAPIPE33 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Left/right and mirroring (checked with YOLO26n-pose): models label sides as
# if the image were an ordinary photo, so for a player facing the camera L_*
# lands on the RIGHT of the screen. Our frames are mirrored, where the player's
# own left hand appears on the screen's left; therefore L_WRIST is the
# player's own RIGHT wrist. The adapter does no swap yet (open team decision,
# see docs/phase-1/status.md); if we swap, it is one mapping here.
