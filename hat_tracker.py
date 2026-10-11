"""Webcam face tracking with a hat.

Install: pip install opencv-python, mediapipe, numpy
Run:     python hat_tracker.py   (press q to quit)
"""

import cv2
import numpy as np
import mediapipe as mp #this is the AI model that finds face landmarks

FOREHEAD, LEFT_SIDE, RIGHT_SIDE, UPPER_LIP = 10, 234, 454, 164  # Face Mesh landmark indices

HAT = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED) # this loads the image you use for your hat
MUSTACHE = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED) # this loads the image you use for your mustache


def draw_hat(img, origin, angle, w):
    origin = (origin[0], origin[1] + 0.40 * w) #because the hat was to high, I lowered the origin
    hh, hw = HAT.shape[:2]
    scale = (1.5 * w) / hw  # hat width = 1.5 x face width
    # rotate and scale around the bottom-middle of the hat image
    M = cv2.getRotationMatrix2D((hw / 2, hh), -np.degrees(angle), scale)
    # then move that anchor point onto the forehead landmark
    M[0, 2] += origin[0] - hw / 2
    M[1, 2] += origin[1] - hh
    warped = cv2.warpAffine(HAT, M, (img.shape[1], img.shape[0]))
    alpha = warped[:, :, 3:4] / 255.0
    img[:] = (warped[:, :, :3] * alpha + img * (1 - alpha)).astype(np.uint8)

def draw_mustache(img, origin, angle, w):
    hh, hw = MUSTACHE.shape[:2]
    scale = (0.80 * w) / hw  # mustache width = 0.8 x face width
    # rotate and scale around the CENTER of the image
    M = cv2.getRotationMatrix2D((hw / 2, hh / 2), -np.degrees(angle), scale)
    M[0, 2] += origin[0] - hw / 2
    M[1, 2] += origin[1] - hh / 2
    warped = cv2.warpAffine(MUSTACHE, M, (img.shape[1], img.shape[0]))
    alpha = warped[:, :, 3:4] / 255.0
    img[:] = (warped[:, :, :3] * alpha + img * (1 - alpha)).astype(np.uint8)


def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view
        h, w_img = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for face in result.multi_face_landmarks or []:
            pt = lambda i: np.array([face.landmark[i].x * w_img, face.landmark[i].y * h])
            left, right, top = pt(LEFT_SIDE), pt(RIGHT_SIDE), pt(FOREHEAD)
            lip = pt(UPPER_LIP)

            face_width = np.linalg.norm(right - left)
            d = right - left
            angle = np.arctan2(d[1], d[0])  # head roll
            draw_hat(frame, top, angle, face_width)
            draw_mustache(frame, lip, angle, face_width)

        cv2.imshow("Facial tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()