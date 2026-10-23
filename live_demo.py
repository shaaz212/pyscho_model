"""Live demo of the whole chain. Press q to quit.

camera frame -> pose model (yolo11n-pose.onnx) -> 17 points
  points of the last 36 frames ------------\
  face cropped around the head points ------+-> ONE model (custody_model_int8.onnx)
                                                -> behaviour + emotion
"""
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from face_student import EMOTIONS, prepare

FPS, FRAMES = 10, 36   # same as when we recorded the clips
MODEL = "custody_model_int8.onnx"
behaviours = list(np.load("dataset.npz")["labels"])


def softmax(scores):
    e = np.exp(scores - scores.max())
    return e / e.sum()


def face_crop(frame, points):
    """Grey 48x48 face cut out around nose, eyes and ears (pose points 0-4), or None if not visible."""
    head = points[:5]
    head = head[head[:, 2] > 0.5]
    if len(head) < 3:
        return None
    h, w = frame.shape[:2]
    xs, ys = head[:, 0] * w, head[:, 1] * h
    side = max(40.0, 2.4 * (xs.max() - xs.min()))
    cx, cy = xs.mean(), ys.mean() + 0.15 * side   # eyes are above the middle of a face
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    grey = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (48, 48), interpolation=cv2.INTER_AREA)


def read_pose(result):
    points = np.zeros((17, 3), dtype=np.float32)
    if result.keypoints is not None and len(result.boxes) > 0:
        best = int(result.boxes.conf.argmax())
        xy = result.keypoints.xyn[best].cpu().numpy()
        conf = result.keypoints.conf[best].cpu().numpy()
        points = np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)
    return points


def run_model(session, history, face):
    clip = np.stack(history).reshape(1, FRAMES, -1)
    face_in = prepare(face[None] if face is not None else np.zeros((1, 48, 48), np.uint8)).numpy()
    behaviour, emotion = session.run(None, {"clip": clip, "face": face_in})
    b, e = softmax(behaviour[0]), softmax(emotion[0])
    emotion_text = f"{EMOTIONS[int(e.argmax())]} {e.max():.0%}" if face is not None else "face not visible"
    return f"{behaviours[int(b.argmax())]} {b.max():.0%}", emotion_text


if __name__ == "__main__":
    pose_model = YOLO("yolo11n-pose.onnx", task="pose")
    session = ort.InferenceSession(MODEL)
    history = deque(maxlen=FRAMES)   # the flipbook: last 36 skeleton frames
    behaviour_text, emotion_text, next_tick = "collecting frames...", "", 0.0
    cap = cv2.VideoCapture(0)
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pose_model(frame, verbose=False)[0]
        if time.time() - start >= next_tick:          # keep 10 frames per second, like the clips
            next_tick += 1 / FPS
            points = read_pose(result)
            history.append(points)
            if len(history) == FRAMES:
                behaviour_text, emotion_text = run_model(session, history, face_crop(frame, points))

        img = result.plot()
        cv2.putText(img, f"movement: {behaviour_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(img, f"face: {emotion_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(img, "PROTOTYPE: 35 clips, one person", (10, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("live demo (q to quit)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
