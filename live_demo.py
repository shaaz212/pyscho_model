"""Step 7c: the full chain on the webcam, using only the two ONNX files. Press q to quit.

camera frame -> pose model (yolo11n-pose.onnx) -> 17 points
             -> last 36 frames -> movement model (movement_model.onnx) -> behaviour
"""
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

FPS, FRAMES = 10, 36   # same as when we recorded the clips

labels = list(np.load("dataset.npz")["labels"])
pose_model = YOLO("yolo11n-pose.onnx", task="pose")
movement = ort.InferenceSession("movement_model.onnx")

history = deque(maxlen=FRAMES)   # the flipbook: last 36 skeleton frames
text, next_tick = "collecting frames...", 0.0
cap = cv2.VideoCapture(0)
start = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = pose_model(frame, verbose=False)[0]
    if time.time() - start >= next_tick:          # keep 10 frames per second, like the clips
        next_tick += 1 / FPS
        points = np.zeros((17, 3), dtype=np.float32)
        if result.keypoints is not None and len(result.boxes) > 0:
            best = int(result.boxes.conf.argmax())
            xy = result.keypoints.xyn[best].cpu().numpy()
            conf = result.keypoints.conf[best].cpu().numpy()
            points = np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)
        history.append(points)

        if len(history) == FRAMES:
            clip = np.stack(history).reshape(1, FRAMES, -1)
            scores = movement.run(None, {"clip": clip})[0][0]
            probs = np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()
            text = f"{labels[int(probs.argmax())]} {probs.max():.0%}"

    img = result.plot()
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "PROTOTYPE: 35 clips, one person", (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("live demo (q to quit)", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
