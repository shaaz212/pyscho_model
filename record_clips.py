"""Step 3: record practice clips as skeleton numbers (not video).

Example: python record_clips.py --label pacing --clips 5 --seconds 5
Each clip is saved as data/pose/<label>/<timestamp>.npy with shape (frames, 17, 3):
17 body points, each with x, y (0 to 1 of the frame) and a confidence. Press q to quit.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--clips", type=int, default=5)
parser.add_argument("--seconds", type=int, default=5)
parser.add_argument("--fps", type=int, default=10)
args = parser.parse_args()

out_dir = Path("data/pose") / args.label
out_dir.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)


def read_pose(frame):
    """Return (17, 3) array for the most confident person, or zeros if nobody is seen."""
    result = model(frame, verbose=False)[0]
    if result.keypoints is None or len(result.boxes) == 0:
        return np.zeros((17, 3), dtype=np.float32), result
    best = int(result.boxes.conf.argmax())
    xy = result.keypoints.xyn[best].cpu().numpy()
    conf = result.keypoints.conf[best].cpu().numpy()
    return np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32), result


def show(result, text):
    img = result.plot()
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("record clips (q to quit)", img)
    return cv2.waitKey(1) & 0xFF == ord("q")


for clip in range(1, args.clips + 1):
    # 3-second countdown so you can get ready
    start = time.time()
    while time.time() - start < 3:
        ok, frame = cap.read()
        _, result = read_pose(frame)
        left = 3 - int(time.time() - start)
        if show(result, f"{args.label}: clip {clip}/{args.clips} starts in {left}"):
            raise SystemExit

    frames, next_tick, start = [], 0.0, time.time()
    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if time.time() - start >= next_tick:  # keep only `fps` frames per second
            pose, result = read_pose(frame)
            frames.append(pose)
            next_tick += 1 / args.fps
            if show(result, f"REC {args.label} {clip}/{args.clips}"):
                raise SystemExit

    path = out_dir / f"{int(time.time())}.npy"
    np.save(path, np.stack(frames))
    print(f"saved {path}  shape={np.stack(frames).shape}")

cap.release()
cv2.destroyAllWindows()
