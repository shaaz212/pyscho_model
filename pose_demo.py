"""Step 1: live skeleton (pose) on the webcam. Press q to quit.

A pretrained pose model finds each person and gives 17 keypoints
(nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
"""
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # "n" = nano, the smallest version (downloads on first run)

cap = cv2.VideoCapture(0)  # for an IP camera use: cv2.VideoCapture("rtsp://user:pass@ip/stream")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, verbose=False)[0]  # inference: frozen weights, no learning
    cv2.imshow("pose demo (q to quit)", result.plot())  # draws the stick figure

    if result.keypoints is not None and len(result.keypoints) > 0:
        points = result.keypoints.xy[0]  # 17 rows of (x, y) for the first person
        print(f"person found, nose at ({points[0][0]:.0f}, {points[0][1]:.0f})", end="\r")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
