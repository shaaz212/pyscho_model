"""Live facial-emotion detection from the laptop webcam. Press q to quit.

Each detected face gets a box with its top emotion; the panel on the left
shows the model's confidence for all 7 emotions for the first face.
"""
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL = "./model"

processor = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForImageClassification.from_pretrained(MODEL).eval()
labels = model.config.id2label
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict(face_gray):
    """Return emotion probabilities for one grayscale face crop."""
    image = Image.fromarray(face_gray).convert("RGB")
    with torch.no_grad():
        logits = model(**processor(image, return_tensors="pt")).logits
    return logits.softmax(-1)[0]


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise SystemExit("Could not open the webcam. Is another app using it?")

while True:
    ok, frame = cam.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for i, (x, y, w, h) in enumerate(faces):
        probs = predict(gray[y:y + h, x:x + w])
        conf, idx = probs.max(0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{labels[idx.item()]} {conf:.0%}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if i == 0:
            for row, (j, p) in enumerate(sorted(enumerate(probs.tolist()), key=lambda t: -t[1])):
                y_row = 30 + row * 28
                cv2.rectangle(frame, (10, y_row - 15), (10 + int(p * 150), y_row + 5), (0, 200, 255), -1)
                cv2.putText(frame, f"{labels[j]} {p:.0%}", (170, y_row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Emotion detection (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
