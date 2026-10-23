"""Join the movement model and the small face model into ONE ONNX file.

Inputs : clip (1, 36, 51) skeleton numbers, face (1, 1, 96, 96) grey face crop
Outputs: behaviour (1, 4) scores, emotion (1, 7) scores
The two parts are not mixed together (no joint training data yet); they only travel in one file.
"""
import os

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from face_student import EMOTIONS, build_student, prepare

data = np.load("dataset.npz")
behaviours = list(data["labels"])


class MovementModel(nn.Module):  # same shape as in train_movement.py
    def __init__(self, inputs=51, hidden=32, classes=len(behaviours)):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        _, last = self.gru(x)
        return self.head(last[0])


class Custody(nn.Module):
    def __init__(self, movement, face):
        super().__init__()
        self.movement, self.face = movement, face

    def forward(self, clip, face):
        return self.movement(clip), self.face(face)


movement = MovementModel()
movement.load_state_dict(torch.load("movement_model.pt"))
face = build_student(pretrained=False)
face.load_state_dict(torch.load("face_student.pt"))
model = Custody(movement, face).eval()

torch.onnx.export(model, (torch.zeros(1, 36, 51), torch.zeros(1, 1, 96, 96)), "custody_model.onnx",
                  input_names=["clip", "face"], output_names=["behaviour", "emotion"],
                  external_data=False)
print(f"saved custody_model.onnx  ({os.path.getsize('custody_model.onnx') / 1e6:.1f} MB)")

# Check: the one-file model must give the same answers as the two PyTorch models.
clips = torch.tensor(data["x_test"], dtype=torch.float32).reshape(len(data["x_test"]), 36, -1)
faces = prepare(np.load("data/fer/teacher_test.npz")["images"][:len(clips)])
session = ort.InferenceSession("custody_model.onnx")
with torch.no_grad():
    want_b, want_e = model(clips, faces)
worst_b = worst_e = 0.0
for i in range(len(clips)):
    got_b, got_e = session.run(None, {"clip": clips[i:i + 1].numpy(), "face": faces[i:i + 1].numpy()})
    worst_b = max(worst_b, float(np.abs(got_b - want_b[i:i + 1].numpy()).max()))
    worst_e = max(worst_e, float(np.abs(got_e - want_e[i:i + 1].numpy()).max()))
print("biggest difference, behaviour:", worst_b, " emotion:", worst_e)
print("input names:", [i.name for i in session.get_inputs()], " output names:", [o.name for o in session.get_outputs()])
print("operations used:", sorted({n.op_type for n in __import__('onnx').load('custody_model.onnx').graph.node}))
