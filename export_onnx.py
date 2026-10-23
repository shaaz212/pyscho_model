"""Step 7a: export the movement model to ONNX and check it gives the same answers."""
import os

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

data = np.load("dataset.npz")
labels = list(data["labels"])


class MovementModel(nn.Module):  # same shape as in train_movement.py
    def __init__(self, inputs=51, hidden=32, classes=len(labels)):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        _, last = self.gru(x)
        return self.head(last[0])


model = MovementModel()
model.load_state_dict(torch.load("movement_model.pt"))
model.eval()

example = torch.zeros(1, 36, 51)  # one clip: 36 frames x 51 numbers (17 points x 3)
torch.onnx.export(model, (example,), "movement_model.onnx",
                  input_names=["clip"], output_names=["scores"])
print(f"saved movement_model.onnx  ({os.path.getsize('movement_model.onnx') / 1024:.0f} KB)")

# Check: PyTorch and ONNX Runtime must give the same answers on the same clips.
x = torch.tensor(data["x_test"], dtype=torch.float32)
x = x.reshape(x.shape[0], x.shape[1], -1)
session = ort.InferenceSession("movement_model.onnx")
with torch.no_grad():
    torch_scores = model(x).numpy()
onnx_scores = np.concatenate([session.run(None, {"clip": clip[None].numpy()})[0] for clip in x])

print("biggest difference between the two:", float(np.abs(torch_scores - onnx_scores).max()))
print("same predictions:", bool((torch_scores.argmax(1) == onnx_scores.argmax(1)).all()))
