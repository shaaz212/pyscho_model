"""Step 6: the final exam. Run this once on the locked test pile."""
import numpy as np
import torch
from torch import nn

data = np.load("dataset.npz")
labels = list(data["labels"])
x_test = torch.tensor(data["x_test"], dtype=torch.float32)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], -1)
y_test = torch.tensor(data["y_test"])


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

with torch.no_grad():
    guess = model(x_test).argmax(dim=1)

print(f"test accuracy: {(guess == y_test).float().mean().item():.0%}  ({len(y_test)} clips)\n")
print("clip  real answer       model said")
for i, (real, said) in enumerate(zip(y_test.tolist(), guess.tolist()), 1):
    mark = "ok" if real == said else "WRONG"
    print(f"{i:>4}  {labels[real]:<16}  {labels[said]:<16} {mark}")
