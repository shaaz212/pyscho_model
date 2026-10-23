"""Step 5: train a tiny GRU that reads a skeleton flipbook and names the behaviour.

Same 5 steps as Lesson 3: take batch -> forward pass -> loss -> backward pass -> update.
Only train and validation are used here. The test pile stays locked.
"""
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
data = np.load("dataset.npz")
labels = list(data["labels"])


def to_tensor(x):
    x = torch.tensor(x, dtype=torch.float32)       # (clips, 36 frames, 17 points, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)   # (clips, 36 frames, 51 numbers)


x_train, y_train = to_tensor(data["x_train"]), torch.tensor(data["y_train"])
x_val, y_val = to_tensor(data["x_val"]), torch.tensor(data["y_val"])


class MovementModel(nn.Module):
    def __init__(self, inputs=51, hidden=32, classes=len(labels)):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)  # reads the frames in order
        self.head = nn.Linear(hidden, classes)                # one score per behaviour

    def forward(self, x):
        _, last = self.gru(x)           # memory after the last frame
        return self.head(last[0])


model = MovementModel()
print("weights:", sum(p.numel() for p in model.parameters()))

LEARNING_RATE, EPOCHS, BATCH = 0.01, 40, 8   # hyperparameters: we choose these
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def accuracy(x, y):
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


best_val = -1.0
for epoch in range(1, EPOCHS + 1):
    order = torch.randperm(len(x_train))
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]                    # 1. take a batch
        loss = loss_fn(model(x_train[idx]), y_train[idx])  # 2. forward pass, 3. loss
        optimizer.zero_grad()
        loss.backward()                             # 4. backward pass (which direction)
        optimizer.step()                            # 5. update (how far = learning rate)

    train_acc, val_acc = accuracy(x_train, y_train), accuracy(x_val, y_val)
    if val_acc >= best_val:                         # keep the epoch with the best mock exam
        best_val = val_acc
        torch.save(model.state_dict(), "movement_model.pt")
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:2}  loss {loss.item():.3f}  train {train_acc:.0%}  validation {val_acc:.0%}")

print(f"best validation: {best_val:.0%}  (saved to movement_model.pt)")
