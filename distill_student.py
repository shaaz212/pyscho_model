"""Train the small face model (student) two ways at once:
  1. copy the big teacher's answers (distillation) on the 5,000 images it labelled
  2. learn the real labels on all 28,709 FER-2013 training images

Validation = 1,000 images from FER's own validation file. The test pile is NOT used here.
"""
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from face_student import FER_TO_TEACHER, build_student, prepare

torch.manual_seed(0)


def load(split):
    df = pd.read_parquet(f"data/fer/data/{split}-00000-of-00001.parquet")
    images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("L")) for d in df["image"]])
    return df, images, torch.tensor(np.array(FER_TO_TEACHER)[df["label"].to_numpy()])


df, images, labels = load("train")
x = prepare(images)

# line the teacher's answers up with the right training images (same seed as teacher_logits.py)
teacher = np.load("data/fer/teacher_train.npz")
rows = df.sample(n=5000, random_state=0).index.to_numpy()
assert (images[rows] == teacher["images"]).all(), "teacher answers do not match the images"
t_logits = torch.zeros(len(x), 7)
t_logits[rows] = torch.tensor(teacher["logits"])
has_teacher = torch.zeros(len(x), dtype=torch.bool)
has_teacher[rows] = True

_, val_images, val_labels = load("valid")
val_images, val_labels = prepare(val_images[:1000]), val_labels[:1000]

student = build_student()
EPOCHS, BATCH, LR, T = 7, 64, 2e-3, 2.0
optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LR, total_steps=EPOCHS * ((len(x) + BATCH - 1) // BATCH))
ce = nn.CrossEntropyLoss()
kl = nn.KLDivLoss(reduction="batchmean")

best = -1.0
for epoch in range(1, EPOCHS + 1):
    student.train()
    order = torch.randperm(len(x))
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        batch = x[idx]
        flip = torch.rand(len(idx)) < 0.5                                  # mirror half the faces
        batch = torch.where(flip[:, None, None, None], batch.flip(3), batch)
        batch = batch * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1))         # brightness change
        out = student(batch)
        loss = ce(out, labels[idx])
        m = has_teacher[idx]
        if m.any():
            loss = loss + kl(nn.functional.log_softmax(out[m] / T, dim=1),
                             nn.functional.softmax(t_logits[idx][m] / T, dim=1)) * T * T
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    student.eval()
    with torch.no_grad():
        acc = (student(val_images).argmax(1) == val_labels).float().mean().item()
    if acc >= best:
        best = acc
        torch.save(student.state_dict(), "face_student.pt")
    print(f"epoch {epoch}  loss {loss.item():.3f}  validation accuracy {acc:.0%}", flush=True)

print(f"best validation accuracy: {best:.0%}  (saved face_student.pt)")
