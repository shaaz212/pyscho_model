"""Run the big face model (the teacher) on FER-2013 images and save what it says.

Saves data/fer/teacher_<split>.npz with: images (N,48,48) uint8, labels (N,), logits (N,7).
The student will later learn to copy these logits (distillation).
Example: python teacher_logits.py --split train --n 6000
"""
import argparse
import io
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True)   # train, valid or test
parser.add_argument("--n", type=int, default=1000)
args = parser.parse_args()

df = pd.read_parquet(f"data/fer/data/{args.split}-00000-of-00001.parquet")
df = df.sample(n=min(args.n, len(df)), random_state=0)  # fixed seed: same pick every time

teacher = AutoModelForImageClassification.from_pretrained("model").eval()
print("teacher labels:", teacher.config.id2label)

images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("L")) for d in df["image"]])
labels = df["label"].to_numpy()
print("images:", images.shape)


def prepare(batch):
    x = torch.tensor(batch, dtype=torch.float32) / 255.0                 # 0..1
    x = torch.nn.functional.interpolate(x[:, None], size=224, mode="bilinear")  # 48 -> 224
    x = x.repeat(1, 3, 1, 1)                                             # grey -> 3 channels
    return (x - 0.5) / 0.5                                               # teacher's normalisation


logits, start = [], time.time()
with torch.inference_mode():
    for i in range(0, len(images), 32):
        logits.append(teacher(pixel_values=prepare(images[i:i + 32])).logits.numpy())
        done = min(i + 32, len(images))
        print(f"\r{done}/{len(images)}  {done / (time.time() - start):.1f} images/s", end="", flush=True)
print()

np.savez(f"data/fer/teacher_{args.split}.npz", images=images, labels=labels, logits=np.concatenate(logits))
print("saved")
