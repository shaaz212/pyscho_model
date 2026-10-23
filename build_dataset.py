"""Step 4: turn recorded clips into train / validation / test piles.

Split by recording time (filenames are timestamps), not randomly, so test clips
are never neighbours of training clips (leakage). Per label: last 2 clips = test,
the 1 before = validation, the rest = train. Saves dataset.npz.
"""
from pathlib import Path

import numpy as np

FRAMES = 36          # every clip is cut to this length (our shortest usable clip)
MAX_MISSING = 0.2    # skip clips where nobody was found in more than 20% of frames

labels = sorted(p.name for p in Path("data/pose").iterdir() if p.is_dir())
piles = {"train": ([], []), "val": ([], []), "test": ([], [])}

for label_id, label in enumerate(labels):
    clips = []
    for f in sorted(Path("data/pose", label).glob("*.npy")):  # oldest first
        clip = np.load(f)
        nobody = (clip[:, :, 2].sum(axis=1) == 0).mean()
        if len(clip) >= FRAMES and nobody <= MAX_MISSING:
            clips.append(clip[:FRAMES])
        else:
            print(f"skipped {f.name}: {len(clip)} frames, nobody in {nobody:.0%}")
    for name, group in (("test", clips[-2:]), ("val", clips[-3:-2]), ("train", clips[:-3])):
        piles[name][0].extend(group)
        piles[name][1].extend([label_id] * len(group))

arrays = {}
for name, (x, y) in piles.items():
    arrays[f"x_{name}"], arrays[f"y_{name}"] = np.stack(x), np.array(y)
    counts = {labels[i]: int((np.array(y) == i).sum()) for i in range(len(labels))}
    print(f"{name:5} {arrays[f'x_{name}'].shape}  {counts}")

np.savez("dataset.npz", labels=np.array(labels), **arrays)
print("saved dataset.npz, labels:", labels)
