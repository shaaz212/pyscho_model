"""Quantize custody_model.onnx to int8 and run the final exam on the locked test pile.

Calibration = a few training samples that show the quantizer the normal range of the numbers.
The GRU (movement) part has no int8 version in ONNX Runtime, so it stays float; it is tiny.
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from face_student import FER_TO_TEACHER, prepare

data = np.load("dataset.npz")
train_clips = data["x_train"].reshape(len(data["x_train"]), 36, -1).astype(np.float32)
train_faces = prepare(np.load("data/fer/teacher_train.npz")["images"][:200]).numpy()


class Reader(CalibrationDataReader):
    def __init__(self):
        self.items = iter(range(len(train_faces)))

    def get_next(self):
        i = next(self.items, None)
        if i is None:
            return None
        return {"clip": train_clips[i % len(train_clips)][None], "face": train_faces[i][None]}


# Found on validation faces: quantizing the first 2 conv layers breaks the face model
# (accuracy 69% -> 15%), because they produce a few huge values. Keep those two in float.
nodes = onnx.load("custody_model.onnx").graph.node
convs = [n.name for n in nodes if n.op_type == "Conv"]
gemms = [n.name for n in nodes if n.op_type == "Gemm"]
quantize_static("custody_model.onnx", "custody_model_int8.onnx", Reader(),
                quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                op_types_to_quantize=["Conv", "Gemm"], nodes_to_quantize=convs[2:] + gemms)
print(f"float: {os.path.getsize('custody_model.onnx') / 1e6:.1f} MB   int8: {os.path.getsize('custody_model_int8.onnx') / 1e6:.1f} MB")

# ---- final exam on the locked test pile (run once, no tuning afterwards) ----
test = np.load("data/fer/teacher_test.npz")
faces = prepare(test["images"]).numpy()
true = np.array(FER_TO_TEACHER)[test["labels"]]
test_clips = data["x_test"].reshape(len(data["x_test"]), 36, -1).astype(np.float32)
test_clip_labels = data["y_test"]

print("\nfinal exam, 1000 face images (teacher = big 86M-weight model):")
print(f"  teacher (big)       {(test['logits'].argmax(1) == true).mean():.0%}")
for name in ("custody_model.onnx", "custody_model_int8.onnx"):
    session = ort.InferenceSession(name)
    emotions = np.concatenate([
        session.run(None, {"clip": test_clips[:1], "face": faces[i:i + 1]})[1] for i in range(len(faces))])
    behaviours = np.concatenate([
        session.run(None, {"clip": test_clips[i:i + 1], "face": faces[:1]})[0] for i in range(len(test_clips))])
    print(f"  {name:<24} emotion {(emotions.argmax(1) == true).mean():.0%}"
          f"   behaviour {(behaviours.argmax(1) == test_clip_labels).mean():.0%} ({len(test_clips)} clips)")
