"""Try several int8 settings and pick the best using VALIDATION faces (not the test pile)."""
import io
import os
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from PIL import Image

from face_student import FER_TO_TEACHER, prepare

out_dir = sys.argv[1]
data = np.load("dataset.npz")
clip = data["x_train"].reshape(len(data["x_train"]), 36, -1).astype(np.float32)[:1]

val = pd.read_parquet("data/fer/data/valid-00000-of-00001.parquet").iloc[:500]
val_faces = prepare(np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("L")) for d in val["image"]])).numpy()
val_true = np.array(FER_TO_TEACHER)[val["label"].to_numpy()]
calib = prepare(np.load("data/fer/teacher_train.npz")["images"][:300]).numpy()


class Reader(CalibrationDataReader):
    def __init__(self):
        self.it = iter(range(len(calib)))

    def get_next(self):
        i = next(self.it, None)
        return None if i is None else {"clip": clip, "face": calib[i][None]}


def emotion_accuracy(path):
    s = ort.InferenceSession(path)
    out = np.concatenate([s.run(None, {"clip": clip, "face": val_faces[i:i + 1]})[1] for i in range(len(val_faces))])
    return (out.argmax(1) == val_true).mean()


print(f"float baseline        validation emotion {emotion_accuracy('custody_model.onnx'):.0%}", flush=True)
configs = {
    "A conv+gemm, int8/int8":   dict(ops=["Conv", "Gemm"], act=QuantType.QInt8,  method=CalibrationMethod.MinMax),
    "B conv+gemm, uint8 act":   dict(ops=["Conv", "Gemm"], act=QuantType.QUInt8, method=CalibrationMethod.MinMax),
    "C conv+gemm, percentile":  dict(ops=["Conv", "Gemm"], act=QuantType.QUInt8, method=CalibrationMethod.Percentile),
    "D conv only, uint8 act":   dict(ops=["Conv"],         act=QuantType.QUInt8, method=CalibrationMethod.MinMax),
}
for name, c in configs.items():
    path = os.path.join(out_dir, name[0] + ".onnx")
    quantize_static("custody_model.onnx", path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=c["act"], op_types_to_quantize=c["ops"],
                    calibrate_method=c["method"])
    print(f"{name:<26} {os.path.getsize(path) / 1e6:.1f} MB   validation emotion {emotion_accuracy(path):.0%}", flush=True)
