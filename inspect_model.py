
from safetensors import safe_open

model_path = "model/model.safetensors"

with safe_open(model_path, framework="pt", device="cpu") as f:
    print("Total tensors:", len(f.keys()))
    print("\nModel contents:\n")

    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
