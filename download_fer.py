"""Download the public FER-2013 face images (parquet files) into data/fer/."""
from huggingface_hub import hf_hub_download

for name in ("train", "valid", "test"):
    path = hf_hub_download("AutumnQiu/fer2013", f"data/{name}-00000-of-00001.parquet",
                           repo_type="dataset", local_dir="data/fer")
    print("saved", path)
