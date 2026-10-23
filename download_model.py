from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dima806/facial_emotions_image_detection",
    local_dir="./model"
)

print("Model downloaded successfully.")