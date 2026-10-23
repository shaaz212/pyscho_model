from transformers import pipeline

print("Loading local model...")

classifier = pipeline(
    "image-classification",
    model="./model",
)

print("Model loaded successfully!")
