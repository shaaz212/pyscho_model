"""Fine-tune the local ViT facial-expression model on your own images.

Expected layout (one folder per emotion):
    data/train/<label>/*.jpg
    data/val/<label>/*.jpg

If the folder names match the model's labels (sad, disgust, angry, neutral,
fear, surprise, happy) the existing classifier head is kept; otherwise a new
head is created for your labels.
"""
import argparse

import numpy as np
import torch
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./data")
parser.add_argument("--model", default="./model")
parser.add_argument("--output", default="./model-finetuned")
parser.add_argument("--epochs", type=float, default=3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-5)
args = parser.parse_args()

processor = AutoImageProcessor.from_pretrained(args.model)
size = processor.size["height"]
normalize = transforms.Normalize(processor.image_mean, processor.image_std)

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize,
])
val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    normalize,
])

train_ds = datasets.ImageFolder(f"{args.data}/train", transform=train_tf)
val_ds = datasets.ImageFolder(f"{args.data}/val", transform=val_tf)
classes = train_ds.classes

model = AutoModelForImageClassification.from_pretrained(args.model)
if set(classes) == set(model.config.label2id):
    label2id = model.config.label2id
else:
    print(f"Labels {classes} differ from the model's; training a new classifier head.")
    label2id = {c: i for i, c in enumerate(classes)}
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=len(classes),
        label2id=label2id,
        id2label={i: c for c, i in label2id.items()},
        ignore_mismatched_sizes=True,
    )

# ImageFolder numbers classes alphabetically; map them onto the model's label ids.
folder_to_model = {i: label2id[c] for i, c in enumerate(classes)}


def collate(batch):
    return {
        "pixel_values": torch.stack([img for img, _ in batch]),
        "labels": torch.tensor([folder_to_model[y] for _, y in batch]),
    }


def compute_metrics(pred):
    return {"accuracy": (np.argmax(pred.predictions, axis=1) == pred.label_ids).mean()}


training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    weight_decay=0.02,
    warmup_steps=0.1,  # float < 1 = fraction of total steps
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="accuracy",
    logging_steps=50,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
    compute_metrics=compute_metrics,
)
trainer.train()

# Reload the best epoch with from_pretrained rather than load_best_model_at_end: in
# transformers 5 the Trainer's reload skips ViT's key renaming and only partially loads.
best = AutoModelForImageClassification.from_pretrained(trainer.state.best_model_checkpoint)
trainer.model = trainer.model_wrapped = best
print(trainer.evaluate())

trainer.save_model(args.output)
processor.save_pretrained(args.output)
print(f"Saved fine-tuned model to {args.output}")
