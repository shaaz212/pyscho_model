"""Lesson 3 playground: watch a tiny model learn, epoch by epoch.

This model has only 2 weights (our face model has 86 million) and learns
to convert Celsius to Fahrenheit. The true formula is F = C x 1.8 + 32,
but the model doesn't know that: it starts with random weights and has
to learn them from 20 examples.

Change the 3 settings below, run it again, and see what happens.
"""
import torch

LEARNING_RATE = 0.5   # how big each weight adjustment is. Try 0.001 and 1.5
EPOCHS = 30           # how many times we go through all 20 examples
BATCH_SIZE = 4        # examples per adjustment (step)

torch.manual_seed(0)

# Training data: 20 temperatures with the correct answers (the labels).
celsius = torch.linspace(-50, 50, 20)
fahrenheit = celsius * 1.8 + 32
x = celsius / 50  # scaled to -1..1, which makes learning easier

# The model's 2 weights, starting random. requires_grad tells PyTorch to learn them.
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=LEARNING_RATE)

print(f"Start:    model thinks F = C x {w.item() / 50:.2f} + {b.item():.1f}\n")

for epoch in range(1, EPOCHS + 1):
    order = torch.randperm(len(x))  # shuffle the examples each epoch
    for i in range(0, len(x), BATCH_SIZE):
        batch = order[i:i + BATCH_SIZE]           # 1. take a batch
        guess = x[batch] * w + b                  # 2. forward pass: the model guesses
        loss = ((guess - fahrenheit[batch]) ** 2).mean()  # 3. loss: how wrong it was
        optimizer.zero_grad()
        loss.backward()                           # 4. backward pass: which way to turn each weight
        optimizer.step()                          # 5. update: turn the weights a little

    with torch.no_grad():
        error = (x * w + b - fahrenheit).abs().mean().item()
    print(f"Epoch {epoch:2d}: off by {error:8.2f} degrees F on average   "
          f"model thinks F = C x {w.item() / 50:.2f} + {b.item():.1f}")

print(f"\nTrue answer:            F = C x 1.80 + 32.0")
with torch.no_grad():
    print(f"Test: 100 degrees C -> model says {(2 * w + b).item():.1f} degrees F (correct: 212.0 degrees F)")
