# AI Training Course: My Study Notes

**Started:** 2026-10-22
**Course doc (full lessons):** https://claude.ai/artifact/STHjxMEErJwJkbHeU4qcPh

## How to continue in a new chat

Open a new Claude chat in this folder and say:

> "I'm learning AI model training as a beginner. Read `STUDY_NOTES.md`. First give me a review quiz on Lessons 1–3, then continue with Lesson 4. Teach with simple stories and small quizzes."

---

## Progress

| Lesson | Topic | Status |
|---|---|---|
| 1 | What an AI model is (weights, training vs inference) | ✅ Done |
| 2 | Tools: PyTorch, Hugging Face, CPU vs GPU, CUDA | ✅ Done |
| 3 | How training works (loss, learning rate, batch, epoch) | ✅ Done |
| 4 | Data (labels, train/val/test, overfitting, leakage) | ⏳ Both questions passed. Still to learn: augmentation, leakage example, public datasets |
| 5 | Measuring a model (accuracy, precision, recall) | Not started |
| 6 | Our model and fine-tuning (ViT, `train.py`) | Not started |
| 7 | Making it light and fast (quantization, server sizing) | Not started |
| 8 | Our use case, honestly (custody monitoring) | Not started |
| 9 | The full system (cameras to guard alerts) | Not started |

**Next session plan:**
1. Review quiz on Lessons 1–3 (warm-up).
2. Answer the 2 Lesson 4 questions at the bottom of this file.
3. Continue Lesson 4: augmentation, leakage, datasets.

**Remember from the Lesson 3 final test:**
- `loss.backward()` = **which direction** to turn the knobs. The **learning rate** = **how far** to turn them.
- Batch ≠ epoch: a batch = 16 darts thrown at once; an epoch = the whole bucket.

---

## My project files

| File | What it does | How to run |
|---|---|---|
| `model/` | The downloaded, trained emotion model (86 million weights) | n/a |
| `webcam_demo.py` | Live emotion detection on my webcam | `python webcam_demo.py` (press q to quit) |
| `lesson3_playground.py` | Tiny model (2 weights) that learns °C → °F, to watch training happen | `python lesson3_playground.py` |
| `train.py` | Fine-tunes the face model on my own images | `python train.py --epochs 3` |
| `test_model.py` | Checks the model loads | `python test_model.py` |

---

## Lesson 1: What an AI model is

- A model is a big math formula that turns an input (a face photo) into an output ("happy 92%").
- **Weights** = the adjustable numbers (knobs) inside the formula. Our model has **86 million**.
- The model file (`model.safetensors`, 340 MB) holds **only the weights**, not the training images. 86 million × 4 bytes ≈ 344 MB.
- **Training** = adjusting the weights using labelled examples. It happens once, first, on a strong computer.
- **Inference** = using the frozen weights to predict. This is what runs on every camera frame. The model does **not** learn during inference.
- Order: train → save weights → camera sends a frame → model predicts.

**Say it to your manager:** "A model is a formula learned from examples. Training builds it once on a powerful machine; after that it runs on every camera frame on our server."

**Extra:** a model file doesn't contain the photos, but models trained on small data can leak hints about them. Once we train on detainee footage, we protect the model file like sensitive data.

---

## Lesson 2: The tools

| Tool | What it is |
|---|---|
| **Python** | The programming language |
| **PyTorch** | The engine that does the math. Free, open source. Created by Meta (2016), now run by the PyTorch Foundation (Linux Foundation): Meta, Microsoft, Google, AWS, AMD, NVIDIA |
| **Hugging Face** | Company and website with ready-made models. Our model came from there |
| **transformers** | Hugging Face's Python library, which we use to load and train the model |
| **CUDA** | NVIDIA's software that lets PyTorch use an NVIDIA GPU |

- **CPU** = a few powerful cores. **GPU** = thousands of simple cores working at the same time (in parallel), so it's 20–50× faster for training.
- PyTorch runs on the **CPU** without CUDA, just more slowly. My laptop has Intel graphics (no NVIDIA), so everything runs on the CPU.
- A GPU server is **not automatically** faster. It needs all three of:
  1. an NVIDIA GPU
  2. the NVIDIA driver and the CUDA build of PyTorch installed
  3. code that moves the model and images to the GPU (`train.py` does this automatically through `Trainer`; `webcam_demo.py` would need a small change)
- Costs: the software is free. We pay for hardware, electricity and people.

**Say it to your manager:** "PyTorch is the industry-standard free, open-source AI engine. We use it with Hugging Face's pre-trained models. For production we need an NVIDIA GPU server."

---

## Lesson 3: How training works

### The dart story

| Training word | In the dart story |
|---|---|
| Weights | How you aim |
| Forward pass | Throw the dart (the model guesses) |
| Loss | How far from the centre you missed |
| Backward pass | Work out which way to adjust |
| Update | Adjust your aim a little |
| Batch | Throw 16 darts, then adjust based on the average miss |
| Epoch | Go through your whole bucket of darts once |

**The 5 steps in order:** take a batch → forward pass → loss → backward pass → update.

### The playground (`lesson3_playground.py`)

The model learns: **Answer = Celsius × ❓ + ❓** (correct: × 1.80 + 32).
Each line of output = one epoch. The knobs move closer to 1.80 and 32, and "off by" (the loss) shrinks.

### Learning rate

- **Learning rate = how far the model turns its knobs after each mistake.**
- **We** choose it (a hyperparameter). The **model** learns the weights.
- Rule (simplified): **turn = learning rate × 2 × error**.

| Learning rate | First turn (error 32) | What happens |
|---|---|---|
| 0.001 | 0.064 | Crawls, far too slow |
| 0.1 | 6.4 | Slides smoothly to the answer |
| 0.2 | 12.8 | Faster, still safe |
| 0.5 | 32 | Lands in one step |
| 1.0 | 64 | Bounces forever |
| 1.5 | 96 | Overshoots more each time, then explodes (inf, nan) |

- "Big" depends on the model: 0.01 is too **small** for the playground but **far too big** for fine-tuning our face model (we use 0.00001 = `1e-5`).
- How we choose it: proven recipes + a **learning-rate sweep** (try a few values, keep the one where the loss goes down steadily).
- Fine-tuning needs a tiny learning rate because the model is already an expert. Big swings make it forget (**catastrophic forgetting**).

### Weights vs hyperparameters

| | Who decides | Examples |
|---|---|---|
| **Weights** | The model learns them | The 1.80 and 32 in the playground; the 86 million numbers in `model.safetensors` |
| **Hyperparameters** | We choose them | Learning rate, epochs, batch size |

### The 4 loss patterns (reading training output)

| Pattern | Numbers look like | Meaning | Fix |
|---|---|---|---|
| 🐢 Slow | 51 → 50 → 49 → ... | Learning rate too small | Make it bigger |
| ✅ Good | 32 → 21 → 14 → ... → 0 | Just right | Keep it |
| 💥 Exploding | 20 → 900 → nan | Learning rate too big | Make it smaller |
| 😐 Stuck | 1.95 → 1.95 → 1.95 | Learned nothing (1.95 = random guessing among 7) | Learning rate far too small, or a bug such as wrong labels |

### Steps calculation

steps per epoch = images ÷ batch size. Total steps = steps per epoch × epochs.
Example: 2,000 images, batch 16, 5 epochs → 125 per epoch → **625 total**.

### Why several epochs?

Each step only turns the knobs a little. After epoch 1 the playground was still off by 32 degrees; it needed about 15 epochs. But too many epochs causes memorising (overfitting, Lesson 4).

**Say it to your manager:** "The model guesses, measures its error, and adjusts itself slightly, thousands of times. The learning rate is a setting we choose that controls how big each correction is. Making it bigger doesn't make training faster; it breaks the model. To go faster, we use a GPU."

### Lesson 3 practice questions (with answers)

1. **Put in order: update, loss, take batch, backward pass, forward pass.**
   → Take batch → forward pass → loss → backward pass → update.
2. **Loss stays at 1.95 for every epoch. What does it mean?**
   → Stuck: the model learned nothing (no better than random guessing).
3. **2,000 images, batch 16, 5 epochs: how many steps?**
   → 125 per epoch, 625 in total.
4. **Someone uses `--lr 0.01` and accuracy drops to 14%. Why?**
   → Learning rate far too big for fine-tuning; the model overshot and forgot what it knew. 14% ≈ 1 in 7 = random. Use 0.00001.
5. **Why not train for 100 epochs to be safe?**
   → Overfitting: the model memorises the training photos and gets worse on new ones. Keep the epoch with the best validation score.
6. **Output 40 → 39 → 38 → ... → 30. Fix?**
   → 🐢 Slow: make the learning rate bigger.
7. **Output 10 → 150 → 4000 → nan. Fix?**
   → 💥 Exploding: make the learning rate smaller.
8. **1,000 images, batch 10, 3 epochs?**
   → 100 per epoch, 300 in total.
9. **W or H: learning rate, the learned 1.80, epochs, batch size, 86 million numbers?**
   → H, W, H, H, W.
10. **Who learns the weights? Who chooses the learning rate?**
    → The model learns the weights; we choose the learning rate.
11. **Learning rate 0.2, error 32: how far does the knob turn on the first step?**
    → 0.2 × 2 × 32 = 12.8.
12. **Manager: "Make the learning rate 100× bigger so it finishes faster."**
    → It would overshoot and could break the model; to go faster, use a GPU.
13. **500 images, batch 25, 4 epochs?**
    → 20 per epoch, 80 in total.
14. **Learning rate 0.05, error 20: how far does the knob turn?**
    → 0.05 × 2 × 20 = 2.
15. **Pattern 100 → 60 → 35 → 20 → 12?**
    → ✅ Good: big drops each step. (🐢 Slow looks like 100 → 99 → 98: tiny drops.)
16. **What does `loss.backward()` do?**
    → The backward pass: works out which way and how much to turn every knob to reduce the error. `optimizer.step()` then turns them (the update).
17. **Manager: "Training on your laptop took 3 hours. How can we make it faster?"**
    → Use a GPU (Google Colab or the NVIDIA server), and test on a small set first. Don't raise the learning rate: slowness comes from the CPU, not the learning rate.

---

## Lesson 4: Data (in progress)

### The exam story

- **Labels** = the answer key. In our project, the folder name is the label (`data/train/happy/` = "happy"). No labels = no loss = no learning.

| Pile | Exam story | Used for |
|---|---|---|
| **Train** | Practice book | The model learns from these |
| **Validation** | Mock exam during the course | Checking progress after each epoch |
| **Test** | Final exam, kept secret | One final honest score for the manager |

- If the final exam uses the practice-book questions, a student who memorised the answers scores 100% without understanding anything.
- **Overfitting** = memorising the practice book. Sign: the train score keeps going up while the validation score goes down.

### ✅ Questions answered (2026-10-23)

**1. Which epoch should we keep, and why?**

| Epoch | Practice score (train) | Mock-exam score (validation) |
|---|---|---|
| 1 | 60% | 58% |
| 2 | 75% | 70% |
| 3 | 85% | 74% |
| 4 | 93% | 72% |
| 5 | 99% | 68% |

→ **Epoch 3**: validation is highest. After that, train keeps going up but validation goes down = memorising (overfitting). Epoch 5's 99% is on **train**, so it means nothing.

**2. A colleague says "Our model got 99% accuracy!" What is the first question to ask them?**

→ "99% on which data? Is it the **test** set, data the model has **never seen**?" If it's training data, the number is meaningless.

### Pose project (hands-on, 2026-10-23)

- `pose_demo.py`: YOLO11n-pose (2.9M weights, 6 MB) gives 17 body points per frame.
- One frame = 17 × 2 = 34 numbers. A clip = (frames, 17, 3): x, y, confidence. 10 s at 10 fps = (100, 17, 3).
- `record_clips.py` recorded 4 behaviours: sitting_still, head_down_still, pacing, waving.
- `build_dataset.py` split by time (last 2 clips = test, 1 before = validation), skipping clips with missing people. Result: 23 train, 4 validation, 8 test.
- `train_movement.py` trains a tiny GRU (8,292 weights). Train reached ~91%, validation jumped 25–75% (4 clips: 1 clip = 25 points), so it was overfitting.
- `evaluate_movement.py`: test 75% (6 of 8 clips). Not trustworthy with 8 clips and one person. Real fix = more data (more people, more days).
- Next: Step 7, export to ONNX and quantize (I should run the commands myself and predict first).

### Still to learn in Lesson 4

- Augmentation (flip, rotate, brightness, so the model can't memorise)
- Leakage, with a real example: our model's page claims 91%, but it was likely tested on copies of its training images
- Public datasets (FER-2013, RAF-DB, AffectNet), and why none of them look like jail cameras

---

## Important things for the manager (from all lessons)

- Facial emotion **alone** cannot reliably detect suicide risk. People at risk often look calm, and cell cameras often can't see faces. Behaviour detection (posture, stillness, self-harm movements, a covered camera) matters more.
- Every alert must go to a **guard who checks in person**. Physical rounds continue.
- The model's claimed 91% accuracy is probably too high. Realistic accuracy on standard benchmarks is around 65–75%, and on our cameras it's unknown until tested.
- **Legal review** is needed before collecting any detainee footage.
- Everything runs **on-prem**; no data leaves the site.
