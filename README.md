# Character-Level Text Generator
### Built from scratch using Python + NumPy only + no ML frameworks

---

## What this is

A language model that learns to generate text one character at a time.  
Trained on Shakespeare. Built without PyTorch, TensorFlow, or any AI library.  
Every single component , tokenization, forward pass, backpropagation, gradient descent , is written manually in NumPy.

> This project exists to prove understanding, not just usage.  
> Anyone can call an API. This is what happens underneath one.

---

## Live demo

> Coming after training is complete , link will appear here

---

## How it works

The model reads a sequence of characters and predicts what comes next.  
It learns by making a prediction, measuring how wrong it was, and adjusting its weights.  
Repeat millions of times. The loss goes down. The text starts making sense.

```
Input  →  "To be or not to b"
Output →  "e"
```

Then it feeds that output back in, predicts the next, and keeps going.

---

## Theory implemented (from scratch)

Every concept below is implemented manually — not imported from a library.

| Concept | Where it appears |
|---|---|
| Character-level tokenization | `tokenizer.py` — Step 2 |
| Integer encoding (char → ID) | `tokenizer.py` — Step 3 |
| Dense embeddings | `train.py` — W_embed matrix |
| Forward pass  Z = XW + B | `train.py` — forward() |
| ReLU activation | `train.py` — np.maximum(0, z) |
| Softmax | `train.py` — probability distribution |
| Cross-entropy loss | `train.py` — loss() |
| Backpropagation (chain rule) | `train.py` — backward() |
| Gradient descent weight update | `train.py` — W = W - lr * grad |
| Train / Val / Test split (80/10/10) | `tokenizer.py` — Step 5 |
| Sliding window X/Y sequence pairs | `tokenizer.py` — Step 6 |
| Temperature-controlled sampling | `generate.py` |

---

## Project structure

```
text-generator/
│
├── data.txt               ← training text (Shakespeare)
├── tokenizer.py           ← Phase 1: data preparation
├── train.py               ← Phase 2: model + training loop
├── generate.py            ← Phase 3: text generation
│
├── data_prepared.npz      ← saved sequences (created by tokenizer.py)
├── vocab.json             ← character lookup tables
└── weights.npy            ← trained model weights (created by train.py)
```

---

## How to run it yourself

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/text-generator
cd text-generator
pip install numpy
```

**2. Prepare the data**
```bash
python tokenizer.py
```
Output: `data_prepared.npz` and `vocab.json`

**3. Train the model**
```bash
python train.py
```
You will see loss printing every 10 epochs. It should drop from ~4.1 to ~1.7.

**4. Generate text**
```bash
python generate.py
```
Enter any seed text. The model will generate 200 characters from it.

---

## What the terminal output looks like

**During training:**
```
Epoch 0,   Loss: 4.1732
Epoch 10,  Loss: 3.2451
Epoch 50,  Loss: 2.1034
Epoch 100, Loss: 1.7823
Training complete. Weights saved.
```

**During generation:**
```
Enter seed: To be or not
Generating...

To be or not the world is come to be
A man that shall not speak the state of men
The sun and shadow of the broken day
```

---

## What I learned building this

- How tokenization converts raw text into numbers a model can process
- Why we need embeddings instead of one-hot encoding (efficiency)
- How the forward pass produces a probability for every possible next character
- How backpropagation traces the error backwards through every layer using the chain rule
- Why learning rate matters — too high = exploding loss, too low = never converges
- How temperature controls the creativity vs coherence tradeoff during generation

---

## Tech stack

- Python 3.10+
- NumPy (the only dependency)
- No PyTorch. No TensorFlow. No Keras. No Hugging Face.

---

## What comes next (Phase 2 projects)

- [ ] Attention mechanism visualizer
- [ ] Full transformer block implementation
- [ ] RAG-powered document Q&A app

---

## About

Currently studying Generative AI at Planet Beyond, Pakistan.  
Building every concept from scratch before using frameworks.  

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)
