# Character-Level Text Generator & Full-Stack Web App
### Built from scratch: Python + NumPy core. Zero ML frameworks. Minimalist Matte Web UI.

---

## What this is

A language model that reads Shakespeare and learns to generate new text — one character at a time.

No PyTorch. No TensorFlow. No Hugging Face. No shortcuts.

Every component is written manually:
tokenization, embeddings, forward pass, backpropagation, gradient descent, and temperature-controlled sampling.

Recently upgraded into a **Full-Stack Web Application** featuring a Flask API and a gorgeous, minimalist matte-themed frontend with live generation tracing.

> Anyone can call an API.
> This is what happens inside one, and how you deploy it.

---

## Results after training

```
Seed: "To be or not"

Low temperature (0.4) — safe and coherent:
To be or not the king, and so the world
shall see the sun of all the state of men

High temperature (1.2) — creative:
To be or not the beauty of the broken crown
that hath no fellow in the mortal eyes
```

Loss dropped from 4.17 to 1.78 over 100 epochs.

---

## How it works — 4 Phases

```
data.txt  ->  tokenizer.py  ->  train.py  ->  generate.py  ->  app.py (Web App)
              (Phase 1)         (Phase 2)      (Phase 3)         (Phase 4)
```

**Phase 1 — Tokenization:**
Read 1.1M characters, build 65-char vocabulary, encode to integers, create 892,000 training pairs, split 80/10/10.

**Phase 2 — Training:**
Embedding layer, hidden layer with ReLU, softmax output, cross-entropy loss, backprop via chain rule, gradient descent.

**Phase 3 — Generation:**
Load weights, feed seed text, sample next character from probability distribution, repeat. Temperature controls creativity.

**Phase 4 — Serving (Web UI):**
Flask API wraps the numpy model. A beautiful, minimalist matte UI built with vanilla JS and CSS connects to the API. It features a typewriter effect and a live sidebar that traces the model's exact thought process (`Context -> Selected Character`) step-by-step.

---

## Theory implemented (all from scratch)

| Concept | File |
|---|---|
| Character-level tokenization | tokenizer.py |
| Integer encoding | tokenizer.py |
| Sliding window X/Y pairs | tokenizer.py |
| Train / Val / Test split (80/10/10) | tokenizer.py |
| Dense embeddings | train.py |
| Forward pass  Z = XW + B | train.py |
| ReLU activation | train.py |
| Softmax | train.py |
| Cross-entropy loss | train.py |
| Backpropagation (chain rule) | train.py |
| Gradient descent | train.py |
| Gradient clipping | train.py |
| Temperature-controlled sampling | generate.py |
| RESTful API & Server | app.py |
| Asynchronous UI & Live Tracing | index.html |

---

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/character-text-generator
cd character-text-generator
pip install numpy flask flask-cors

# Download tiny_shakespeare.txt, rename to data.txt, put in this folder

python tokenizer.py    # Phase 1 — prepare data
python train.py        # Phase 2 — train (watch loss drop)
python generate.py     # Phase 3 — generate text interactively
```

**🚀 Run the Web App (Windows):**
Simply double-click `start_website.bat`. It boots up the local server and opens your browser to the minimalist dashboard. Or manually run `python app.py` and visit `http://localhost:5000`.

---

## Project files

```
text-generator/
├── app.py             Phase 4 — Flask API for serving the model
├── templates/
│   └── index.html     Phase 4 — Aesthetic Matte UI & Live Tracing
├── start_website.bat  Phase 4 — Quick launcher for Windows
├── tokenizer.py       Phase 1 — data preparation
├── train.py           Phase 2 — model and training loop
├── generate.py        Phase 3 — interactive terminal generation
├── vocab.json         auto-created by tokenizer.py
├── data_prepared.npz  auto-created by tokenizer.py
└── weights.npz        auto-created by train.py
```

---

## Tech stack

**Backend & AI Core:**
- Python 3.10+
- NumPy (Core AI math, absolutely zero ML frameworks)
- Flask & Flask-CORS (API Serving)

**Frontend UI:**
- Vanilla JavaScript (Async/Await Fetch)
- HTML5 & CSS3 (Custom Glassmorphism & Minimal Matte Design)
- Custom Grid & Flexbox layouts (No Bootstrap/Tailwind)

---

## What I learned

- Tokenization is just a lookup table. The learning happens after.
- The X/Y shift by one character is the entire training signal.
- Backprop is the chain rule applied backwards through each layer.
- Temperature is a one-line change that completely transforms output style.
- Watching loss drop in real time is the moment everything clicks.

---

## What comes next

- Attention mechanism visualizer
- RAG-powered document Q&A system
- Fine-tuned domain-specific model with LoRA

---

Studying Generative AI at Planet Beyond, Pakistan.
Building every concept from scratch before using frameworks.

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
