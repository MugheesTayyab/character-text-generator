# 🧠 MatteMind | Character-Level Transformer
### A from-scratch Transformer language model built using Python + NumPy core. Zero ML frameworks. Minimalist Matte Web UI.

---

[![Vercel Deployment](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel&logoColor=white)](https://vercel.com)
[![Language](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Library](https://img.shields.io/badge/NumPy-1.20%2B-darkblue?logo=numpy&logoColor=white)](https://numpy.org/)
[![Web Framework](https://img.shields.io/badge/Flask-2.0%2B-lightgrey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

This repository implements a Character-Level Language Model designed and built completely from scratch using **pure Python and NumPy**. It contains no PyTorch, TensorFlow, Hugging Face, or other high-level machine learning frameworks. 

Every single mathematical operation—including tokenization, character embeddings, forward passes, scaled dot-product attention, temperature-controlled sampling, backpropagation, and gradient updates—has been manually implemented.

To showcase the model, the project includes a **Full-Stack Web Application** featuring a Flask API server and a custom-designed, matte-themed glassmorphic web dashboard with live generation tracing and an interactive **Transformer Math Lab**.

---

## 🧠 Recruiter Technical Deep-Dive (First Principles)

This project was built to master the underlying mathematical and engineering concepts behind Large Language Models (LLMs) by implementing them from scratch without library abstractions.

### 1. Causal Scaled Self-Attention Mechanics
We project our input sequence embeddings $X \in \mathbb{R}^{T \times D}$ into Queries ($Q$), Keys ($K$), and Values ($V$) using weight matrices $W_q, W_k, W_v \in \mathbb{R}^{D \times D}$:
$$Q = XW_q, \quad K = XW_k, \quad V = XW_v$$
The attention weights matrix $A$ represents token-to-token semantic alignment, computed as:
$$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right), \quad \text{Attention}(Q, K, V) = A V$$

*   **The Scaling Factor ($\frac{1}{\sqrt{d_k}}$):** When the embedding dimension $d_k$ is large, dot products grow large in magnitude, pushing the Softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ scales the variance of the products to $1.0$, preventing vanishing gradients during backpropagation.

### 2. Analytical Backpropagation (Derivation & Implementation)
Without automatic differentiation tools, we derived and coded the gradients analytically. The backpropagation follows the chain rule backwards from the loss $L$ (Cross-Entropy):
*   **Logits Gradient:** $\frac{\partial L}{\partial z_2} = \text{probs} - y \quad$ (where $y$ is the one-hot target)
*   **Output Projection:** $dW_2 = O_{flat}^T \cdot dz_2$
*   **Value Matrix Gradient:** $dV = A^T \cdot dO \quad$ (where $dO$ is reshaped from the projection layer gradient)
*   **Attention Coefficients Gradient:** $dA = dO \cdot V^T$
*   **Softmax Backpropagation:** Gradients must be backpropagated through the Softmax function:
    $$d\text{scores} = \frac{1}{\sqrt{d_k}} A \odot \left(dA - \sum (A \odot dA)\right)$$
*   **Query and Key Gradients:**
    $$dQ = d\text{scores} \cdot K, \quad dK = d\text{scores}^T \cdot Q$$
*   **Weight Projections:**
    $$dW_q = X^T \cdot dQ, \quad dW_k = X^T \cdot dK, \quad dW_v = X^T \cdot dV$$

*   **Embedding Gradients Accumulation:** Because embeddings are discrete index lookups, gradients are accumulated across matching indices using vectorized accumulation:
    ```python
    np.add.at(dW_embed, x_ids, demb)
    ```

### 3. Performance Engineering & Vectorization
*   **60x Speedup via Vectorization:** We refactored single-sample training loops into vectorized mini-batches (batch size $128$). This allowed NumPy to compile operations into parallelized BLAS/LAPACK routines, dropping training epoch times from minutes to seconds.
*   **Adam Optimizer from Scratch:** Implemented running first ($m$) and second ($v$) moment estimates of gradients with bias corrections:
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
    This allowed fast, stable training convergence, achieving a final validation loss of **2.20**.

---

## 🌟 Key Features

*   **From-Scratch Architecture:** Every layer and equation is written in pure NumPy.
*   **Interactive Math Lab:** Collapsible first-principles dashboard explaining the self-attention, backpropagation, and vectorization equations.
*   **Aesthetic Matte UI:** A premium, custom design system utilizing warm Copper, Amber, and stone elements with smooth glassmorphic blur filters and hover translations.
*   **Telemetry Tracing & Predictions:** Visual bars showing probability distributions over character selections and dynamic attention weights.
*   **Dynamic Inference Control:** Sliders to control Temperature, Top-K, and Top-P (Nucleus) sampling options in real time.

---

## 📸 User Interface Showcase

Below are walk-through screenshots of the interactive dashboard:

<div align="center">
  <img src="1.PNG" width="45%" alt="Dashboard Overview" />
  <img src="2.PNG" width="45%" alt="Interactive Settings" />
  <br />
  <img src="3.PNG" width="45%" alt="Live Token Tracing" />
  <img src="4.PNG" width="45%" alt="Typewriter Output View" />
</div>

---

## ⚙️ How It Works (The 4-Phase Pipeline)

```
data.txt  ──>  tokenizer.py  ──>  train_batch.py  ──>  generate.py  ──>  app.py (Web App)
               (Phase 1)          (Phase 2)          (Phase 3)         (Phase 4)
```

1. **Phase 1 — Tokenization (`tokenizer.py`):**
   Reads raw text (e.g., Shakespeare), builds a character vocabulary map, encodes characters to integers, and prepares training sliding windows (X/Y pairs).
2. **Phase 2 — Vectorized Training (`train_batch.py`):**
   Trains character embeddings, Query/Key/Value projections, dense projection layers, cross-entropy loss, manual chain-rule backpropagation, Adam optimizations, and saves the trained parameters to `weights.npz` and vocabulary mappings to `vocab.json`.
3. **Phase 3 — CLI Generation (`generate.py`):**
   A simple terminal application to load the trained weights and vocabulary, allowing interactive text generation with custom temperature.
4. **Phase 4 — Web Serving (`app.py` & `templates/`):**
   A Flask web server hosting a RESTful endpoint `/api/generate` that performs inference, filters probabilities using Top-K and Top-P, and returns output characters alongside trace logs to the front-end dashboard.

---

## 🔬 Project Files Directory

```
MatteMind/
├── app.py             Phase 4 — Flask API serving the model with Top-K/Top-P options
├── templates/
│   └── index.html     Phase 4 — Custom Matte UI dashboard with Telemetry & Math Lab
├── train_batch.py     Phase 2 — Vectorized Adam batch training script (fast)
├── tokenizer.py       Phase 1 — data preparation & vocab mapping
├── train.py           Phase 2 — Single-sample baseline training loop (educational)
├── generate.py        Phase 3 — interactive terminal generation
├── start_website.bat  Phase 4 — Quick launcher for Windows
├── vocab.json         auto-created vocabulary maps
├── weights.npz        trained high-accuracy weights (Val Loss: 2.20)
└── data_prepared.npz  auto-created npz arrays
```

---

## 🚀 Local Installation & Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/MugheesTayyab/character-text-generator.git
cd character-text-generator
```

### 2. Install dependencies
Ensure you have Python 3.10+ installed. Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Training & Running
The repository includes pre-trained model weights (`weights.npz`) and vocabulary maps (`vocab.json`) so you can run the web server immediately.

If you wish to retrain the model on your own dataset:
1. Place your training text in the root directory and name it `data.txt`.
2. Prepare the dataset:
   ```bash
   python tokenizer.py
   ```
3. Run the high-speed vectorized training:
   ```bash
   python train_batch.py
   ```
4. Test generator in the terminal:
   ```bash
   python generate.py
   ```

### 4. Boot the Web App
Simply run the Flask web application:
```bash
python app.py
```
Open your web browser and navigate to `http://localhost:5000` to interact with the UI dashboard. On Windows, you can also double-click `start_website.bat` to launch automatically.

---

## ⚡ Deployment to Vercel

The project is fully pre-configured for deployment as a Python serverless application on Vercel:

1. **Push your code to GitHub** (make sure `requirements.txt` and `vercel.json` are present in the root).
2. Go to your **[Vercel Dashboard](https://vercel.com)**.
3. Click **"Add New Project"** and select/import the **`character-text-generator`** repository.
4. Keep the default options and click **"Deploy"**. Vercel will automatically resolve your builds using `@vercel/python` and publish your app.

---

## 🎓 Academic Context
* **Studies:** Generative AI at Planet Beyond, Pakistan.
* **Objective:** Developing a deep, first-principles understanding of Transformer mathematics and neural network backpropagation before utilizing commercial frameworks.

Developed with 💻 by [Muhammad Mughees Tayyab](https://github.com/MugheesTayyab).