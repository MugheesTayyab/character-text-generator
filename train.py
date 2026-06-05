# ================================================
#  PHASE 2 — ATTENTION IS ALL YOU NEED (TRANSFORMER)
#  train.py
#  Only using: Python + NumPy (no ML libraries)
# ================================================

import numpy as np
import json
import os

np.random.seed(42)

# ------------------------------------------------
#  SETTINGS
# ------------------------------------------------
EPOCHS        = 2
LEARNING_RATE = 0.01
EMBED_DIM     = 32      # d_model
PRINT_EVERY   = 2

# ------------------------------------------------
#  STEP 1 — LOAD THE PREPARED DATA
# ------------------------------------------------
if not os.path.exists("data_prepared.npz") or not os.path.exists("vocab.json"):
    print("ERROR: Run tokenizer.py first.")
    exit()

data    = np.load("data_prepared.npz", allow_pickle=True)
X_train = data["X_train"]
Y_train = data["Y_train"]
X_val   = data["X_val"]
Y_val   = data["Y_val"]

with open("vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = vocab["vocab_size"]
seq_len    = vocab["seq_len"]
ix_to_char = { int(k): v for k, v in vocab["ix_to_char"].items() }

# Use a larger subset so it actually learns to predict real words!
# 50,000 samples will take a bit longer but the predictions will be much better.
MAX_SAMPLES = min(50000, len(X_train))

# ------------------------------------------------
#  STEP 2 — INITIALIZE TRANSFORMER WEIGHTS
# ------------------------------------------------
# Token and Positional Embeddings
W_embed = np.random.randn(vocab_size, EMBED_DIM) * 0.01
W_pos   = np.random.randn(seq_len, EMBED_DIM) * 0.01

# Self-Attention Weights (Queries, Keys, Values)
Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.01
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.01
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.01

# Final output projection
W2 = np.random.randn(seq_len * EMBED_DIM, vocab_size) * 0.01
b2 = np.zeros(vocab_size)

# ------------------------------------------------
#  STEP 3 — FORWARD PASS (SELF-ATTENTION)
#  "Attention Is All You Need" paper implementation.
#  Why it works: Instead of looking at characters
#  one by one, Query (Q) matches with Key (K) to 
#  find which past characters are relevant, then 
#  extracts their Value (V).
# ------------------------------------------------
def forward(x_ids):
    # 1. Embeddings + Position (so it knows order)
    emb = W_embed[x_ids] + W_pos

    # 2. Linear projections for Q, K, V
    Q = emb @ Wq
    K = emb @ Wk
    V = emb @ Wv

    # 3. Scaled Dot-Product Attention
    # scores = Q * K^T / sqrt(d_k)
    scores = (Q @ K.T) / np.sqrt(EMBED_DIM)
    
    # Softmax to get attention weights (sum to 1)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Context vectors = Attention * Values
    O = A @ V

    # 5. Flatten and project to vocabulary size
    O_flat = O.flatten()
    z2 = O_flat @ W2 + b2

    # 6. Output Softmax probabilities
    exp_z2 = np.exp(z2 - np.max(z2))
    probs  = exp_z2 / exp_z2.sum()

    return emb, Q, K, V, scores, A, O, O_flat, probs


def compute_loss(probs, target_ix):
    return -np.log(probs[target_ix] + 1e-9)


# ------------------------------------------------
#  STEP 4 — BACKWARD PASS
#  Applying the chain rule exactly backwards through
#  the Self-Attention mechanism.
# ------------------------------------------------
def backward(emb, Q, K, V, scores, A, O, O_flat, probs, target_ix, x_ids):
    # Output layer gradient
    dz2 = probs.copy()
    dz2[target_ix] -= 1.0

    dW2 = O_flat[:, None] @ dz2[None, :]
    db2 = dz2

    # Gradient through flattening
    dO_flat = dz2 @ W2.T
    dO = dO_flat.reshape(seq_len, EMBED_DIM)

    # Gradient through Attention: O = A @ V
    dV = A.T @ dO
    dA = dO @ V.T

    # Gradient through Softmax (Attention weights A)
    dscores = A * (dA - np.sum(A * dA, axis=-1, keepdims=True))
    dscores = dscores / np.sqrt(EMBED_DIM)

    # Gradient through Scaled Dot Product: scores = Q @ K.T
    dQ = dscores @ K
    dK = dscores.T @ Q

    # Gradient through Q, K, V projections
    dWq = emb.T @ dQ
    dWk = emb.T @ dK
    dWv = emb.T @ dV

    # Gradient into embeddings
    demb = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    return dW2, db2, dWq, dWk, dWv, demb, x_ids


def update_weights(dW2, db2, dWq, dWk, dWv, demb, x_ids):
    global W2, b2, Wq, Wk, Wv, W_embed, W_pos

    # Gradient clipping
    clip = 5.0
    dW2  = np.clip(dW2,  -clip, clip)
    dWq  = np.clip(dWq,  -clip, clip)
    dWk  = np.clip(dWk,  -clip, clip)
    dWv  = np.clip(dWv,  -clip, clip)
    demb = np.clip(demb, -clip, clip)

    # Gradient descent update
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2
    Wq -= LEARNING_RATE * dWq
    Wk -= LEARNING_RATE * dWk
    Wv -= LEARNING_RATE * dWv

    # Update embeddings and positional encodings
    W_embed[x_ids] -= LEARNING_RATE * demb
    W_pos -= LEARNING_RATE * demb


def get_val_loss(num_samples=200):
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    total_loss = 0
    for i in indices:
        _, _, _, _, _, _, _, _, probs = forward(X_val[i])
        total_loss += compute_loss(probs, Y_val[i][-1])
    return total_loss / num_samples


# ------------------------------------------------
#  TRAINING LOOP
# ------------------------------------------------
print("\nStarting Transformer Training...")
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    total_loss = 0
    indices = np.random.permutation(MAX_SAMPLES)

    for i in indices:
        x_ids  = X_train[i]
        target = Y_train[i][-1]

        emb, Q, K, V, scores, A, O, O_flat, probs = forward(x_ids)
        loss = compute_loss(probs, target)
        total_loss += loss

        grads = backward(emb, Q, K, V, scores, A, O, O_flat, probs, target, x_ids)
        update_weights(*grads)

    avg_train_loss = total_loss / MAX_SAMPLES
    avg_val_loss   = get_val_loss()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        np.savez("weights.npz",
                 W_embed=W_embed, W_pos=W_pos, 
                 Wq=Wq, Wk=Wk, Wv=Wv, W2=W2, b2=b2)

    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch:>3} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

print("Training Complete! Weights saved to weights.npz")