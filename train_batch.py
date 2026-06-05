import numpy as np
import json
import os
import time

np.random.seed(42)

# ------------------------------------------------
#  SETTINGS
# ------------------------------------------------
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 0.001  # Adam LR
EMBED_DIM = 32         # Keep same to match weights.npz structure
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

# ------------------------------------------------
#  LOAD THE DATA
# ------------------------------------------------
if not os.path.exists("data_prepared.npz") or not os.path.exists("vocab.json"):
    print("ERROR: Run tokenizer.py first.")
    exit(1)

data = np.load("data_prepared.npz")
X_train = data["X_train"]
Y_train = data["Y_train"]
X_val = data["X_val"]
Y_val = data["Y_val"]

with open("vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = vocab["vocab_size"]
seq_len = vocab["seq_len"]

# ------------------------------------------------
#  INITIALIZE WEIGHTS (or load existing if wanted, but clean init works great)
# ------------------------------------------------
W_embed = np.random.randn(vocab_size, EMBED_DIM) * 0.01
W_pos = np.random.randn(seq_len, EMBED_DIM) * 0.01
Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.01
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.01
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.01
W2 = np.random.randn(seq_len * EMBED_DIM, vocab_size) * 0.01
b2 = np.zeros(vocab_size)

# Initialize Adam Moments
m_W_embed, v_W_embed = np.zeros_like(W_embed), np.zeros_like(W_embed)
m_W_pos, v_W_pos = np.zeros_like(W_pos), np.zeros_like(W_pos)
m_Wq, v_Wq = np.zeros_like(Wq), np.zeros_like(Wq)
m_Wk, v_Wk = np.zeros_like(Wk), np.zeros_like(Wk)
m_Wv, v_Wv = np.zeros_like(Wv), np.zeros_like(Wv)
m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)

t = 0  # Adam timestep

# ------------------------------------------------
#  BATCH FORWARD PASS
# ------------------------------------------------
def forward_batch(x_ids):
    # x_ids: (BATCH_SIZE, seq_len)
    emb = W_embed[x_ids] + W_pos  # (BATCH_SIZE, seq_len, EMBED_DIM)
    
    Q = emb @ Wq  # (BATCH_SIZE, seq_len, EMBED_DIM)
    K = emb @ Wk  # (BATCH_SIZE, seq_len, EMBED_DIM)
    V = emb @ Wv  # (BATCH_SIZE, seq_len, EMBED_DIM)
    
    # Attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(EMBED_DIM)  # (BATCH_SIZE, seq_len, seq_len)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (BATCH_SIZE, seq_len, seq_len)
    
    O = np.matmul(A, V)  # (BATCH_SIZE, seq_len, EMBED_DIM)
    O_flat = O.reshape(x_ids.shape[0], -1)  # (BATCH_SIZE, seq_len * EMBED_DIM)
    
    z2 = O_flat @ W2 + b2  # (BATCH_SIZE, vocab_size)
    
    exp_z2 = np.exp(z2 - np.max(z2, axis=-1, keepdims=True))
    probs = exp_z2 / np.sum(exp_z2, axis=-1, keepdims=True)  # (BATCH_SIZE, vocab_size)
    
    return emb, Q, K, V, scores, A, O, O_flat, probs

# ------------------------------------------------
#  BATCH BACKWARD PASS
# ------------------------------------------------
def backward_batch(x_ids, target, emb, Q, K, V, A, O_flat, probs):
    # probs: (BATCH_SIZE, vocab_size)
    # target: (BATCH_SIZE,)
    batch_size = x_ids.shape[0]
    
    dz2 = probs.copy()
    dz2[np.arange(batch_size), target] -= 1.0
    dz2 /= batch_size  # Normalise by batch size
    
    dW2 = O_flat.T @ dz2
    db2 = np.sum(dz2, axis=0)
    
    dO_flat = dz2 @ W2.T
    dO = dO_flat.reshape(batch_size, seq_len, EMBED_DIM)
    
    dV = np.matmul(A.transpose(0, 2, 1), dO)
    dA = np.matmul(dO, V.transpose(0, 2, 1))
    
    # Gradient through softmax
    dscores = A * (dA - np.sum(A * dA, axis=-1, keepdims=True)) / np.sqrt(EMBED_DIM)
    
    dQ = np.matmul(dscores, K)
    dK = np.matmul(dscores.transpose(0, 2, 1), Q)
    
    dWq = np.sum(np.matmul(emb.transpose(0, 2, 1), dQ), axis=0)
    dWk = np.sum(np.matmul(emb.transpose(0, 2, 1), dK), axis=0)
    dWv = np.sum(np.matmul(emb.transpose(0, 2, 1), dV), axis=0)
    
    demb = np.matmul(dQ, Wq.T) + np.matmul(dK, Wk.T) + np.matmul(dV, Wv.T)
    
    # Embedding gradients
    dW_pos = np.sum(demb, axis=0)
    dW_embed = np.zeros_like(W_embed)
    np.add.at(dW_embed, x_ids, demb)
    
    return dW_embed, dW_pos, dWq, dWk, dWv, dW2, db2

# ------------------------------------------------
#  ADAM OPTIMIZER STEP
# ------------------------------------------------
def adam_step(dW_embed, dW_pos, dWq, dWk, dWv, dW2, db2):
    global W_embed, W_pos, Wq, Wk, Wv, W2, b2, t
    global m_W_embed, v_W_embed, m_W_pos, v_W_pos, m_Wq, v_Wq, m_Wk, v_Wk, m_Wv, v_Wv, m_W2, v_W2, m_b2, v_b2
    
    t += 1
    
    # Clip gradients
    clip_val = 5.0
    dW_embed = np.clip(dW_embed, -clip_val, clip_val)
    dW_pos = np.clip(dW_pos, -clip_val, clip_val)
    dWq = np.clip(dWq, -clip_val, clip_val)
    dWk = np.clip(dWk, -clip_val, clip_val)
    dWv = np.clip(dWv, -clip_val, clip_val)
    dW2 = np.clip(dW2, -clip_val, clip_val)
    db2 = np.clip(db2, -clip_val, clip_val)
    
    # Update running moments
    def update_param(W, dW, m, v):
        m = BETA1 * m + (1 - BETA1) * dW
        v = BETA2 * v + (1 - BETA2) * (dW ** 2)
        m_hat = m / (1 - BETA1 ** t)
        v_hat = v / (1 - BETA2 ** t)
        W -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPS)
        return m, v
        
    m_W_embed, v_W_embed = update_param(W_embed, dW_embed, m_W_embed, v_W_embed)
    m_W_pos, v_W_pos = update_param(W_pos, dW_pos, m_W_pos, v_W_pos)
    m_Wq, v_Wq = update_param(Wq, dWq, m_Wq, v_Wq)
    m_Wk, v_Wk = update_param(Wk, dWk, m_Wk, v_Wk)
    m_Wv, v_Wv = update_param(Wv, dWv, m_Wv, v_Wv)
    m_W2, v_W2 = update_param(W2, dW2, m_W2, v_W2)
    m_b2, v_b2 = update_param(b2, db2, m_b2, v_b2)

# ------------------------------------------------
#  EVALUATION
# ------------------------------------------------
def evaluate(num_samples=1000):
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    total_loss = 0.0
    # Process in mini-batches to speed up evaluation
    eval_batch_size = 256
    for i in range(0, num_samples, eval_batch_size):
        batch_idx = indices[i:i+eval_batch_size]
        x_batch = X_val[batch_idx]
        y_batch = Y_val[batch_idx][:, -1]
        
        _, _, _, _, _, _, _, _, probs = forward_batch(x_batch)
        loss = -np.log(probs[np.arange(len(batch_idx)), y_batch] + 1e-9)
        total_loss += np.sum(loss)
    return total_loss / num_samples

# ------------------------------------------------
#  TRAINING LOOP
# ------------------------------------------------
print("Starting Vectorized Batch Training using Adam Optimizer...")
best_val_loss = float("inf")

num_train_samples = min(150000, len(X_train))

for epoch in range(EPOCHS):
    t0 = time.time()
    # Shuffle dataset
    indices = np.random.permutation(num_train_samples)
    epoch_losses = []
    
    for step_idx in range(0, num_train_samples, BATCH_SIZE):
        batch_idx = indices[step_idx : step_idx + BATCH_SIZE]
        if len(batch_idx) < BATCH_SIZE:
            continue
            
        x_batch = X_train[batch_idx]
        y_batch = Y_train[batch_idx][:, -1]  # Next token is target
        
        emb, Q, K, V, scores, A, O, O_flat, probs = forward_batch(x_batch)
        loss = -np.log(probs[np.arange(BATCH_SIZE), y_batch] + 1e-9)
        epoch_losses.append(np.mean(loss))
        
        grads = backward_batch(x_batch, y_batch, emb, Q, K, V, A, O_flat, probs)
        adam_step(*grads)
        
    avg_train_loss = np.mean(epoch_losses)
    avg_val_loss = evaluate()
    t1 = time.time()
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {t1-t0:.2f}s")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        np.savez("weights.npz",
                 W_embed=W_embed, W_pos=W_pos, 
                 Wq=Wq, Wk=Wk, Wv=Wv, W2=W2, b2=b2)
        print(" -> Saved new best weights to weights.npz!")

print(f"Training Complete! Best Validation Loss achieved: {best_val_loss:.4f}")
