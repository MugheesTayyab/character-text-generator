import numpy as np
import time

BATCH_SIZE = 128
SEQ_LEN = 25
EMBED_DIM = 32
VOCAB_SIZE = 65

W_embed = np.random.randn(VOCAB_SIZE, EMBED_DIM)
W_pos = np.random.randn(SEQ_LEN, EMBED_DIM)
Wq = np.random.randn(EMBED_DIM, EMBED_DIM)
Wk = np.random.randn(EMBED_DIM, EMBED_DIM)
Wv = np.random.randn(EMBED_DIM, EMBED_DIM)
W2 = np.random.randn(SEQ_LEN * EMBED_DIM, VOCAB_SIZE)
b2 = np.zeros(VOCAB_SIZE)

x_ids = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
target = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE,))

t0 = time.time()
for _ in range(100): # 12800 samples
    emb = W_embed[x_ids] + W_pos
    Q = emb @ Wq
    K = emb @ Wk
    V = emb @ Wv
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(EMBED_DIM)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    O = np.matmul(A, V)
    O_flat = O.reshape(BATCH_SIZE, -1)
    z2 = O_flat @ W2 + b2
    
    exp_z2 = np.exp(z2 - np.max(z2, axis=-1, keepdims=True))
    probs = exp_z2 / np.sum(exp_z2, axis=-1, keepdims=True)
    
    # backward
    dz2 = probs.copy()
    dz2[np.arange(BATCH_SIZE), target] -= 1.0
    dz2 /= BATCH_SIZE # mean loss
    
    dW2 = O_flat.T @ dz2
    db2 = np.sum(dz2, axis=0)
    
    dO_flat = dz2 @ W2.T
    dO = dO_flat.reshape(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    
    dV = np.matmul(A.transpose(0, 2, 1), dO)
    dA = np.matmul(dO, V.transpose(0, 2, 1))
    
    dscores = A * (dA - np.sum(A * dA, axis=-1, keepdims=True)) / np.sqrt(EMBED_DIM)
    
    dQ = np.matmul(dscores, K)
    dK = np.matmul(dscores.transpose(0, 2, 1), Q)
    
    dWq = np.sum(np.matmul(emb.transpose(0, 2, 1), dQ), axis=0)
    dWk = np.sum(np.matmul(emb.transpose(0, 2, 1), dK), axis=0)
    dWv = np.sum(np.matmul(emb.transpose(0, 2, 1), dV), axis=0)
    
    demb = np.matmul(dQ, Wq.T) + np.matmul(dK, Wk.T) + np.matmul(dV, Wv.T)
    
t1 = time.time()
print(f"Time for 12800 samples: {t1 - t0:.3f}s")
