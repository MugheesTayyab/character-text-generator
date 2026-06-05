import numpy as np
import json
import os

if not os.path.exists("weights.npz") or not os.path.exists("vocab.json"):
    print("ERROR: Run tokenizer.py then train.py first.")
    exit()

with open("vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = vocab["vocab_size"]
seq_len    = vocab["seq_len"]
char_to_ix = vocab["char_to_ix"]
ix_to_char = { int(k): v for k, v in vocab["ix_to_char"].items() }

weights = np.load("weights.npz")
W_embed = weights["W_embed"]
W_pos   = weights["W_pos"]
Wq      = weights["Wq"]
Wk      = weights["Wk"]
Wv      = weights["Wv"]
W2      = weights["W2"]
b2      = weights["b2"]
EMBED_DIM = W_embed.shape[1]

def forward(x_ids):
    emb = W_embed[x_ids] + W_pos
    Q = emb @ Wq
    K = emb @ Wk
    V = emb @ Wv
    
    scores = (Q @ K.T) / np.sqrt(EMBED_DIM)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    O = A @ V
    z2 = O.flatten() @ W2 + b2
    exp_z2 = np.exp(z2 - np.max(z2))
    probs = exp_z2 / exp_z2.sum()
    return probs, A

def generate(seed_text, num_chars=300, temperature=0.8):
    context = [char_to_ix.get(ch, 0) for ch in seed_text]
    while len(context) < seq_len:
        context = [0] + context
    context = context[-seq_len:]
    result = seed_text
    
    for _ in range(num_chars):
        x_ids = np.array(context)
        probs, _ = forward(x_ids)
        
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs - np.max(probs))
        probs = probs / probs.sum()
        
        next_ix = np.random.choice(vocab_size, p=probs)
        result += ix_to_char[next_ix]
        context = context[1:] + [next_ix]
        
    return result

print("-" * 50)
print(f"  INTERACTIVE TRANSFORMER GENERATION")
print("-" * 50)
while True:
    seed = input("  Your seed: ").strip()
    if seed.lower() in ("quit", "exit", "q"): break
    if not seed: continue
    
    out = generate(seed, 200, 0.8)
    print(f"\n{out}\n")
    print("-" * 50)