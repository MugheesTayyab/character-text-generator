# ================================================
#  PHASE 2 — MODEL AND TRAINING LOOP
#  train.py
#  Only using: Python + NumPy (no ML libraries)
# ================================================
#  HOW TO RUN:
#    Step 1: run tokenizer.py first (creates data files)
#    Step 2: python train.py
#    Step 3: wait — training prints loss every 10 epochs
#
#  FILES NEEDED:  data_prepared.npz,  vocab.json
#  FILE CREATED:  weights.npy  (your trained model)
# ================================================

import numpy as np
import json
import os

np.random.seed(42)   # makes results reproducible


# ------------------------------------------------
#  SETTINGS — tweak these to experiment
# ------------------------------------------------

EPOCHS        = 100      # how many times to loop over the data (reduced for faster testing)
LEARNING_RATE = 0.01    # step size for weight updates (smaller = safer)
EMBED_DIM     = 32      # size of each character's embedding vector
HIDDEN_SIZE   = 128     # number of neurons in the hidden layer
PRINT_EVERY   = 2       # print loss every N epochs


# ------------------------------------------------
#  STEP 1 — LOAD THE PREPARED DATA
#  tokenizer.py already saved everything to disk.
#  We just load it here.
# ------------------------------------------------

if not os.path.exists("data_prepared.npz") or not os.path.exists("vocab.json"):
    print("ERROR: Run tokenizer.py first to create the data files.")
    exit()

data    = np.load("data_prepared.npz", allow_pickle=True)
X_train = data["X_train"]    # shape: (num_sequences, seq_len)
Y_train = data["Y_train"]    # shape: (num_sequences, seq_len)
X_val   = data["X_val"]
Y_val   = data["Y_val"]

with open("vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = vocab["vocab_size"]
seq_len    = vocab["seq_len"]
ix_to_char = { int(k): v for k, v in vocab["ix_to_char"].items() }

print(f"Data loaded!")
print(f"  Vocab size      : {vocab_size}")
print(f"  Sequence length : {seq_len}")
print(f"  Train sequences : {len(X_train):,}")
print(f"  Val sequences   : {len(X_val):,}")


# ------------------------------------------------
#  STEP 2 — INITIALIZE THE MODEL WEIGHTS
#
#  Our model has 3 layers:
#
#  [Input: character IDs]
#       ↓
#  Embedding layer  (W_embed)  — turns each char ID into a vector
#       ↓
#  Hidden layer     (W1, b1)   — learns patterns in the sequence
#       ↓
#  Output layer     (W2, b2)   — gives a score for each character
#       ↓
#  [Output: probability of next character]
#
#  We multiply weights by 0.01 to keep initial values small.
#  Large initial values → exploding gradients → NaN loss.
# ------------------------------------------------

# Embedding: one row per character, each row is a vector of size EMBED_DIM
W_embed = np.random.randn(vocab_size, EMBED_DIM) * 0.01

# Hidden layer: takes flattened embeddings, outputs HIDDEN_SIZE values
# Input size = seq_len * EMBED_DIM  (all embeddings concatenated)
W1 = np.random.randn(seq_len * EMBED_DIM, HIDDEN_SIZE) * 0.01
b1 = np.zeros(HIDDEN_SIZE)

# Output layer: takes hidden layer output, gives score for each char in vocab
W2 = np.random.randn(HIDDEN_SIZE, vocab_size) * 0.01
b2 = np.zeros(vocab_size)

print(f"\nModel initialized!")
print(f"  Embedding matrix : {W_embed.shape}  (vocab_size x embed_dim)")
print(f"  Hidden layer W1  : {W1.shape}")
print(f"  Output layer W2  : {W2.shape}")

# Count total parameters (weights) in the model
total_params = W_embed.size + W1.size + b1.size + W2.size + b2.size
print(f"  Total parameters : {total_params:,}")


# ------------------------------------------------
#  STEP 3 — FORWARD PASS
#
#  THEORY: Z = XW + B  (from your notes)
#
#  This function takes a sequence of character IDs,
#  runs them through all 3 layers, and returns
#  a probability for every character in the vocabulary.
#
#  The probabilities tell us: "given this sequence,
#  how likely is each character to come next?"
# ------------------------------------------------

def forward(x_ids):
    # --- Embedding lookup ---
    # Replace each character ID with its embedding vector
    # x_ids shape: (seq_len,)  →  emb shape: (seq_len, EMBED_DIM)
    emb = W_embed[x_ids]

    # Flatten all embeddings into one long vector
    # (seq_len, EMBED_DIM) → (seq_len * EMBED_DIM,)
    emb_flat = emb.flatten()

    # --- Hidden layer ---
    # Z = XW + B  (the core equation from your notes)
    z1 = emb_flat @ W1 + b1        # raw hidden layer output

    # ReLU activation — turns negatives to 0, keeps positives
    # This adds non-linearity so the model can learn complex patterns
    a1 = np.maximum(0, z1)         # ReLU: max(0, x)

    # --- Output layer ---
    z2 = a1 @ W2 + b2              # raw scores for each character (logits)

    # Softmax — converts raw scores to probabilities that sum to 1
    # We subtract max(z2) first for numerical stability (prevents overflow)
    exp_z2 = np.exp(z2 - np.max(z2))
    probs  = exp_z2 / exp_z2.sum()  # each value is between 0 and 1

    # Return everything — we need them all in the backward pass
    return emb, emb_flat, z1, a1, probs


# ------------------------------------------------
#  STEP 4 — LOSS FUNCTION
#
#  THEORY: Cross-Entropy Loss (from your notes)
#
#  We measure how wrong the model's prediction is.
#  If the model says the correct character has 0.9 probability
#  → loss is small (good prediction)
#  If the model says the correct character has 0.01 probability
#  → loss is large (bad prediction)
#
#  Formula: loss = -log(probability of correct character)
# ------------------------------------------------

def compute_loss(probs, target_ix):
    # Add small number (1e-9) to avoid log(0) which is undefined
    return -np.log(probs[target_ix] + 1e-9)


# ------------------------------------------------
#  STEP 5 — BACKWARD PASS (BACKPROPAGATION)
#
#  THEORY: Chain Rule  dL/dw = dL/da × da/dz × dz/dw
#
#  We trace the error backwards through the network.
#  At each layer we calculate: "how much did this weight
#  contribute to the mistake?"
#
#  This gives us the gradient — the direction to nudge
#  each weight to make the model less wrong next time.
# ------------------------------------------------

def backward(emb_flat, z1, a1, probs, target_ix, x_ids):

    # --- Output layer gradient ---
    # Gradient of softmax + cross-entropy combined is very simple:
    # subtract 1 from the probability of the correct character
    dz2 = probs.copy()
    dz2[target_ix] -= 1.0          # this is the chain rule result for softmax + CE

    # Gradient for W2 and b2
    dW2 = a1[:, None] @ dz2[None, :]   # outer product
    db2 = dz2

    # --- Hidden layer gradient ---
    # Pass gradient back through W2
    da1 = dz2 @ W2.T

    # Pass gradient back through ReLU
    # ReLU derivative: 1 where input > 0, 0 where input <= 0
    dz1 = da1 * (z1 > 0)

    # Gradient for W1 and b1
    dW1 = emb_flat[:, None] @ dz1[None, :]
    db1 = dz1

    # --- Embedding layer gradient ---
    demb_flat = dz1 @ W1.T
    demb      = demb_flat.reshape(seq_len, EMBED_DIM)

    return dW1, db1, dW2, db2, demb, x_ids


# ------------------------------------------------
#  STEP 6 — WEIGHT UPDATE (GRADIENT DESCENT)
#
#  THEORY: W_new = W_old - (learning_rate × gradient)
#
#  We nudge every weight in the direction that
#  reduces the loss. The learning rate controls
#  how big that nudge is.
#
#  We also clip gradients to prevent them from
#  becoming too large (exploding gradient problem).
# ------------------------------------------------

def update_weights(dW1, db1, dW2, db2, demb, x_ids):
    global W1, b1, W2, b2, W_embed

    # Clip gradients — prevents exploding gradients
    # If any gradient > 5.0 or < -5.0 we cap it
    clip = 5.0
    dW1  = np.clip(dW1,  -clip, clip)
    dW2  = np.clip(dW2,  -clip, clip)
    demb = np.clip(demb, -clip, clip)

    # Update each weight:  W = W - learning_rate * gradient
    W1      -= LEARNING_RATE * dW1
    b1      -= LEARNING_RATE * db1
    W2      -= LEARNING_RATE * dW2
    b2      -= LEARNING_RATE * db2

    # Only update embeddings for characters that appeared in this sequence
    W_embed[x_ids] -= LEARNING_RATE * demb


# ------------------------------------------------
#  STEP 7 — VALIDATION LOSS
#
#  We check loss on data the model has NEVER trained on.
#  If train loss drops but val loss rises = overfitting.
#  Both should drop together = healthy training.
# ------------------------------------------------

def get_val_loss(num_samples=200):
    # Check on a random subset of validation data (faster than full val set)
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    total_loss = 0
    for i in indices:
        x_ids  = X_val[i]
        target = Y_val[i][-1]
        _, _, _, _, probs = forward(x_ids)
        total_loss += compute_loss(probs, target)
    return total_loss / num_samples


# ------------------------------------------------
#  STEP 8 — TRAINING LOOP
#
#  This is the core loop that runs every epoch:
#
#  1. Forward pass  → get predictions
#  2. Compute loss  → measure how wrong
#  3. Backward pass → find which weights caused the error
#  4. Update weights → nudge weights to reduce error
#  5. Repeat
#
#  We shuffle the training data each epoch so the
#  model doesn't memorize the order of sequences.
# ------------------------------------------------

print(f"\nStarting training...")
print(f"  Epochs        : {EPOCHS}")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  Hidden size   : {HIDDEN_SIZE}")
print(f"  Embed dim     : {EMBED_DIM}")
print("-" * 45)
print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}")
print("-" * 45)

best_val_loss   = float('inf')
train_losses    = []
val_losses      = []

# Use a small subset for faster training on CPU
# Remove this limit once you confirm training works
MAX_SAMPLES = min(1000, len(X_train))

for epoch in range(EPOCHS):
    total_loss = 0

    # Shuffle indices so we see data in different order each epoch
    indices = np.random.permutation(MAX_SAMPLES)

    for i in indices:
        x_ids  = X_train[i]       # input sequence (25 character IDs)
        target = Y_train[i][-1]   # the ONE character we predict (last in Y)

        # 1. Forward pass — get probabilities
        emb, emb_flat, z1, a1, probs = forward(x_ids)

        # 2. Compute loss — how wrong were we?
        loss = compute_loss(probs, target)
        total_loss += loss

        # 3. Backward pass — compute gradients
        grads = backward(emb_flat, z1, a1, probs, target, x_ids)

        # 4. Update weights — nudge in right direction
        update_weights(*grads)

    # Calculate average losses for this epoch
    avg_train_loss = total_loss / MAX_SAMPLES
    avg_val_loss   = get_val_loss()

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save the best model weights seen so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        np.savez("weights.npz",
                 W_embed=W_embed, W1=W1, b1=b1, W2=W2, b2=b2)

    # Print progress every PRINT_EVERY epochs
    if epoch % PRINT_EVERY == 0:
        marker = " <- best" if avg_val_loss == best_val_loss else ""
        print(f"  {epoch:>6}  {avg_train_loss:>10.4f}  {avg_val_loss:>10.4f}{marker}")

print(f"{'─' * 45}".replace('─', '-'))
print(f"\nTraining complete!")
print(f"  Best val loss : {best_val_loss:.4f}")
print(f"  Weights saved : weights.npz")


# ------------------------------------------------
#  STEP 9 — QUICK GENERATION TEST
#  Generate a small sample right after training
#  so you can immediately see if it learned something
# ------------------------------------------------

print(f"\nQuick generation test with trained model...")
print("-" * 45)

def quick_generate(seed_text, num_chars=150, temperature=0.8):
    char_to_ix = { v: int(k) for k, v in ix_to_char.items() }

    # Build starting context from seed
    context = [char_to_ix.get(ch, 0) for ch in seed_text[-seq_len:]]

    # Pad if seed is shorter than seq_len
    while len(context) < seq_len:
        context = [0] + context

    result = seed_text

    for _ in range(num_chars):
        x_ids = np.array(context[-seq_len:])
        _, _, _, _, probs = forward(x_ids)

        # Temperature scaling — higher = more creative, lower = more repetitive
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs - np.max(probs))
        probs = probs / probs.sum()

        # Sample the next character from the probability distribution
        next_ix = np.random.choice(vocab_size, p=probs)
        result  += ix_to_char[next_ix]
        context.append(next_ix)

    return result

seed = "To be or not"
output = quick_generate(seed)
print(f"  Seed    : '{seed}'")
print(f"  Output  :")
print(f"  {output}")
print("-" * 45)
print(f"\nNext step → run generate.py for interactive generation")