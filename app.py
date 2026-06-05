from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os

# Resolve absolute paths relative to the application file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
CORS(app)

weights_path = os.path.join(BASE_DIR, "weights.npz")
vocab_path = os.path.join(BASE_DIR, "vocab.json")

if not os.path.exists(weights_path) or not os.path.exists(vocab_path):
    raise RuntimeError(f"ERROR: weights.npz or vocab.json not found. Looked in: {BASE_DIR}")

with open(vocab_path, "r") as f:
    vocab = json.load(f)

vocab_size = vocab["vocab_size"]
seq_len    = vocab["seq_len"]
char_to_ix = vocab["char_to_ix"]
ix_to_char = { int(k): v for k, v in vocab["ix_to_char"].items() }

weights = np.load(weights_path)
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

def generate_text(seed_text, num_chars=300, temperature=0.8):
    context = [char_to_ix.get(ch, 0) for ch in seed_text]
    while len(context) < seq_len:
        context = [0] + context
    context = context[-seq_len:]
    result = seed_text
    
    trace = []
    print("\n--- GENERATION STARTED ---")
    
    for step in range(num_chars):
        x_ids = np.array(context)
        probs, A = forward(x_ids)
        
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs - np.max(probs))
        probs = probs / probs.sum()
        
        next_ix = np.random.choice(vocab_size, p=probs)
        chosen_char = ix_to_char[next_ix]
        result += chosen_char
        
        ctx_str = "".join([ix_to_char.get(c, "") for c in context if c != 0])
        
        # Grab the attention weights of the last token
        valid_len = len([c for c in context if c != 0])
        att_weights = A[-1, -valid_len:].tolist() if valid_len > 0 else []
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probs)[-3:][::-1]
        top_3 = [{"char": ix_to_char[idx], "prob": float(probs[idx])} for idx in top_3_idx]

        print(f"Step {step+1:03d} | Context: '{ctx_str}' -> Selected: '{chosen_char}'")
        
        trace.append({
            "step": step + 1,
            "context_str": ctx_str,
            "chosen_char": chosen_char,
            "attention": att_weights,
            "top_3": top_3
        })
        
        context = context[1:] + [next_ix]
        
    print("--- GENERATION COMPLETE ---\n")
    return result, trace

@app.route('/')
def index():
    from flask import render_template
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_api():
    data = request.json
    if not data or 'seed' not in data:
        return jsonify({'error': 'No seed provided'}), 400
        
    seed = data['seed']
    num_chars = data.get('length', 300)
    temperature = data.get('temperature', 0.8)
    
    try:
        output, trace = generate_text(seed, num_chars=int(num_chars), temperature=float(temperature))
        return jsonify({'result': output, 'trace': trace})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
