import numpy as np
import json
import os


# ------------------------------------------------
#  SETTINGS  (you can change these)
# ------------------------------------------------

DATA_FILE = "data.txt"   # your text file name
SEQ_LEN   = 25           # how many characters the model reads at once


# ------------------------------------------------
#  STEP 1 — READ THE TEXT FILE
#  We just open the file and read everything
#  inside it as one big string
# ------------------------------------------------

if not os.path.exists(DATA_FILE):
    print("ERROR: data.txt not found. Put it in this folder.")
    exit()

text = open(DATA_FILE, "r", encoding="utf-8").read()

print(f"Text loaded!")
print(f"Total characters: {len(text):,}")
print(f"First 100 chars : {repr(text[:100])}")


# ------------------------------------------------
#  STEP 2 — BUILD THE VOCABULARY
#  Find every unique character in the text.
#  This list of unique chars is our "vocabulary".
#
#  Example: "hello" -> ['e', 'h', 'l', 'o']
#  (sorted and no duplicates)
# ------------------------------------------------

chars      = sorted(set(text))   # unique chars, sorted A-Z
vocab_size = len(chars)

print(f"\nVocabulary size : {vocab_size} unique characters")
print(f"All characters  : {repr(''.join(chars))}")


# ------------------------------------------------
#  STEP 3 — CREATE TWO LOOKUP TABLES
#  We need to convert chars <-> numbers
#  because the model only understands numbers.
#
#  char_to_ix : 'a' -> 39  (char to integer)
#  ix_to_char : 39  -> 'a' (integer to char)
# ------------------------------------------------

char_to_ix = { ch: i  for i, ch in enumerate(chars) }
ix_to_char = { i:  ch for i, ch in enumerate(chars) }

# Quick check - show 5 examples
print(f"\nSample encodings:")
for ch in chars[:5]:
    print(f"  '{ch}'  ->  {char_to_ix[ch]}")


# ------------------------------------------------
#  STEP 4 — ENCODE THE WHOLE TEXT
#  Replace every character with its integer ID.
#  "To be" -> [44, 47, 1, 40, 43]
# ------------------------------------------------

encoded = np.array([char_to_ix[ch] for ch in text], dtype=np.int32)

print(f"\nEncoded! First 10 IDs : {encoded[:10].tolist()}")
print(f"Decoded back          : {repr(''.join(ix_to_char[i] for i in encoded[:10]))}")


# ------------------------------------------------
#  STEP 5 — SPLIT INTO TRAIN / VAL / TEST
#  We divide the data into 3 parts:
#
#  Train (80%) -> model learns from this
#  Val   (10%) -> we check progress here
#  Test  (10%) -> final check at the very end
#
#  We do NOT shuffle - text must stay in order!
# ------------------------------------------------

n          = len(encoded)
train_end  = int(n * 0.80)   # 80% mark
val_end    = int(n * 0.90)   # 90% mark

train_data = encoded[:train_end]
val_data   = encoded[train_end:val_end]
test_data  = encoded[val_end:]

print(f"\nData split:")
print(f"  Train : {len(train_data):,} chars")
print(f"  Val   : {len(val_data):,} chars")
print(f"  Test  : {len(test_data):,} chars")


# ------------------------------------------------
#  STEP 6 — BUILD X and Y PAIRS
#  This is how the model learns.
#
#  We slide a window of SEQ_LEN chars across the text.
#  X = the window                  (what model reads)
#  Y = window shifted 1 to right   (what model must predict)
#
#  Example with SEQ_LEN = 5:
#
#  Text :  T  o  _  b  e  _  o  r
#  X[0] :  T  o  _  b  e          <- model reads this
#  Y[0] :     o  _  b  e  _       <- model must predict this
#
#  At every position, model predicts the NEXT character.
# ------------------------------------------------

def make_sequences(data, seq_len):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i     : i + seq_len])      # input window
        Y.append(data[i + 1 : i + seq_len + 1])  # target (shifted by 1)
    return np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)


print(f"\nBuilding sequences (SEQ_LEN={SEQ_LEN})...")

X_train, Y_train = make_sequences(train_data, SEQ_LEN)
X_val,   Y_val   = make_sequences(val_data,   SEQ_LEN)

print(f"  X_train shape : {X_train.shape}  <- (num sequences, seq length)")
print(f"  Y_train shape : {Y_train.shape}")


# ------------------------------------------------
#  QUICK SANITY CHECK
#  Decode one X/Y pair back to text so we can
#  see with our own eyes that it looks correct
# ------------------------------------------------

sample_x = ''.join(ix_to_char[i] for i in X_train[0])
sample_y = ''.join(ix_to_char[i] for i in Y_train[0])

print(f"\nSample pair #0:")
print(f"  X (input)  : {repr(sample_x)}")
print(f"  Y (target) : {repr(sample_y)}")
print(f"  Y is just X shifted one character to the right - correct!")


# ------------------------------------------------
#  STEP 7 — SAVE EVERYTHING
#  We save two files:
#
#  data_prepared.npz -> the X and Y number arrays
#  vocab.json        -> the lookup tables (chars <-> numbers)
#
#  train.py will load these files automatically.
# ------------------------------------------------

try:
    np.savez("data_prepared.npz",
        X_train = X_train,
        Y_train = Y_train,
        X_val   = X_val,
        Y_val   = Y_val,
        test    = test_data
    )
except Exception as e:
    print(f"ERROR saving data_prepared.npz: {e}")
    exit()

vocab = {
    "vocab_size"  : vocab_size,
    "seq_len"     : SEQ_LEN,
    "char_to_ix"  : char_to_ix,
    "ix_to_char"  : { str(k): v for k, v in ix_to_char.items() }
}

try:
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
except Exception as e:
    print(f"ERROR saving vocab.json: {e}")
    exit()

print(f"\nSaved: data_prepared.npz")
print(f"Saved: vocab.json")


# ------------------------------------------------
#  DONE!
# ------------------------------------------------

print(f"""
================================================
  ALL DONE - SUMMARY
================================================
  Characters in file   : {len(text):,}
  Unique characters    : {vocab_size}
  Sequence length      : {SEQ_LEN}
  Training sequences   : {len(X_train):,}
  Validation sequences : {len(X_val):,}

  
================================================
""")
