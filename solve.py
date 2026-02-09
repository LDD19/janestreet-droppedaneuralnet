"""
Solve the dropped neural net puzzle using simulated annealing on GPU.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from copy import deepcopy

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

pieces_dir = "pieces"

# Load all pieces
pieces = {}
inp_layers = []  # (48->96)
out_layers = []  # (96->48)
last_layer_idx = None

for i in range(97):
    piece = torch.load(f'{pieces_dir}/piece_{i}.pth', weights_only=True, map_location=device)
    pieces[i] = piece
    w = piece['weight']

    if w.shape == (96, 48):
        inp_layers.append(i)
    elif w.shape == (48, 96):
        out_layers.append(i)
    elif w.shape == (1, 48):
        last_layer_idx = i

print(f"inp layers: {len(inp_layers)}, out layers: {len(out_layers)}, LastLayer: {last_layer_idx}")

# Load historical data
df = pd.read_csv('historical_data.csv')
X = torch.tensor(df.iloc[:, :48].values, dtype=torch.float32, device=device)
pred = torch.tensor(df['pred'].values, dtype=torch.float32, device=device)

# Pre-compute weights for speed (already on GPU)
inp_weights = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in inp_layers}
out_weights = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']

def apply_block(x, inp_idx, out_idx):
    """Apply a residual block."""
    inp_w, inp_b = inp_weights[inp_idx]
    out_w, out_b = out_weights[out_idx]
    hidden = torch.relu(x @ inp_w.T + inp_b)
    return x + hidden @ out_w.T + out_b

def apply_last_layer(x):
    """Apply final layer."""
    return (x @ ll_w.T + ll_b).squeeze()

@torch.no_grad()
def evaluate_ordering(pairs):
    """Evaluate the ordering by computing MSE."""
    x = X.clone()
    for inp_idx, out_idx in pairs:
        x = apply_block(x, inp_idx, out_idx)
    output = apply_last_layer(x)
    return ((output - pred) ** 2).mean().item()

# Simulated annealing
print("\n--- Simulated Annealing (GPU) ---\n")

import time
start_time = time.time()

random.seed(42)
random_inp = inp_layers.copy()
random_out = out_layers.copy()
random.shuffle(random_inp)
random.shuffle(random_out)
current_pairs = list(zip(random_inp, random_out))

current_mse = evaluate_ordering(current_pairs)
best_pairs = current_pairs.copy()
best_mse = current_mse

print(f"Initial MSE: {current_mse:.6f}")

# Annealing parameters - more aggressive for GPU
T = 10.0
T_min = 0.00001
alpha = 0.999
iterations_per_temp = 500

iteration = 0
while T > T_min:
    for _ in range(iterations_per_temp):
        new_pairs = current_pairs.copy()

        move_type = random.random()
        if move_type < 0.33:
            # Swap ordering of two pairs
            i, j = random.sample(range(48), 2)
            new_pairs[i], new_pairs[j] = new_pairs[j], new_pairs[i]
        elif move_type < 0.66:
            # Swap out layers between two pairs
            i, j = random.sample(range(48), 2)
            inp_i, out_i = new_pairs[i]
            inp_j, out_j = new_pairs[j]
            new_pairs[i] = (inp_i, out_j)
            new_pairs[j] = (inp_j, out_i)
        else:
            # Swap inp layers between two pairs
            i, j = random.sample(range(48), 2)
            inp_i, out_i = new_pairs[i]
            inp_j, out_j = new_pairs[j]
            new_pairs[i] = (inp_j, out_i)
            new_pairs[j] = (inp_i, out_j)

        new_mse = evaluate_ordering(new_pairs)
        delta = new_mse - current_mse

        if delta < 0 or random.random() < np.exp(-delta / T):
            current_pairs = new_pairs
            current_mse = new_mse

            if current_mse < best_mse:
                best_pairs = current_pairs.copy()
                best_mse = current_mse
                print(f"Iter {iteration}, T={T:.4f}: New best MSE = {best_mse:.6f}")

    T *= alpha
    iteration += 1

    if iteration % 100 == 0:
        print(f"Iter {iteration}, T={T:.6f}, current MSE={current_mse:.6f}, best MSE={best_mse:.6f}")

elapsed = time.time() - start_time
print(f"\nFinal best MSE: {best_mse:.6f}")
print(f"Time elapsed: {elapsed:.1f}s")

# Convert to permutation
permutation = []
for inp_idx, out_idx in best_pairs:
    permutation.append(inp_idx)
    permutation.append(out_idx)
permutation.append(last_layer_idx)

print(f"\nPermutation (length {len(permutation)}):")
print(permutation)

# Save to file
with open("solution.txt", "w") as f:
    f.write(",".join(map(str, permutation)))

# Update findings
with open("findings.md", "a") as f:
    f.write(f"\n\n## Results\n")
    f.write(f"- Best MSE achieved: {best_mse:.6f}\n")
    f.write(f"- Permutation saved to solution.txt\n")

print("\nSolution saved to solution.txt")
