"""
Solve using weight matrix analysis to find correct pairings first,
then use beam search for ordering.
"""
import torch
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pieces
pieces = {}
inp_layers = []
out_layers = []
last_layer_idx = None

for i in range(97):
    piece = torch.load(f'pieces/piece_{i}.pth', weights_only=True, map_location=device)
    pieces[i] = piece
    w = piece['weight']
    if w.shape == (96, 48):
        inp_layers.append(i)
    elif w.shape == (48, 96):
        out_layers.append(i)
    elif w.shape == (1, 48):
        last_layer_idx = i

print(f"inp: {len(inp_layers)}, out: {len(out_layers)}, last: {last_layer_idx}")

# Load data
df = pd.read_csv('historical_data.csv')
X = torch.tensor(df.iloc[:, :48].values, dtype=torch.float32, device=device)
pred = torch.tensor(df['pred'].values, dtype=torch.float32, device=device)

inp_weights = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in inp_layers}
out_weights = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']

@torch.no_grad()
def apply_block(x, inp_idx, out_idx):
    inp_w, inp_b = inp_weights[inp_idx]
    out_w, out_b = out_weights[out_idx]
    hidden = torch.relu(x @ inp_w.T + inp_b)
    return x + hidden @ out_w.T + out_b

@torch.no_grad()
def evaluate(pairs):
    x = X.clone()
    for inp_idx, out_idx in pairs:
        x = apply_block(x, inp_idx, out_idx)
    output = (x @ ll_w.T + ll_b).squeeze()
    return ((output - pred) ** 2).mean().item()

# Strategy 1: Find pairs by testing each combination with just that block
print("\n--- Finding optimal pairs by single-block MSE ---")
start = time.time()

# For each pair, compute MSE when using only that block
pair_mse = np.zeros((48, 48))
for i, inp_idx in enumerate(inp_layers):
    for j, out_idx in enumerate(out_layers):
        x = X.clone()
        x = apply_block(x, inp_idx, out_idx)
        output = (x @ ll_w.T + ll_b).squeeze()
        mse = ((output - pred) ** 2).mean().item()
        pair_mse[i, j] = mse
    if i % 10 == 0:
        print(f"  Processed {i}/48 inp layers...")

# Use Hungarian algorithm to find optimal pairing
row_ind, col_ind = linear_sum_assignment(pair_mse)
pairs = [(inp_layers[i], out_layers[j]) for i, j in zip(row_ind, col_ind)]

print(f"Pairing found in {time.time()-start:.1f}s")
print(f"MSE with random order: {evaluate(pairs):.6f}")

# Strategy 2: Greedy ordering
print("\n--- Greedy ordering ---")
start = time.time()

remaining = pairs.copy()
ordered = []

for step in range(48):
    best_mse = float('inf')
    best_pair = None

    for pair in remaining:
        test_order = ordered + [pair]
        mse = evaluate(test_order)
        if mse < best_mse:
            best_mse = mse
            best_pair = pair

    ordered.append(best_pair)
    remaining.remove(best_pair)

    if step % 10 == 0:
        print(f"  Step {step}: MSE = {best_mse:.6f}")

print(f"Greedy ordering in {time.time()-start:.1f}s")
final_mse = evaluate(ordered)
print(f"Final MSE: {final_mse:.6f}")

# Strategy 3: Local search to improve
print("\n--- Local search refinement ---")
start = time.time()

best_order = ordered.copy()
best_mse = final_mse
improved = True
rounds = 0

while improved and rounds < 100:
    improved = False
    rounds += 1

    # Try swapping adjacent pairs
    for i in range(47):
        new_order = best_order.copy()
        new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
        mse = evaluate(new_order)
        if mse < best_mse:
            best_mse = mse
            best_order = new_order
            improved = True

    # Try swapping any two pairs
    for i in range(48):
        for j in range(i+2, 48):
            new_order = best_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            mse = evaluate(new_order)
            if mse < best_mse:
                best_mse = mse
                best_order = new_order
                improved = True

    if rounds % 10 == 0:
        print(f"  Round {rounds}: MSE = {best_mse:.6f}")

print(f"Local search in {time.time()-start:.1f}s")
print(f"Final MSE after refinement: {best_mse:.6f}")

# Build permutation
permutation = []
for inp_idx, out_idx in best_order:
    permutation.append(inp_idx)
    permutation.append(out_idx)
permutation.append(last_layer_idx)

print(f"\nPermutation: {permutation}")

with open("solution.txt", "w") as f:
    f.write(",".join(map(str, permutation)))

print("Saved to solution.txt")
