"""
Solve with joint optimization of pairing AND ordering.
"""
import torch
import pandas as pd
import numpy as np
import random
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
def evaluate(pairs):
    x = X.clone()
    for inp_idx, out_idx in pairs:
        inp_w, inp_b = inp_weights[inp_idx]
        out_w, out_b = out_weights[out_idx]
        hidden = torch.relu(x @ inp_w.T + inp_b)
        x = x + hidden @ out_w.T + out_b
    output = (x @ ll_w.T + ll_b).squeeze()
    return ((output - pred) ** 2).mean().item()

# Load best solution from v2 as starting point
print("\n--- Loading previous best solution ---")
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))

# Convert permutation back to pairs
pairs = []
for i in range(0, 96, 2):
    pairs.append((perm[i], perm[i+1]))

print(f"Loaded {len(pairs)} pairs, MSE = {evaluate(pairs):.6f}")

# Heavy local search: try ALL possible moves
print("\n--- Extended local search ---")
start = time.time()

best_pairs = pairs.copy()
best_mse = evaluate(best_pairs)
print(f"Starting MSE: {best_mse:.6f}")

improved = True
round_num = 0

while improved:
    improved = False
    round_num += 1

    # Move 1: Swap ordering of two pairs
    for i in range(48):
        for j in range(i+1, 48):
            new_pairs = best_pairs.copy()
            new_pairs[i], new_pairs[j] = new_pairs[j], new_pairs[i]
            mse = evaluate(new_pairs)
            if mse < best_mse - 1e-6:
                best_mse = mse
                best_pairs = new_pairs
                improved = True
                print(f"  Round {round_num}: Swap order ({i},{j}) -> MSE = {best_mse:.6f}")

    # Move 2: Swap out layers between two pairs
    for i in range(48):
        for j in range(i+1, 48):
            new_pairs = best_pairs.copy()
            inp_i, out_i = new_pairs[i]
            inp_j, out_j = new_pairs[j]
            new_pairs[i] = (inp_i, out_j)
            new_pairs[j] = (inp_j, out_i)
            mse = evaluate(new_pairs)
            if mse < best_mse - 1e-6:
                best_mse = mse
                best_pairs = new_pairs
                improved = True
                print(f"  Round {round_num}: Swap out ({i},{j}) -> MSE = {best_mse:.6f}")

    # Move 3: Swap inp layers between two pairs
    for i in range(48):
        for j in range(i+1, 48):
            new_pairs = best_pairs.copy()
            inp_i, out_i = new_pairs[i]
            inp_j, out_j = new_pairs[j]
            new_pairs[i] = (inp_j, out_i)
            new_pairs[j] = (inp_i, out_j)
            mse = evaluate(new_pairs)
            if mse < best_mse - 1e-6:
                best_mse = mse
                best_pairs = new_pairs
                improved = True
                print(f"  Round {round_num}: Swap inp ({i},{j}) -> MSE = {best_mse:.6f}")

    # Move 4: Rotate 3 pairs
    if not improved:
        for i in range(48):
            for j in range(i+1, 48):
                for k in range(j+1, 48):
                    new_pairs = best_pairs.copy()
                    # Rotate: i->j->k->i
                    new_pairs[i], new_pairs[j], new_pairs[k] = best_pairs[k], best_pairs[i], best_pairs[j]
                    mse = evaluate(new_pairs)
                    if mse < best_mse - 1e-6:
                        best_mse = mse
                        best_pairs = new_pairs
                        improved = True
                        print(f"  Round {round_num}: Rotate ({i},{j},{k}) -> MSE = {best_mse:.6f}")
                        break
                if improved:
                    break
            if improved:
                break

    print(f"Round {round_num} complete, MSE = {best_mse:.6f}")

print(f"\nLocal search done in {time.time()-start:.1f}s")
print(f"Final MSE: {best_mse:.6f}")

# Save
permutation = []
for inp_idx, out_idx in best_pairs:
    permutation.append(inp_idx)
    permutation.append(out_idx)
permutation.append(last_layer_idx)

print(f"\nPermutation: {permutation}")

with open("solution.txt", "w") as f:
    f.write(",".join(map(str, permutation)))

print("Saved to solution.txt")
