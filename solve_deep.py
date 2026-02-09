"""
Deep local search with all types of swaps and 3-way moves.
"""
import torch
import pandas as pd
import numpy as np
import random
import time
from itertools import combinations

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

# Load current best
print("--- Loading current best ---")
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
pairs = [(perm[i], perm[i+1]) for i in range(0, 96, 2)]

best_pairs = pairs.copy()
best_mse = evaluate(best_pairs)
print(f"Starting MSE: {best_mse:.6f}")

# Deep local search
print("\n--- Deep Local Search ---")
start = time.time()

round_num = 0
while True:
    round_num += 1
    improved = False

    # 2-swap: order
    print(f"Round {round_num}: Trying order swaps...")
    for i in range(48):
        for j in range(i+1, 48):
            new_pairs = best_pairs.copy()
            new_pairs[i], new_pairs[j] = new_pairs[j], new_pairs[i]
            mse = evaluate(new_pairs)
            if mse < best_mse - 1e-8:
                best_mse = mse
                best_pairs = new_pairs
                improved = True
                print(f"  Order swap ({i},{j}): MSE = {best_mse:.8f}")

    # 2-swap: out layers
    print(f"Round {round_num}: Trying out swaps...")
    for i in range(48):
        for j in range(i+1, 48):
            new_pairs = best_pairs.copy()
            inp_i, out_i = new_pairs[i]
            inp_j, out_j = new_pairs[j]
            new_pairs[i] = (inp_i, out_j)
            new_pairs[j] = (inp_j, out_i)
            mse = evaluate(new_pairs)
            if mse < best_mse - 1e-8:
                best_mse = mse
                best_pairs = new_pairs
                improved = True
                print(f"  Out swap ({i},{j}): MSE = {best_mse:.8f}")

    # 2-swap: inp layers
    print(f"Round {round_num}: Trying inp swaps...")
    for i in range(48):
        for j in range(i+1, 48):
            new_pairs = best_pairs.copy()
            inp_i, out_i = new_pairs[i]
            inp_j, out_j = new_pairs[j]
            new_pairs[i] = (inp_j, out_i)
            new_pairs[j] = (inp_i, out_j)
            mse = evaluate(new_pairs)
            if mse < best_mse - 1e-8:
                best_mse = mse
                best_pairs = new_pairs
                improved = True
                print(f"  Inp swap ({i},{j}): MSE = {best_mse:.8f}")

    # 3-rotate: positions
    if not improved:
        print(f"Round {round_num}: Trying 3-rotations...")
        for i, j, k in combinations(range(48), 3):
            # Try both rotation directions
            for new_order in [(k, i, j), (j, k, i)]:
                new_pairs = best_pairs.copy()
                p_i, p_j, p_k = best_pairs[i], best_pairs[j], best_pairs[k]
                new_pairs[i] = best_pairs[new_order[0]]
                new_pairs[j] = best_pairs[new_order[1]]
                new_pairs[k] = best_pairs[new_order[2]]
                mse = evaluate(new_pairs)
                if mse < best_mse - 1e-8:
                    best_mse = mse
                    best_pairs = new_pairs
                    improved = True
                    print(f"  3-rotate ({i},{j},{k}): MSE = {best_mse:.8f}")
                    break
            if improved:
                break

    # 3-swap: out layers among 3 positions
    if not improved:
        print(f"Round {round_num}: Trying 3-way out swaps...")
        count = 0
        for i, j, k in combinations(range(48), 3):
            count += 1
            if count > 5000:  # Limit to avoid too long
                break
            inp_i, out_i = best_pairs[i]
            inp_j, out_j = best_pairs[j]
            inp_k, out_k = best_pairs[k]

            # Try all permutations of out layers
            for new_outs in [(out_j, out_k, out_i), (out_k, out_i, out_j)]:
                new_pairs = best_pairs.copy()
                new_pairs[i] = (inp_i, new_outs[0])
                new_pairs[j] = (inp_j, new_outs[1])
                new_pairs[k] = (inp_k, new_outs[2])
                mse = evaluate(new_pairs)
                if mse < best_mse - 1e-8:
                    best_mse = mse
                    best_pairs = new_pairs
                    improved = True
                    print(f"  3-out swap ({i},{j},{k}): MSE = {best_mse:.8f}")
                    break
            if improved:
                break

    print(f"Round {round_num} complete: MSE = {best_mse:.8f}")

    if not improved:
        print("No improvement found. Stopping.")
        break

    if round_num >= 50:
        print("Max rounds reached.")
        break

print(f"\nDone in {time.time()-start:.1f}s")
print(f"Final MSE: {best_mse:.8f}")

# Save
permutation = []
for inp_idx, out_idx in best_pairs:
    permutation.append(inp_idx)
    permutation.append(out_idx)
permutation.append(last_layer_idx)

with open("solution.txt", "w") as f:
    f.write(",".join(map(str, permutation)))

print("Saved to solution.txt")
