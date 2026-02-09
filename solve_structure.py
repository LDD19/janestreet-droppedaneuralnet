"""
Solve the dropped neural net puzzle by exploiting residual structure.

Key insight: In a residual network x_{k+1} = x_k + f_k(x_k), the total output
depends primarily on which inp/out layers are PAIRED, not their ORDER (first-order
approximation). We can find the optimal pairing by analyzing projected contributions.

Approach:
  1. Compute how each possible (inp, out) pair contributes to the final prediction
     (projected through the last layer) using the original input X.
  2. Find the matching that makes the sum of contributions equal the target.
  3. Optimize ordering via greedy forward construction.
  4. Refine with local search.
"""
import torch
import pandas as pd
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load pieces ──────────────────────────────────────────────────
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

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv('historical_data.csv')
X = torch.tensor(df.iloc[:, :48].values, dtype=torch.float32, device=device)
pred = torch.tensor(df['pred'].values, dtype=torch.float32, device=device)
N = len(X)

inp_wt = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in inp_layers}
out_wt = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']


@torch.no_grad()
def evaluate(pairs):
    x = X.clone()
    for inp_idx, out_idx in pairs:
        iw, ib = inp_wt[inp_idx]
        ow, ob = out_wt[out_idx]
        x = x + torch.relu(x @ iw.T + ib) @ ow.T + ob
    return ((x @ ll_w.T + ll_b).squeeze() - pred).pow(2).mean().item()


# ================================================================
# Phase 1: Find pairing via projected contributions
# ================================================================
print("\n" + "=" * 60)
print("PHASE 1: Projected Contribution Analysis")
print("=" * 60)
t0 = time.time()

# First-order approximation:
#   pred ≈ (X + Σ_k residual_k(X)) @ ll_w.T + ll_b
#   => Σ_k (residual_k(X) @ ll_w.T) ≈ pred - X @ ll_w.T - ll_b
target = pred - (X @ ll_w.T + ll_b).squeeze()  # (N,)
print(f"Target norm: {target.norm().item():.4f}")

# Precompute hidden activations per inp layer (reused across all out layers)
hidden_cache = {}
for idx in inp_layers:
    iw, ib = inp_wt[idx]
    hidden_cache[idx] = torch.relu(X @ iw.T + ib)  # (N, 96)

# Projected contributions: for pair (i,j), how much does it contribute to pred?
proj = {}
for inp_idx in inp_layers:
    h = hidden_cache[inp_idx]
    for out_idx in out_layers:
        ow, ob = out_wt[out_idx]
        residual = h @ ow.T + ob          # (N, 48)
        proj[(inp_idx, out_idx)] = (residual @ ll_w.T).squeeze()  # (N,)

print(f"Computed {len(proj)} projected contributions in {time.time()-t0:.1f}s")

# ── Method A: Least-squares + Hungarian ──────────────────────────
print("\nMethod A: Least-squares + Hungarian")
t1 = time.time()
n_i, n_o = len(inp_layers), len(out_layers)

# Build contribution matrix M (N x n_i*n_o)
M = torch.zeros(N, n_i * n_o, device=device)
for i, inp_idx in enumerate(inp_layers):
    for j, out_idx in enumerate(out_layers):
        M[:, i * n_o + j] = proj[(inp_idx, out_idx)]

# Solve: M @ p ≈ target  (overdetermined: N >> n_i*n_o)
p = torch.linalg.lstsq(M, target.unsqueeze(1)).solution.squeeze().cpu().numpy()
P_mat = p.reshape(n_i, n_o)

# Round to permutation via Hungarian (maximize => negate for minimize)
row_a, col_a = linear_sum_assignment(-P_mat)
pairs_a = [(inp_layers[row_a[k]], out_layers[col_a[k]]) for k in range(n_i)]
mse_a = evaluate(pairs_a)
print(f"  MSE: {mse_a:.6f} ({time.time()-t1:.1f}s)")

# ── Method B: Greedy matching ────────────────────────────────────
print("\nMethod B: Greedy projected matching")
t1 = time.time()

remaining = target.clone()
pairs_b = []
avail_inp = set(inp_layers)
avail_out = set(out_layers)

for step in range(48):
    best_score = float('-inf')
    best_pair = None
    for inp_idx in avail_inp:
        for out_idx in avail_out:
            c = proj[(inp_idx, out_idx)]
            # Reduction in ||remaining||^2 from subtracting c
            score = (2 * torch.dot(remaining, c) - torch.dot(c, c)).item()
            if score > best_score:
                best_score = score
                best_pair = (inp_idx, out_idx)
    pairs_b.append(best_pair)
    remaining -= proj[best_pair]
    avail_inp.remove(best_pair[0])
    avail_out.remove(best_pair[1])

mse_b = evaluate(pairs_b)
print(f"  MSE: {mse_b:.6f} ({time.time()-t1:.1f}s)")
print(f"  Remaining norm: {remaining.norm().item():.4f}")

# ── Method C: Hungarian on correlation cost ──────────────────────
print("\nMethod C: Hungarian on target-correlation cost")
t1 = time.time()

# Cost = how well each pair's contribution correlates with target
cost_c = np.zeros((n_i, n_o))
for i, inp_idx in enumerate(inp_layers):
    for j, out_idx in enumerate(out_layers):
        c = proj[(inp_idx, out_idx)]
        cost_c[i, j] = -torch.dot(c, target).item()  # negative = maximize correlation

row_c, col_c = linear_sum_assignment(cost_c)
pairs_c = [(inp_layers[row_c[k]], out_layers[col_c[k]]) for k in range(n_i)]
mse_c = evaluate(pairs_c)
print(f"  MSE: {mse_c:.6f} ({time.time()-t1:.1f}s)")

# Pick the best pairing
results = [("lstsq+hungarian", mse_a, pairs_a),
           ("greedy", mse_b, pairs_b),
           ("correlation+hungarian", mse_c, pairs_c)]
results.sort(key=lambda x: x[1])
best_name, best_mse, best_pairs = results[0]
print(f"\nBest pairing: {best_name} (MSE {best_mse:.6f})")

# Also load previous best for comparison
try:
    with open("solution.txt", "r") as f:
        old_perm = list(map(int, f.read().strip().split(",")))
    old_pairs = [(old_perm[i], old_perm[i + 1]) for i in range(0, 96, 2)]
    old_mse = evaluate(old_pairs)
    print(f"Previous best MSE: {old_mse:.6f}")
    if old_mse < best_mse:
        print("Previous best is still better — will also refine from there")
        results.append(("previous", old_mse, old_pairs))
        results.sort(key=lambda x: x[1])
        best_name, best_mse, best_pairs = results[0]
except Exception:
    old_mse = float('inf')

# ================================================================
# Phase 2: Greedy ordering
# ================================================================
print("\n" + "=" * 60)
print("PHASE 2: Greedy Ordering")
print("=" * 60)

# Try the top 2 pairings, pick whichever orders better
candidates_for_ordering = results[:2]
overall_best_mse = float('inf')
overall_best_pairs = None

for cand_name, cand_mse, cand_pairs in candidates_for_ordering:
    print(f"\n  Ordering '{cand_name}' (pairing MSE {cand_mse:.6f})...")
    t2 = time.time()

    remaining_pairs = list(cand_pairs)
    ordered = []

    for pos in range(48):
        best_pos_mse = float('inf')
        best_idx = -1

        for idx in range(len(remaining_pairs)):
            trial = ordered + [remaining_pairs[idx]] + \
                    remaining_pairs[:idx] + remaining_pairs[idx + 1:]
            mse = evaluate(trial)
            if mse < best_pos_mse:
                best_pos_mse = mse
                best_idx = idx

        chosen = remaining_pairs.pop(best_idx)
        ordered.append(chosen)

        if pos % 12 == 0 or pos == 47:
            print(f"    Position {pos}: MSE = {best_pos_mse:.6f}")

    mse_ordered = evaluate(ordered)
    print(f"  Ordered MSE: {mse_ordered:.6f} ({time.time()-t2:.1f}s)")

    if mse_ordered < overall_best_mse:
        overall_best_mse = mse_ordered
        overall_best_pairs = ordered

best_pairs = overall_best_pairs
best_mse = overall_best_mse
print(f"\nBest after ordering: {best_mse:.6f}")

# ================================================================
# Phase 3: Local search
# ================================================================
print("\n" + "=" * 60)
print("PHASE 3: Local Search")
print("=" * 60)
t3 = time.time()
print(f"Starting MSE: {best_mse:.8f}")

improved = True
round_num = 0

while improved:
    improved = False
    round_num += 1

    # Move 1: Swap ordering of two blocks
    for i in range(48):
        for j in range(i + 1, 48):
            new = best_pairs.copy()
            new[i], new[j] = new[j], new[i]
            mse = evaluate(new)
            if mse < best_mse - 1e-9:
                best_mse = mse
                best_pairs = new
                improved = True

    # Move 2: Swap out layers between two blocks
    for i in range(48):
        for j in range(i + 1, 48):
            new = best_pairs.copy()
            ii, oi = new[i]
            ij, oj = new[j]
            new[i] = (ii, oj)
            new[j] = (ij, oi)
            mse = evaluate(new)
            if mse < best_mse - 1e-9:
                best_mse = mse
                best_pairs = new
                improved = True

    # Move 3: Swap inp layers between two blocks
    for i in range(48):
        for j in range(i + 1, 48):
            new = best_pairs.copy()
            ii, oi = new[i]
            ij, oj = new[j]
            new[i] = (ij, oi)
            new[j] = (ii, oj)
            mse = evaluate(new)
            if mse < best_mse - 1e-9:
                best_mse = mse
                best_pairs = new
                improved = True

    print(f"  Round {round_num}: MSE = {best_mse:.10f}")

    # Move 4: 3-way rotations (if no improvement from 2-way)
    if not improved:
        print("  Trying 3-way rotations...")
        for i in range(48):
            for j in range(i + 1, 48):
                for k in range(j + 1, 48):
                    # Rotate positions: i→j→k→i
                    new = best_pairs.copy()
                    new[i], new[j], new[k] = best_pairs[k], best_pairs[i], best_pairs[j]
                    mse = evaluate(new)
                    if mse < best_mse - 1e-9:
                        best_mse = mse
                        best_pairs = new
                        improved = True
                        print(f"  3-way rotation ({i},{j},{k}): MSE = {best_mse:.10f}")
                        break
                if improved:
                    break
            if improved:
                break

print(f"\nLocal search done in {time.time()-t3:.1f}s")
print(f"Final MSE: {best_mse:.10f}")

# ================================================================
# Save results
# ================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"New MSE:      {best_mse:.10f}")
print(f"Previous MSE: {old_mse:.10f}")

if best_mse < old_mse:
    print("NEW BEST — saving!")
    perm = []
    for inp_idx, out_idx in best_pairs:
        perm.append(inp_idx)
        perm.append(out_idx)
    perm.append(last_layer_idx)
    with open("solution.txt", "w") as f:
        f.write(",".join(map(str, perm)))
    print(f"Permutation ({len(perm)} elements) saved to solution.txt")
else:
    print("No improvement — keeping previous solution.")

print(f"\nTotal time: {time.time()-t0:.1f}s")
