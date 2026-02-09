"""
Phase 2: Escape the local minimum.

Strategy:
  1. Leave-one-out analysis: identify which blocks are hurting the prediction
  2. Targeted exhaustive search on suspicious blocks (3-way swaps)
  3. Iterated Local Search: random perturbations + fast local search
  4. Full greedy rebuild from scratch trying ALL pairs at each position
"""
import torch
import pandas as pd
import numpy as np
import time
import random
from copy import deepcopy

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

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv('historical_data.csv')
X_full = torch.tensor(df.iloc[:, :48].values, dtype=torch.float32, device=device)
pred_full = torch.tensor(df['pred'].values, dtype=torch.float32, device=device)

# Subsample for fast evaluation during search
SUB = 2000
indices = torch.randperm(len(X_full))[:SUB]
X_sub = X_full[indices]
pred_sub = pred_full[indices]

inp_wt = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in inp_layers}
out_wt = {idx: (pieces[idx]['weight'], pieces[idx]['bias']) for idx in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']


@torch.no_grad()
def evaluate(pairs, fast=False):
    x = (X_sub if fast else X_full).clone()
    p = pred_sub if fast else pred_full
    for inp_idx, out_idx in pairs:
        iw, ib = inp_wt[inp_idx]
        ow, ob = out_wt[out_idx]
        x = x + torch.relu(x @ iw.T + ib) @ ow.T + ob
    return ((x @ ll_w.T + ll_b).squeeze() - p).pow(2).mean().item()


def load_solution():
    with open("solution.txt", "r") as f:
        perm = list(map(int, f.read().strip().split(",")))
    return [(perm[i], perm[i + 1]) for i in range(0, 96, 2)]


def save_solution(pairs):
    perm = []
    for inp_idx, out_idx in pairs:
        perm.append(inp_idx)
        perm.append(out_idx)
    perm.append(last_layer_idx)
    with open("solution.txt", "w") as f:
        f.write(",".join(map(str, perm)))


def local_search(pairs, fast=True, max_rounds=20):
    """Fast 2-way local search."""
    best = list(pairs)
    best_mse = evaluate(best, fast=fast)

    for rnd in range(max_rounds):
        improved = False

        # Swap ordering
        for i in range(48):
            for j in range(i + 1, 48):
                new = list(best)
                new[i], new[j] = new[j], new[i]
                mse = evaluate(new, fast=fast)
                if mse < best_mse - 1e-9:
                    best_mse = mse
                    best = new
                    improved = True

        # Swap out layers
        for i in range(48):
            for j in range(i + 1, 48):
                new = list(best)
                ii, oi = new[i]
                ij, oj = new[j]
                new[i] = (ii, oj)
                new[j] = (ij, oi)
                mse = evaluate(new, fast=fast)
                if mse < best_mse - 1e-9:
                    best_mse = mse
                    best = new
                    improved = True

        # Swap inp layers
        for i in range(48):
            for j in range(i + 1, 48):
                new = list(best)
                ii, oi = new[i]
                ij, oj = new[j]
                new[i] = (ij, oi)
                new[j] = (ii, oj)
                mse = evaluate(new, fast=fast)
                if mse < best_mse - 1e-9:
                    best_mse = mse
                    best = new
                    improved = True

        if not improved:
            break

    return best, best_mse


# ================================================================
# Load current best
# ================================================================
best_pairs = load_solution()
best_mse = evaluate(best_pairs, fast=False)
print(f"Current best MSE (full): {best_mse:.8f}")
best_mse_fast = evaluate(best_pairs, fast=True)
print(f"Current best MSE (sub):  {best_mse_fast:.8f}")

# ================================================================
# Strategy 1: Leave-one-out analysis
# ================================================================
print("\n" + "=" * 60)
print("STRATEGY 1: Leave-One-Out Analysis")
print("=" * 60)
t0 = time.time()

loo_scores = []
for k in range(48):
    # Skip block k (identity)
    pairs_without_k = best_pairs[:k] + best_pairs[k + 1:]
    mse_without = evaluate(pairs_without_k, fast=True)
    delta = mse_without - best_mse_fast  # positive = block helps, negative = block hurts
    loo_scores.append((k, delta, best_pairs[k]))

# Sort by delta (most negative first = most suspicious)
loo_scores.sort(key=lambda x: x[1])

print(f"Leave-one-out analysis done in {time.time()-t0:.1f}s")
print("\nMost suspicious blocks (negative delta = hurting):")
for k, delta, pair in loo_scores[:10]:
    print(f"  Position {k:2d}: pair ({pair[0]:2d}, {pair[1]:2d}), delta = {delta:+.6f}")
print("\nMost confident blocks (large positive delta = clearly helping):")
for k, delta, pair in loo_scores[-5:]:
    print(f"  Position {k:2d}: pair ({pair[0]:2d}, {pair[1]:2d}), delta = {delta:+.6f}")

suspicious = [k for k, delta, _ in loo_scores if delta < 0.001]
print(f"\n{len(suspicious)} suspicious positions (delta < 0.001)")

# ================================================================
# Strategy 2: Targeted 3-way search on suspicious blocks
# ================================================================
print("\n" + "=" * 60)
print("STRATEGY 2: Targeted 3-Way Search")
print("=" * 60)
t1 = time.time()

# For pairs of suspicious blocks + each other block, try all 3-way rearrangements
sus_set = set(suspicious[:15])  # Focus on top 15 most suspicious
print(f"Searching 3-way moves involving {len(sus_set)} suspicious positions...")

search_pairs = list(best_pairs)
search_mse = evaluate(search_pairs, fast=True)
found_improvement = False

for s in sorted(sus_set):
    for j in range(48):
        if j == s:
            continue
        for k in range(j + 1, 48):
            if k == s:
                continue
            # Try all 6 permutations of blocks at positions s, j, k
            blocks = [search_pairs[s], search_pairs[j], search_pairs[k]]
            from itertools import permutations
            for perm in permutations(blocks):
                if perm == (blocks[0], blocks[1], blocks[2]):
                    continue  # skip identity
                new = list(search_pairs)
                new[s], new[j], new[k] = perm[0], perm[1], perm[2]
                mse = evaluate(new, fast=True)
                if mse < search_mse - 1e-9:
                    search_mse = mse
                    search_pairs = new
                    found_improvement = True

if found_improvement:
    # Verify with full data
    full_mse = evaluate(search_pairs, fast=False)
    print(f"  Improved! Fast MSE: {search_mse:.8f}, Full MSE: {full_mse:.8f}")
    if full_mse < best_mse:
        best_mse = full_mse
        best_pairs = search_pairs
        save_solution(best_pairs)
        print(f"  Saved new best: {best_mse:.8f}")
else:
    print("  No improvement from targeted 3-way search")
print(f"  Time: {time.time()-t1:.1f}s")

# ================================================================
# Strategy 3: Iterated Local Search (random perturbation + search)
# ================================================================
print("\n" + "=" * 60)
print("STRATEGY 3: Iterated Local Search (100 restarts)")
print("=" * 60)
t2 = time.time()

n_restarts = 100

for restart in range(n_restarts):
    # Start from current best
    pairs = list(best_pairs)

    # Random perturbation with varying intensity
    intensity = random.choice([2, 3, 4, 5, 6, 8, 10, 15])

    # Mix of perturbation types
    for _ in range(intensity):
        move = random.choice(['swap_order', 'swap_inp', 'swap_out', 'shuffle_seg'])
        i, j = random.sample(range(48), 2)

        if move == 'swap_order':
            pairs[i], pairs[j] = pairs[j], pairs[i]
        elif move == 'swap_inp':
            ii, oi = pairs[i]
            ij, oj = pairs[j]
            pairs[i] = (ij, oi)
            pairs[j] = (ii, oj)
        elif move == 'swap_out':
            ii, oi = pairs[i]
            ij, oj = pairs[j]
            pairs[i] = (ii, oj)
            pairs[j] = (ij, oi)
        elif move == 'shuffle_seg':
            start = random.randint(0, 44)
            length = random.randint(3, min(8, 48 - start))
            seg = pairs[start:start + length]
            random.shuffle(seg)
            pairs[start:start + length] = seg

    # Fast local search
    pairs, mse = local_search(pairs, fast=True, max_rounds=10)

    if restart % 20 == 0:
        print(f"  Restart {restart:3d}: fast MSE = {mse:.8f} (best = {best_mse:.8f})")

    # Check if improved (verify with full data)
    if mse < best_mse_fast - 1e-6:
        full_mse = evaluate(pairs, fast=False)
        if full_mse < best_mse:
            best_mse = full_mse
            best_mse_fast = evaluate(pairs, fast=True)
            best_pairs = pairs
            save_solution(best_pairs)
            print(f"  ** Restart {restart}: NEW BEST {best_mse:.8f} **")

print(f"ILS done in {time.time()-t2:.1f}s")
print(f"Best MSE after ILS: {best_mse:.8f}")

# ================================================================
# Strategy 4: Full greedy rebuild from scratch
# ================================================================
print("\n" + "=" * 60)
print("STRATEGY 4: Greedy Rebuild (all pairs at each position)")
print("=" * 60)
t3 = time.time()

# Try building the solution from scratch, one block at a time.
# At each position, try ALL available (inp, out) pairs.
# Use the current best for remaining positions as proxy.

ref_pairs = list(best_pairs)  # reference for remaining blocks


def greedy_build(ref_pairs, reverse=False):
    """Build solution greedily, trying all available pairs at each position.
    Uses ref_pairs for remaining blocks as evaluation proxy."""
    avail_inp = set(inp_layers)
    avail_out = set(out_layers)
    built = []

    positions = list(range(48))
    if reverse:
        positions = positions[::-1]

    for step, pos in enumerate(positions):
        best_pos_mse = float('inf')
        best_pair = None

        # Build the trial chain for evaluation
        for inp_idx in list(avail_inp):
            for out_idx in list(avail_out):
                candidate = (inp_idx, out_idx)

                if reverse:
                    # Build: ref_pairs for unplaced earlier positions + ... + built (reversed)
                    remaining_early = []
                    for p in range(pos):
                        rp = ref_pairs[p]
                        # Use ref pair, but swap if layers are taken
                        ri, ro = rp
                        if ri in avail_inp and ro in avail_out and ri != inp_idx and ro != out_idx:
                            remaining_early.append(rp)
                        else:
                            remaining_early.append(rp)  # imperfect but fast
                    trial = remaining_early + [candidate] + list(reversed(built))
                else:
                    # Build: built so far + candidate + ref_pairs for remaining
                    remaining_later = []
                    for p in range(pos + 1, 48):
                        remaining_later.append(ref_pairs[p])
                    trial = built + [candidate] + remaining_later

                mse = evaluate(trial, fast=True)
                if mse < best_pos_mse:
                    best_pos_mse = mse
                    best_pair = candidate

        built.append(best_pair)
        avail_inp.remove(best_pair[0])
        avail_out.remove(best_pair[1])

        if step % 12 == 0 or step == 47:
            print(f"    Step {step} (pos {pos}): MSE = {best_pos_mse:.6f}")

    if reverse:
        built = list(reversed(built))
    return built


# Forward greedy
print("\n  Forward greedy build:")
pairs_fwd = greedy_build(ref_pairs, reverse=False)
mse_fwd = evaluate(pairs_fwd, fast=False)
print(f"  Forward MSE: {mse_fwd:.8f}")

# Backward greedy
print("\n  Backward greedy build:")
pairs_bwd = greedy_build(ref_pairs, reverse=True)
mse_bwd = evaluate(pairs_bwd, fast=False)
print(f"  Backward MSE: {mse_bwd:.8f}")

# Local search on the better one
better = pairs_fwd if mse_fwd < mse_bwd else pairs_bwd
better_mse = min(mse_fwd, mse_bwd)
print(f"\n  Refining better rebuild (MSE {better_mse:.8f}) with local search...")
better, better_mse_fast = local_search(better, fast=True, max_rounds=20)
better_mse = evaluate(better, fast=False)
print(f"  After local search: {better_mse:.8f}")

if better_mse < best_mse:
    best_mse = better_mse
    best_pairs = better
    save_solution(best_pairs)
    print(f"  Saved new best: {best_mse:.8f}")

print(f"  Time: {time.time()-t3:.1f}s")

# ================================================================
# Final refinement: full local search on best result
# ================================================================
print("\n" + "=" * 60)
print("FINAL: Full Local Search (full data)")
print("=" * 60)
t4 = time.time()

best_pairs, best_mse = local_search(best_pairs, fast=False, max_rounds=10)
print(f"Final MSE: {best_mse:.10f}")
save_solution(best_pairs)
print(f"Time: {time.time()-t4:.1f}s")
print(f"\nTotal time: {time.time()-t0:.1f}s")
