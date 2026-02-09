"""
Extended ILS v2: Best-improvement local search with prefix caching.
Previous first-improvement was too weak (converged to MSE 0.3+).
This uses best-improvement (scans all swaps, picks best) but with
prefix caching to skip recomputing the unchanged head of the chain.
"""
import torch
import pandas as pd
import numpy as np
import time
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load ─────────────────────────────────────────────────────────
pieces = {}
inp_layers, out_layers, last_layer_idx = [], [], None
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

df = pd.read_csv('historical_data.csv')
X_full = torch.tensor(df.iloc[:, :48].values, dtype=torch.float32, device=device)
pred_full = torch.tensor(df['pred'].values, dtype=torch.float32, device=device)

SUB = 1000
idx = torch.randperm(len(X_full))[:SUB]
X_sub = X_full[idx]
pred_sub = pred_full[idx]

inp_wt = {i: (pieces[i]['weight'], pieces[i]['bias']) for i in inp_layers}
out_wt = {i: (pieces[i]['weight'], pieces[i]['bias']) for i in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']


@torch.no_grad()
def apply_block(x, inp_idx, out_idx):
    iw, ib = inp_wt[inp_idx]
    ow, ob = out_wt[out_idx]
    return x + torch.relu(x @ iw.T + ib) @ ow.T + ob


@torch.no_grad()
def evaluate(pairs, fast=False):
    x = (X_sub if fast else X_full).clone()
    for inp_idx, out_idx in pairs:
        x = apply_block(x, inp_idx, out_idx)
    return ((x @ ll_w.T + ll_b).squeeze() - (pred_sub if fast else pred_full)).pow(2).mean().item()


@torch.no_grad()
def compute_states(pairs, X):
    """Compute intermediate states: states[k] = input to block k."""
    states = [X.clone()]
    x = states[0]
    for inp_idx, out_idx in pairs:
        x = apply_block(x, inp_idx, out_idx)
        states.append(x.clone())
    return states


@torch.no_grad()
def eval_from(pairs, start_state, start_pos, pred):
    """Evaluate chain from position start_pos using cached prefix."""
    x = start_state.clone()
    for k in range(start_pos, len(pairs)):
        x = apply_block(x, pairs[k][0], pairs[k][1])
    return ((x @ ll_w.T + ll_b).squeeze() - pred).pow(2).mean().item()


def local_search_cached(pairs, fast=True, max_rounds=8):
    """Best-improvement with prefix caching."""
    best = list(pairs)
    best_mse = evaluate(best, fast=fast)
    X = X_sub if fast else X_full
    pred = pred_sub if fast else pred_full

    for rnd in range(max_rounds):
        states = compute_states(best, X)
        found_move = None
        found_mse = best_mse

        for i in range(48):
            prefix = states[i]
            for j in range(i + 1, 48):
                for move in range(3):
                    new = list(best)
                    if move == 0:
                        new[i], new[j] = new[j], new[i]
                    elif move == 1:
                        ii, oi = new[i]; ij, oj = new[j]
                        new[i] = (ii, oj); new[j] = (ij, oi)
                    else:
                        ii, oi = new[i]; ij, oj = new[j]
                        new[i] = (ij, oi); new[j] = (ii, oj)

                    mse = eval_from(new, prefix, i, pred)
                    if mse < found_mse - 1e-9:
                        found_mse = mse
                        found_move = list(new)

        if found_move is not None:
            best = found_move
            best_mse = found_mse
        else:
            break

    return best, best_mse


def perturb(pairs, intensity):
    pairs = list(pairs)
    for _ in range(intensity):
        move = random.choice(['swap_order', 'swap_inp', 'swap_out',
                              'shuffle_seg', 'rotate3'])
        if move == 'rotate3':
            a, b, c = random.sample(range(48), 3)
            pairs[a], pairs[b], pairs[c] = pairs[c], pairs[a], pairs[b]
        elif move == 'shuffle_seg':
            start = random.randint(0, 44)
            length = random.randint(3, min(8, 48 - start))
            seg = pairs[start:start + length]
            random.shuffle(seg)
            pairs[start:start + length] = seg
        else:
            a, b = random.sample(range(48), 2)
            if move == 'swap_order':
                pairs[a], pairs[b] = pairs[b], pairs[a]
            elif move == 'swap_inp':
                ia, oa = pairs[a]; ib, ob = pairs[b]
                pairs[a] = (ib, oa); pairs[b] = (ia, ob)
            elif move == 'swap_out':
                ia, oa = pairs[a]; ib, ob = pairs[b]
                pairs[a] = (ia, ob); pairs[b] = (ib, oa)
    return pairs


def save(pairs):
    perm = []
    for a, b in pairs:
        perm.extend([a, b])
    perm.append(last_layer_idx)
    with open("solution.txt", "w") as f:
        f.write(",".join(map(str, perm)))


# ── Load current best ────────────────────────────────────────────
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
best_pairs = [(perm[i], perm[i + 1]) for i in range(0, 96, 2)]
best_mse = evaluate(best_pairs, fast=False)
best_mse_fast = evaluate(best_pairs, fast=True)
print(f"Starting MSE: {best_mse:.8f}")

# ── Warmup: time one local search to estimate runtime ─────────
print("Timing one local search round...")
t_warm = time.time()
_ = local_search_cached(best_pairs, fast=True, max_rounds=1)
round_time = time.time() - t_warm
print(f"  1 round = {round_time:.1f}s")
est_per_restart = round_time * 6  # ~6 rounds avg before convergence
n_restarts = 200
print(f"  Est. per restart: ~{est_per_restart:.0f}s, total ~{est_per_restart*n_restarts/60:.0f}min for {n_restarts} restarts")

# ── ILS ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ILS: {n_restarts} restarts (best-improvement + prefix cache)")
print(f"{'='*60}")
t0 = time.time()
improvements = 0

for restart in range(n_restarts):
    # At 0.027, use mostly small perturbations
    r = random.random()
    if r < 0.6:
        intensity = random.randint(1, 3)
    elif r < 0.85:
        intensity = random.randint(3, 6)
    elif r < 0.95:
        intensity = random.randint(6, 12)
    else:
        intensity = random.randint(12, 20)

    pairs = perturb(best_pairs, intensity)
    pairs, mse = local_search_cached(pairs, fast=True, max_rounds=8)

    if restart % 25 == 0:
        elapsed = time.time() - t0
        rate = (restart + 1) / elapsed if elapsed > 0 else 0
        eta = (n_restarts - restart) / rate if rate > 0 else 0
        print(f"  [{restart:3d}/{n_restarts}] fast={mse:.6f}  best={best_mse:.8f}  "
              f"{elapsed:.0f}s  {improvements}imp  ETA:{eta:.0f}s")

    if mse < best_mse_fast - 1e-6:
        full_mse = evaluate(pairs, fast=False)
        if full_mse < best_mse:
            best_mse = full_mse
            best_mse_fast = evaluate(pairs, fast=True)
            best_pairs = pairs
            save(best_pairs)
            improvements += 1
            print(f"  ** Restart {restart}: NEW BEST {best_mse:.8f} (intensity={intensity}) **")

elapsed = time.time() - t0
print(f"\nILS done: {elapsed:.0f}s, {improvements} improvements")
print(f"Best MSE: {best_mse:.8f}")

# ── Final polish on full data ────────────────────────────────────
print(f"\nFinal full-data local search...")
t1 = time.time()
best_pairs, best_mse = local_search_cached(best_pairs, fast=False, max_rounds=5)
save(best_pairs)
print(f"Final MSE: {best_mse:.10f} ({time.time()-t1:.0f}s)")
print(f"Total: {time.time()-t0:.0f}s")
