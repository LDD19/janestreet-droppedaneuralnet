"""
3-way cross-swap solver.

Key insight: 2-way swaps can only swap layers between 2 positions. But the correct
solution may need an inp from position A and an out from position B combined at
position K — a 3-way rearrangement unreachable by any sequence of 2-way swaps.

Cross-swap(k,a,b):
  k: (inp_a, out_b)     — new cross-pair
  a: (inp_k, keep out_a) — give k's inp to a
  b: (keep inp_b, out_k) — give k's out to b

This explores 48*47*46 = 103,776 moves per pass.
"""
import torch
import pandas as pd
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
sidx = torch.randperm(len(X_full))[:SUB]
X_sub = X_full[sidx]
pred_sub = pred_full[sidx]

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
    for a, b in pairs:
        x = apply_block(x, a, b)
    return ((x @ ll_w.T + ll_b).squeeze() - (pred_sub if fast else pred_full)).pow(2).mean().item()


@torch.no_grad()
def compute_states(pairs, X):
    states = [X.clone()]
    x = states[0]
    for a, b in pairs:
        x = apply_block(x, a, b)
        states.append(x.clone())
    return states


@torch.no_grad()
def eval_from(pairs, start_state, start_pos, pred):
    x = start_state.clone()
    for k in range(start_pos, len(pairs)):
        x = apply_block(x, pairs[k][0], pairs[k][1])
    return ((x @ ll_w.T + ll_b).squeeze() - pred).pow(2).mean().item()


def save(pairs):
    perm = []
    for a, b in pairs:
        perm.extend([a, b])
    perm.append(last_layer_idx)
    with open("solution.txt", "w") as f:
        f.write(",".join(map(str, perm)))


def twoway_search(pairs, X, pred):
    """One pass of best-improvement 2-way search with prefix caching."""
    best = list(pairs)
    best_mse = eval_from(best, X, 0, pred)
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
        return found_move, found_mse, True
    return best, best_mse, False


def crossswap_search(pairs, X, pred):
    """One pass of 3-way cross-swap search with prefix caching."""
    best = list(pairs)
    best_mse = eval_from(best, X, 0, pred)
    states = compute_states(best, X)
    found_move = None
    found_mse = best_mse
    count = 0

    for k in range(48):
        for a in range(48):
            if a == k:
                continue
            for b in range(48):
                if b == k or b == a:
                    continue
                new = list(best)
                # k gets (inp from a, out from b)
                new[k] = (best[a][0], best[b][1])
                # a keeps its out, gets k's inp
                new[a] = (best[k][0], best[a][1])
                # b keeps its inp, gets k's out
                new[b] = (best[b][0], best[k][1])

                start = min(k, a, b)
                mse = eval_from(new, states[start], start, pred)
                count += 1
                if mse < found_mse - 1e-9:
                    found_mse = mse
                    found_move = list(new)

    if found_move is not None:
        return found_move, found_mse, True, count
    return best, best_mse, False, count


def rotation_search(pairs, X, pred):
    """One pass of 3-way rotation search with prefix caching."""
    best = list(pairs)
    best_mse = eval_from(best, X, 0, pred)
    states = compute_states(best, X)
    found_move = None
    found_mse = best_mse

    for i in range(48):
        for j in range(i + 1, 48):
            for k in range(j + 1, 48):
                # Two rotation directions
                for direction in range(2):
                    new = list(best)
                    if direction == 0:
                        new[i], new[j], new[k] = best[k], best[i], best[j]
                    else:
                        new[i], new[j], new[k] = best[j], best[k], best[i]

                    start = i  # i is always the smallest
                    mse = eval_from(new, states[start], start, pred)
                    if mse < found_mse - 1e-9:
                        found_mse = mse
                        found_move = list(new)

    if found_move is not None:
        return found_move, found_mse, True
    return best, best_mse, False


# ── Load current best ────────────────────────────────────────────
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
best_pairs = [(perm[i], perm[i + 1]) for i in range(0, 96, 2)]
best_mse = evaluate(best_pairs, fast=False)
print(f"Starting MSE: {best_mse:.8f}")

# ── Phase 1: Exhaust 2-way moves ────────────────────────────────
print(f"\n{'='*60}")
print("Phase 1: 2-way local search (subsampled)")
print(f"{'='*60}")
t0 = time.time()
rnd = 0
while True:
    rnd += 1
    best_pairs, mse, improved = twoway_search(best_pairs, X_sub, pred_sub)
    print(f"  2-way round {rnd}: fast MSE = {mse:.8f}  improved={improved}")
    if not improved:
        break

full_mse = evaluate(best_pairs, fast=False)
print(f"After 2-way: full MSE = {full_mse:.8f} ({time.time()-t0:.0f}s)")
if full_mse < best_mse:
    best_mse = full_mse
    save(best_pairs)

# ── Phase 2: Alternate 3-way and 2-way ──────────────────────────
print(f"\n{'='*60}")
print("Phase 2: 3-way cross-swap + rotation search")
print(f"{'='*60}")

any_3way = True
while any_3way:
    any_3way = False

    # Cross-swap search
    t1 = time.time()
    print("\n  Cross-swap search (103K moves)...")
    best_pairs, mse, improved, count = crossswap_search(best_pairs, X_sub, pred_sub)
    print(f"    {count} evals in {time.time()-t1:.0f}s, fast MSE = {mse:.8f}, improved={improved}")
    if improved:
        any_3way = True
        full_mse = evaluate(best_pairs, fast=False)
        print(f"    Full MSE = {full_mse:.8f}")
        if full_mse < best_mse:
            best_mse = full_mse
            save(best_pairs)
            print(f"    Saved!")

        # Do 2-way after 3-way improvement
        rnd = 0
        while True:
            rnd += 1
            best_pairs, mse, imp2 = twoway_search(best_pairs, X_sub, pred_sub)
            print(f"    2-way round {rnd}: fast MSE = {mse:.8f}")
            if not imp2:
                break
        full_mse = evaluate(best_pairs, fast=False)
        if full_mse < best_mse:
            best_mse = full_mse
            save(best_pairs)
            print(f"    Full MSE = {full_mse:.8f} — saved!")

    # Rotation search
    t1 = time.time()
    print("\n  Rotation search (34K moves)...")
    best_pairs, mse, improved = rotation_search(best_pairs, X_sub, pred_sub)
    print(f"    {time.time()-t1:.0f}s, fast MSE = {mse:.8f}, improved={improved}")
    if improved:
        any_3way = True
        full_mse = evaluate(best_pairs, fast=False)
        print(f"    Full MSE = {full_mse:.8f}")
        if full_mse < best_mse:
            best_mse = full_mse
            save(best_pairs)
            print(f"    Saved!")

        rnd = 0
        while True:
            rnd += 1
            best_pairs, mse, imp2 = twoway_search(best_pairs, X_sub, pred_sub)
            print(f"    2-way round {rnd}: fast MSE = {mse:.8f}")
            if not imp2:
                break
        full_mse = evaluate(best_pairs, fast=False)
        if full_mse < best_mse:
            best_mse = full_mse
            save(best_pairs)

# ── Phase 3: Final polish on full data ───────────────────────────
print(f"\n{'='*60}")
print("Phase 3: Full-data 2-way local search")
print(f"{'='*60}")
t3 = time.time()
rnd = 0
while True:
    rnd += 1
    best_pairs, mse, improved = twoway_search(best_pairs, X_full, pred_full)
    print(f"  Round {rnd}: MSE = {mse:.10f}  improved={improved}")
    if not improved:
        break

best_mse = mse
save(best_pairs)
print(f"\nFinal MSE: {best_mse:.10f}")
print(f"Total time: {time.time()-t0:.0f}s")
