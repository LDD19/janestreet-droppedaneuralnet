"""
ILS with full data — no subsampling.
Previous runs showed subsample (1000) diverges from full data (0.013 vs 0.023).
Full-data search found 21 rounds of improvements the subsample missed.
"""
import torch
import pandas as pd
import time
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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
X = torch.tensor(df.iloc[:, :48].values, dtype=torch.float32, device=device)
pred = torch.tensor(df['pred'].values, dtype=torch.float32, device=device)

inp_wt = {i: (pieces[i]['weight'], pieces[i]['bias']) for i in inp_layers}
out_wt = {i: (pieces[i]['weight'], pieces[i]['bias']) for i in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']


@torch.no_grad()
def apply_block(x, inp_idx, out_idx):
    iw, ib = inp_wt[inp_idx]
    ow, ob = out_wt[out_idx]
    return x + torch.relu(x @ iw.T + ib) @ ow.T + ob


@torch.no_grad()
def compute_states(pairs):
    states = [X.clone()]
    x = states[0]
    for a, b in pairs:
        x = apply_block(x, a, b)
        states.append(x.clone())
    return states


@torch.no_grad()
def eval_from(pairs, start_state, start_pos):
    x = start_state.clone()
    for k in range(start_pos, len(pairs)):
        x = apply_block(x, pairs[k][0], pairs[k][1])
    return ((x @ ll_w.T + ll_b).squeeze() - pred).pow(2).mean().item()


def evaluate(pairs):
    return eval_from(pairs, X, 0)


def save(pairs):
    perm = []
    for a, b in pairs:
        perm.extend([a, b])
    perm.append(last_layer_idx)
    with open("solution.txt", "w") as f:
        f.write(",".join(map(str, perm)))


def local_search(pairs, max_rounds=30):
    """Best-improvement 2-way + rotation search on FULL data, prefix cached."""
    best = list(pairs)
    best_mse = evaluate(best)

    for rnd in range(max_rounds):
        states = compute_states(best)
        found_move = None
        found_mse = best_mse

        # 2-way swaps
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
                    mse = eval_from(new, prefix, i)
                    if mse < found_mse - 1e-9:
                        found_mse = mse
                        found_move = list(new)

        # 3-way rotations (only if no 2-way improvement)
        if found_move is None:
            for i in range(48):
                for j in range(i + 1, 48):
                    for k in range(j + 1, 48):
                        for d in range(2):
                            new = list(best)
                            if d == 0:
                                new[i], new[j], new[k] = best[k], best[i], best[j]
                            else:
                                new[i], new[j], new[k] = best[j], best[k], best[i]
                            mse = eval_from(new, states[i], i)
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
        move = random.choice(['swap_order', 'swap_inp', 'swap_out', 'rotate3'])
        if move == 'rotate3':
            a, b, c = random.sample(range(48), 3)
            pairs[a], pairs[b], pairs[c] = pairs[c], pairs[a], pairs[b]
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


# ── Load current best ────────────────────────────────────────────
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
best_pairs = [(perm[i], perm[i + 1]) for i in range(0, 96, 2)]
best_mse = evaluate(best_pairs)
print(f"Starting MSE: {best_mse:.10f}")

# ── Initial polish ───────────────────────────────────────────────
print(f"\nInitial full-data local search...")
t0 = time.time()
best_pairs, best_mse = local_search(best_pairs)
save(best_pairs)
print(f"After polish: {best_mse:.10f} ({time.time()-t0:.0f}s)")

# ── ILS on full data ─────────────────────────────────────────────
n_restarts = 50
print(f"\n{'='*60}")
print(f"ILS: {n_restarts} restarts on FULL data")
print(f"{'='*60}")
t1 = time.time()
improvements = 0

for restart in range(n_restarts):
    # Small perturbations — we're close
    intensity = random.choices([1, 2, 3, 4, 5, 8],
                               weights=[30, 30, 20, 10, 5, 5])[0]

    pairs = perturb(best_pairs, intensity)
    pairs, mse = local_search(pairs, max_rounds=15)

    elapsed = time.time() - t1
    rate = (restart + 1) / elapsed if elapsed > 0 else 0
    eta = (n_restarts - restart - 1) / rate if rate > 0 else 0

    if restart % 5 == 0:
        print(f"  [{restart:2d}/{n_restarts}] MSE={mse:.10f}  "
              f"best={best_mse:.10f}  {elapsed:.0f}s  ETA:{eta:.0f}s")

    if mse < best_mse - 1e-9:
        best_mse = mse
        best_pairs = pairs
        save(best_pairs)
        improvements += 1
        print(f"  ** Restart {restart}: NEW BEST {best_mse:.10f} (intensity={intensity}) **")

print(f"\nILS done: {time.time()-t1:.0f}s, {improvements} improvements")
print(f"Final MSE: {best_mse:.10f}")
