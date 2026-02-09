"""
Break through MSE plateau with analysis + new neighborhoods + targeted search.

Phase 1: Leave-one-out analysis — identify harmful/suspect blocks
Phase 2: New neighborhood local search (or-opt, 2-opt, 2-way, 3-way)
Phase 3: Targeted exhaustive re-pairing/re-ordering of worst blocks
"""
import torch
import pandas as pd
import time
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load data ─────────────────────────────────────────────────
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

inp_wt = {i: (pieces[i]['weight'], pieces[i]['bias']) for i in inp_layers}
out_wt = {i: (pieces[i]['weight'], pieces[i]['bias']) for i in out_layers}
ll_w, ll_b = pieces[last_layer_idx]['weight'], pieces[last_layer_idx]['bias']
ll_wt = ll_w.T

inp_sorted = sorted(inp_layers)
out_sorted = sorted(out_layers)
inp_id2seq = {pid: idx for idx, pid in enumerate(inp_sorted)}
out_id2seq = {pid: idx for idx, pid in enumerate(out_sorted)}

all_inp_w = torch.stack([inp_wt[p][0] for p in inp_sorted])
all_inp_b = torch.stack([inp_wt[p][1] for p in inp_sorted])
all_out_w = torch.stack([out_wt[p][0] for p in out_sorted])
all_out_b = torch.stack([out_wt[p][1] for p in out_sorted])
all_inp_wt = all_inp_w.transpose(1, 2).contiguous()
all_out_wt = all_out_w.transpose(1, 2).contiguous()


@torch.no_grad()
def apply_block(x, inp_pid, out_pid):
    iw, ib = inp_wt[inp_pid]
    ow, ob = out_wt[out_pid]
    return x + torch.relu(x @ iw.T + ib) @ ow.T + ob


@torch.no_grad()
def apply_block_batched(x, inp_seq, out_seq):
    iwt = all_inp_wt[inp_seq]
    ib = all_inp_b[inp_seq]
    owt = all_out_wt[out_seq]
    ob = all_out_b[out_seq]
    h = torch.bmm(x, iwt) + ib.unsqueeze(1)
    h = torch.relu(h)
    h = torch.bmm(h, owt) + ob.unsqueeze(1)
    return x + h


def pairs_to_seq(pairs):
    ci = torch.tensor([inp_id2seq[a] for a, b in pairs], dtype=torch.long, device=device)
    co = torch.tensor([out_id2seq[b] for a, b in pairs], dtype=torch.long, device=device)
    return ci, co


def seq_to_pairs(ci, co):
    return [(inp_sorted[ci[k].item()], out_sorted[co[k].item()]) for k in range(48)]


@torch.no_grad()
def compute_states(pairs, X):
    states = [X]
    x = X
    for a, b in pairs:
        x = apply_block(x, a, b)
        states.append(x)
    return states


@torch.no_grad()
def eval_mse(pairs, X, p):
    x = X
    for a, b in pairs:
        x = apply_block(x, a, b)
    return ((x @ ll_wt + ll_b).squeeze() - p).pow(2).mean().item()


@torch.no_grad()
def batched_eval(all_ci, all_co, start_state, start_pos, p, chunk_size=64):
    N = all_ci.shape[0]
    results = torch.empty(N, device=device)
    for cs in range(0, N, chunk_size):
        ce = min(cs + chunk_size, N)
        ci = all_ci[cs:ce]
        co = all_co[cs:ce]
        B = ce - cs
        x = start_state.unsqueeze(0).expand(B, -1, -1).clone()
        for k in range(start_pos, 48):
            x = apply_block_batched(x, ci[:, k], co[:, k])
        out = (x @ ll_wt + ll_b).squeeze(-1)
        results[cs:ce] = (out - p).pow(2).mean(dim=1)
    return results


def save(pairs):
    perm = []
    for a, b in pairs:
        perm.extend([a, b])
    perm.append(last_layer_idx)
    with open("solution.txt", "w") as f:
        f.write(",".join(map(str, perm)))


# ── Load current best ────────────────────────────────────────
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
best_pairs = [(perm[i], perm[i + 1]) for i in range(0, 96, 2)]
best_mse = eval_mse(best_pairs, X_full, pred_full)
print(f"Current MSE: {best_mse:.10f}\n")


# ══════════════════════════════════════════════════════════════
# PHASE 1: Leave-one-out analysis
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: Leave-one-out analysis")
print("=" * 60)
t1 = time.time()

loo_results = []
for k in range(48):
    pairs_skip = best_pairs[:k] + best_pairs[k + 1:]
    x = X_full
    for a, b in pairs_skip:
        x = apply_block(x, a, b)
    mse_skip = ((x @ ll_wt + ll_b).squeeze() - pred_full).pow(2).mean().item()
    delta = mse_skip - best_mse
    loo_results.append((k, delta, mse_skip, best_pairs[k]))

# Sort by delta (most harmful first = most negative delta)
loo_results.sort(key=lambda x: x[1])

print(f"\nBlocks sorted by skip-delta (negative = harmful, removal helps):")
suspect_positions = []
for k, delta, mse_skip, pair in loo_results:
    marker = ""
    if delta < -1e-5:
        marker = " *** HARMFUL ***"
        suspect_positions.append(k)
    elif delta < 0.001:
        marker = " (weak)"
        suspect_positions.append(k)
    if marker or delta < 0.005:
        print(f"  pos {k:2d} pair={pair}: delta={delta:+.6f} skip_MSE={mse_skip:.6f}{marker}")

print(f"\n{len(suspect_positions)} suspect positions: {suspect_positions}")
print(f"Phase 1: {time.time()-t1:.1f}s\n")


# ══════════════════════════════════════════════════════════════
# PHASE 2: New neighborhood local search (full data)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 2: Or-opt + 2-opt + 2-way + 3-way local search")
print("=" * 60)
t2 = time.time()

best_ci, best_co = pairs_to_seq(best_pairs)
CHUNK = 64  # conservative for full 10K data


def _build_2way_candidates(i, bci, bco):
    n_j = 47 - i
    if n_j == 0:
        return None, None, 0
    n_cand = n_j * 3
    js = torch.arange(i + 1, 48, device=device)
    cci = bci.unsqueeze(0).expand(n_cand, -1).clone()
    cco = bco.unsqueeze(0).expand(n_cand, -1).clone()
    m0 = torch.arange(0, n_cand, 3, device=device)
    m1 = torch.arange(1, n_cand, 3, device=device)
    m2 = torch.arange(2, n_cand, 3, device=device)
    cci[m0, i] = bci[js]; cci[m0, js] = bci[i]
    cco[m0, i] = bco[js]; cco[m0, js] = bco[i]
    cco[m1, i] = bco[js]; cco[m1, js] = bco[i]
    cci[m2, i] = bci[js]; cci[m2, js] = bci[i]
    return cci, cco, n_cand


def _build_oropt_candidates(s, bci, bco):
    """Or-opt: relocate block from position s to every other position,
    and relocate every other block to position s."""
    n_moves = 47 - s
    if n_moves == 0:
        return None, None, 0
    n_cand = 2 * n_moves  # Type A + Type B

    cci = bci.unsqueeze(0).expand(n_cand, -1).clone()
    cco = bco.unsqueeze(0).expand(n_cand, -1).clone()

    # Type A: move block s to position j (j > s)
    # New order: ..., s+1, ..., j, s, j+1, ...
    for c in range(n_moves):
        j = s + 1 + c
        cci[c, s:j] = bci[s + 1:j + 1]
        cco[c, s:j] = bco[s + 1:j + 1]
        cci[c, j] = bci[s]
        cco[c, j] = bco[s]

    # Type B: move block i (i > s) to position s
    # New order: ..., i, s, s+1, ..., i-1, i+1, ...
    off = n_moves
    for c in range(n_moves):
        i = s + 1 + c
        cci[off + c, s] = bci[i]
        cco[off + c, s] = bco[i]
        cci[off + c, s + 1:i + 1] = bci[s:i]
        cco[off + c, s + 1:i + 1] = bco[s:i]

    return cci, cco, n_cand


def _build_2opt_candidates(s, bci, bco, max_len=10):
    """2-opt: reverse segment [s..j] for j = s+2 to min(s+max_len, 47)."""
    max_j = min(s + max_len, 47)
    n_cand = max_j - s - 1  # j from s+2 to max_j
    if n_cand <= 0:
        return None, None, 0

    cci = bci.unsqueeze(0).expand(n_cand, -1).clone()
    cco = bco.unsqueeze(0).expand(n_cand, -1).clone()

    for c in range(n_cand):
        j = s + 2 + c
        cci[c, s:j + 1] = bci[s:j + 1].flip(0)
        cco[c, s:j + 1] = bco[s:j + 1].flip(0)

    return cci, cco, n_cand


def _build_3way_candidates(i, bci, bco):
    """3-way rotations at position i."""
    jj, kk = [], []
    for j in range(i + 1, 48):
        for k in range(j + 1, 48):
            jj.append(j); kk.append(k)
    if not jj:
        return None, None, 0
    jj_t = torch.tensor(jj, device=device)
    kk_t = torch.tensor(kk, device=device)
    n_cand = 2 * len(jj)
    cci = bci.unsqueeze(0).expand(n_cand, -1).clone()
    cco = bco.unsqueeze(0).expand(n_cand, -1).clone()
    d0 = torch.arange(0, n_cand, 2, device=device)
    d1 = torch.arange(1, n_cand, 2, device=device)
    cci[d0, i] = bci[kk_t]; cci[d0, jj_t] = bci[i]; cci[d0, kk_t] = bci[jj_t]
    cco[d0, i] = bco[kk_t]; cco[d0, jj_t] = bco[i]; cco[d0, kk_t] = bco[jj_t]
    cci[d1, i] = bci[jj_t]; cci[d1, jj_t] = bci[kk_t]; cci[d1, kk_t] = bci[i]
    cco[d1, i] = bco[jj_t]; cco[d1, jj_t] = bco[kk_t]; cco[d1, kk_t] = bco[i]
    return cci, cco, n_cand


def try_neighborhood(name, builder, start_pos_fn, bci, bco, states, current_mse):
    """Try all moves from a neighborhood, return best if improving."""
    found_mse = current_mse
    found_ci = found_co = None

    for i in range(48):
        cci, cco, n = builder(i, bci, bco)
        if n == 0:
            continue
        sp = start_pos_fn(i)
        mses = batched_eval(cci, cco, states[sp], sp, pred_full, chunk_size=CHUNK)
        bidx = mses.argmin().item()
        if mses[bidx] < found_mse - 1e-9:
            found_mse = mses[bidx].item()
            found_ci = cci[bidx].clone()
            found_co = cco[bidx].clone()

    return found_mse, found_ci, found_co


max_rounds = 30
for rnd in range(max_rounds):
    states = compute_states(best_pairs, X_full)
    improved = False

    # Try neighborhoods in priority order
    for name, builder, start_fn in [
        ("2-way",  _build_2way_candidates,  lambda i: i),
        ("or-opt", _build_oropt_candidates, lambda i: i),
        ("2-opt",  _build_2opt_candidates,  lambda i: i),
        ("3-way",  _build_3way_candidates,  lambda i: i),
    ]:
        t_n = time.time()
        found_mse, found_ci, found_co = try_neighborhood(
            name, builder, start_fn, best_ci, best_co, states, best_mse
        )
        if found_ci is not None:
            best_mse = found_mse
            best_ci, best_co = found_ci, found_co
            best_pairs = seq_to_pairs(best_ci, best_co)
            save(best_pairs)
            print(f"  Round {rnd:2d}: {name:6s} → MSE {best_mse:.10f} ({time.time()-t_n:.1f}s)")
            improved = True
            break  # Restart with new states
        else:
            print(f"  Round {rnd:2d}: {name:6s} — no improvement ({time.time()-t_n:.1f}s)")

    if not improved:
        print(f"  Converged after {rnd} rounds.")
        break

print(f"\nPhase 2 MSE: {best_mse:.10f} ({time.time()-t2:.0f}s)\n")


# ══════════════════════════════════════════════════════════════
# PHASE 3: Targeted exhaustive search on worst blocks
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PHASE 3: Targeted exhaustive search")
print("=" * 60)
t3 = time.time()

# Re-run LOO on the (possibly improved) solution
loo2 = []
for k in range(48):
    pairs_skip = best_pairs[:k] + best_pairs[k + 1:]
    x = X_full
    for a, b in pairs_skip:
        x = apply_block(x, a, b)
    mse_skip = ((x @ ll_wt + ll_b).squeeze() - pred_full).pow(2).mean().item()
    loo2.append((k, mse_skip - best_mse))

loo2.sort(key=lambda x: x[1])
# Take blocks with smallest delta (most harmful or least helpful)
worst_K = [pos for pos, delta in loo2[:6]]
worst_K.sort()
print(f"Targeting positions: {worst_K}")
print(f"  (deltas: {[f'{d:.6f}' for _, d in loo2[:6]]})")

K = len(worst_K)
# Collect the inp/out seq indices at those positions
worst_inp = [best_ci[p].item() for p in worst_K]
worst_out = [best_co[p].item() for p in worst_K]
start_pos = worst_K[0]

# Phase 3a: Exhaustive re-pairing (keep positions fixed, try all inp/out assignments)
print(f"\n3a: Exhaustive re-pairing of {K} blocks ({K}!={len(list(itertools.permutations(range(K))))} pairings)...")
inp_perms = list(itertools.permutations(worst_inp))
out_perms = list(itertools.permutations(worst_out))

n_cand = len(out_perms)  # Fix inp ordering, permute out
cci = best_ci.unsqueeze(0).expand(n_cand, -1).clone()
cco = best_co.unsqueeze(0).expand(n_cand, -1).clone()
for c, op in enumerate(out_perms):
    for idx, pos in enumerate(worst_K):
        cco[c, pos] = op[idx]

states = compute_states(best_pairs, X_full)
mses = batched_eval(cci, cco, states[start_pos], start_pos, pred_full, chunk_size=CHUNK)
bidx = mses.argmin().item()
if mses[bidx] < best_mse - 1e-9:
    best_mse = mses[bidx].item()
    best_ci = cci[bidx].clone()
    best_co = cco[bidx].clone()
    best_pairs = seq_to_pairs(best_ci, best_co)
    save(best_pairs)
    print(f"  ** RE-PAIR improvement: MSE {best_mse:.10f}")
else:
    print(f"  No re-pair improvement (best candidate: {mses[bidx].item():.10f})")

# Also try permuting inp (fix out)
cci2 = best_ci.unsqueeze(0).expand(len(inp_perms), -1).clone()
cco2 = best_co.unsqueeze(0).expand(len(inp_perms), -1).clone()
for c, ip in enumerate(inp_perms):
    for idx, pos in enumerate(worst_K):
        cci2[c, pos] = ip[idx]

states = compute_states(best_pairs, X_full)
mses2 = batched_eval(cci2, cco2, states[start_pos], start_pos, pred_full, chunk_size=CHUNK)
bidx2 = mses2.argmin().item()
if mses2[bidx2] < best_mse - 1e-9:
    best_mse = mses2[bidx2].item()
    best_ci = cci2[bidx2].clone()
    best_co = cco2[bidx2].clone()
    best_pairs = seq_to_pairs(best_ci, best_co)
    save(best_pairs)
    print(f"  ** RE-INP improvement: MSE {best_mse:.10f}")
else:
    print(f"  No re-inp improvement (best candidate: {mses2[bidx2].item():.10f})")

# Phase 3b: Exhaustive re-ordering (keep pairings, try all orderings of K blocks)
print(f"\n3b: Exhaustive re-ordering of {K} blocks...")
order_perms = list(itertools.permutations(range(K)))
n_cand = len(order_perms)
cci3 = best_ci.unsqueeze(0).expand(n_cand, -1).clone()
cco3 = best_co.unsqueeze(0).expand(n_cand, -1).clone()
for c, perm in enumerate(order_perms):
    for idx, pos in enumerate(worst_K):
        src = worst_K[perm[idx]]
        cci3[c, pos] = best_ci[src]
        cco3[c, pos] = best_co[src]

states = compute_states(best_pairs, X_full)
mses3 = batched_eval(cci3, cco3, states[start_pos], start_pos, pred_full, chunk_size=CHUNK)
bidx3 = mses3.argmin().item()
if mses3[bidx3] < best_mse - 1e-9:
    best_mse = mses3[bidx3].item()
    best_ci = cci3[bidx3].clone()
    best_co = cco3[bidx3].clone()
    best_pairs = seq_to_pairs(best_ci, best_co)
    save(best_pairs)
    print(f"  ** RE-ORDER improvement: MSE {best_mse:.10f}")
else:
    print(f"  No re-order improvement (best candidate: {mses3[bidx3].item():.10f})")

# Phase 3c: Full combinatorial (re-pair + re-order) if K ≤ 5
if K <= 5:
    print(f"\n3c: Full combinatorial search ({K}!² = {len(inp_perms)*len(order_perms)} candidates)...")
    all_candidates_ci = []
    all_candidates_co = []
    for op in out_perms:
        for perm in order_perms:
            ci = best_ci.clone()
            co = best_co.clone()
            for idx, pos in enumerate(worst_K):
                src_idx = perm[idx]
                ci[pos] = worst_inp[src_idx]
                co[pos] = op[src_idx]
            all_candidates_ci.append(ci)
            all_candidates_co.append(co)

    all_ci_t = torch.stack(all_candidates_ci)
    all_co_t = torch.stack(all_candidates_co)
    states = compute_states(best_pairs, X_full)
    mses4 = batched_eval(all_ci_t, all_co_t, states[start_pos], start_pos, pred_full,
                         chunk_size=CHUNK)
    bidx4 = mses4.argmin().item()
    if mses4[bidx4] < best_mse - 1e-9:
        best_mse = mses4[bidx4].item()
        best_ci = all_ci_t[bidx4].clone()
        best_co = all_co_t[bidx4].clone()
        best_pairs = seq_to_pairs(best_ci, best_co)
        save(best_pairs)
        print(f"  ** FULL COMBO improvement: MSE {best_mse:.10f}")
    else:
        print(f"  No combo improvement (best candidate: {mses4[bidx4].item():.10f})")
else:
    print(f"\n3c: Skipped full combinatorial (K={K} > 5, would be {len(inp_perms)**2} candidates)")

print(f"\nPhase 3: {time.time()-t3:.0f}s")
print(f"\n{'='*60}")
print(f"FINAL MSE: {best_mse:.10f}")
print(f"Total time: {time.time()-t1:.0f}s")
print(f"{'='*60}")
