"""
ILS with rotating subsample — GPU-batched local search.

- local_search_fast: FP16 first-improvement, 2-way + or-opt fallback (ILS restarts)
- local_search_batched: FP32 best-improvement, 2-way + 3-way (final polish)
"""
import torch
import pandas as pd
import time
import random

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
ll_wt = ll_w.T  # (48, 1) pre-transposed

# ── Stacked weights for batched ops ──────────────────────────
inp_sorted = sorted(inp_layers)
out_sorted = sorted(out_layers)
inp_id2seq = {pid: idx for idx, pid in enumerate(inp_sorted)}
out_id2seq = {pid: idx for idx, pid in enumerate(out_sorted)}

all_inp_w = torch.stack([inp_wt[p][0] for p in inp_sorted])  # (48, 96, 48)
all_inp_b = torch.stack([inp_wt[p][1] for p in inp_sorted])  # (48, 96)
all_out_w = torch.stack([out_wt[p][0] for p in out_sorted])  # (48, 48, 96)
all_out_b = torch.stack([out_wt[p][1] for p in out_sorted])  # (48, 48)

# Pre-transpose for bmm
all_inp_wt = all_inp_w.transpose(1, 2).contiguous()  # (48, 48, 96)
all_out_wt = all_out_w.transpose(1, 2).contiguous()   # (48, 96, 48)

# ── FP16 copies for fast subsample evaluation ─────────────────
all_inp_wt_f16 = all_inp_wt.half()
all_inp_b_f16 = all_inp_b.half()
all_out_wt_f16 = all_out_wt.half()
all_out_b_f16 = all_out_b.half()
inp_wt_f16 = {i: (w.half(), b.half()) for i, (w, b) in inp_wt.items()}
out_wt_f16 = {i: (w.half(), b.half()) for i, (w, b) in out_wt.items()}
ll_wt_f16 = ll_wt.half()
ll_b_f16 = ll_b.half()


@torch.no_grad()
def apply_block(x, inp_pid, out_pid):
    iw, ib = inp_wt[inp_pid]
    ow, ob = out_wt[out_pid]
    return x + torch.relu(x @ iw.T + ib) @ ow.T + ob


@torch.no_grad()
def apply_block_batched(x, inp_seq, out_seq):
    """x: (B, S, 48), indices: (B,) long → (B, S, 48)"""
    iwt = all_inp_wt[inp_seq]
    ib = all_inp_b[inp_seq]
    owt = all_out_wt[out_seq]
    ob = all_out_b[out_seq]
    h = torch.bmm(x, iwt) + ib.unsqueeze(1)
    h = torch.relu(h)
    h = torch.bmm(h, owt) + ob.unsqueeze(1)
    return x + h


@torch.no_grad()
def apply_block_f16(x, inp_pid, out_pid):
    iw, ib = inp_wt_f16[inp_pid]
    ow, ob = out_wt_f16[out_pid]
    return x + torch.relu(x @ iw.T + ib) @ ow.T + ob


@torch.no_grad()
def apply_block_batched_f16(x, inp_seq, out_seq):
    """FP16 batched block. Uses tensor cores on RTX 20xx+/H100."""
    iwt = all_inp_wt_f16[inp_seq]
    ib = all_inp_b_f16[inp_seq]
    owt = all_out_wt_f16[out_seq]
    ob = all_out_b_f16[out_seq]
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
def compute_states_f16(pairs, X):
    """Compute prefix states in FP16."""
    x = X.half()
    states = [x]
    for a, b in pairs:
        x = apply_block_f16(x, a, b)
        states.append(x)
    return states


@torch.no_grad()
def eval_mse(pairs, X, p):
    x = X
    for a, b in pairs:
        x = apply_block(x, a, b)
    return ((x @ ll_wt + ll_b).squeeze() - p).pow(2).mean().item()


@torch.no_grad()
def eval_mse_f16(pairs, X, p):
    """FP16 forward pass, FP32 final MSE."""
    x = X.half()
    for a, b in pairs:
        x = apply_block_f16(x, a, b)
    return ((x.float() @ ll_wt + ll_b).squeeze() - p).pow(2).mean().item()


@torch.no_grad()
def batched_eval(all_ci, all_co, start_state, start_pos, p, chunk_size=256):
    """FP32 batched eval."""
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


@torch.no_grad()
def batched_eval_f16(all_ci, all_co, start_state, start_pos, p, chunk_size=512):
    """FP16 batched eval. 2-4x faster via tensor cores, larger chunks (half memory)."""
    N = all_ci.shape[0]
    results = torch.empty(N, device=device)

    for cs in range(0, N, chunk_size):
        ce = min(cs + chunk_size, N)
        ci = all_ci[cs:ce]
        co = all_co[cs:ce]
        B = ce - cs

        x = start_state.unsqueeze(0).expand(B, -1, -1).clone()
        for k in range(start_pos, 48):
            x = apply_block_batched_f16(x, ci[:, k], co[:, k])

        # Final MSE in fp32 for precision
        out = (x.float() @ ll_wt + ll_b).squeeze(-1)
        results[cs:ce] = (out - p).pow(2).mean(dim=1)

    return results


def _build_2way_candidates(i, best_ci, best_co):
    """Build candidate index tensors for all 2-way moves at position i."""
    n_j = 47 - i
    if n_j == 0:
        return None, None, 0
    n_cand = n_j * 3
    js = torch.arange(i + 1, 48, device=device)

    cand_ci = best_ci.unsqueeze(0).expand(n_cand, -1).clone()
    cand_co = best_co.unsqueeze(0).expand(n_cand, -1).clone()

    m0 = torch.arange(0, n_cand, 3, device=device)
    m1 = torch.arange(1, n_cand, 3, device=device)
    m2 = torch.arange(2, n_cand, 3, device=device)

    # Move 0: swap full blocks at i,j
    cand_ci[m0, i] = best_ci[js]; cand_ci[m0, js] = best_ci[i]
    cand_co[m0, i] = best_co[js]; cand_co[m0, js] = best_co[i]
    # Move 1: swap out layers only
    cand_co[m1, i] = best_co[js]; cand_co[m1, js] = best_co[i]
    # Move 2: swap inp layers only
    cand_ci[m2, i] = best_ci[js]; cand_ci[m2, js] = best_ci[i]

    return cand_ci, cand_co, n_cand


def _build_oropt_candidates(s, best_ci, best_co):
    """Or-opt: relocate block from/to position s."""
    n_moves = 47 - s
    if n_moves == 0:
        return None, None, 0
    n_cand = 2 * n_moves
    cci = best_ci.unsqueeze(0).expand(n_cand, -1).clone()
    cco = best_co.unsqueeze(0).expand(n_cand, -1).clone()
    # Type A: move block s to position j (j > s)
    for c in range(n_moves):
        j = s + 1 + c
        cci[c, s:j] = best_ci[s + 1:j + 1]
        cco[c, s:j] = best_co[s + 1:j + 1]
        cci[c, j] = best_ci[s]
        cco[c, j] = best_co[s]
    # Type B: move block i (i > s) to position s
    off = n_moves
    for c in range(n_moves):
        i = s + 1 + c
        cci[off + c, s] = best_ci[i]
        cco[off + c, s] = best_co[i]
        cci[off + c, s + 1:i + 1] = best_ci[s:i]
        cco[off + c, s + 1:i + 1] = best_co[s:i]
    return cci, cco, n_cand


def local_search_fast(pairs, X, p, max_rounds=10, chunk_2=512):
    """FP16 first-improvement: 2-way + or-opt fallback."""
    best = list(pairs)
    best_mse = eval_mse_f16(best, X, p)
    best_ci, best_co = pairs_to_seq(best)

    for rnd in range(max_rounds):
        states = compute_states_f16(best, X)
        improved = False

        positions = list(range(48))
        random.shuffle(positions)

        # 2-way first-improvement
        for i in positions:
            cand_ci, cand_co, n_cand = _build_2way_candidates(i, best_ci, best_co)
            if n_cand == 0:
                continue
            mses = batched_eval_f16(cand_ci, cand_co, states[i], i, p, chunk_size=chunk_2)
            bidx = mses.argmin().item()
            if mses[bidx] < best_mse - 1e-7:
                best_mse = mses[bidx].item()
                best_ci = cand_ci[bidx].clone()
                best_co = cand_co[bidx].clone()
                best = seq_to_pairs(best_ci, best_co)
                improved = True
                break

        # Or-opt fallback when 2-way exhausted
        if not improved:
            random.shuffle(positions)
            for i in positions:
                cci, cco, n = _build_oropt_candidates(i, best_ci, best_co)
                if n == 0:
                    continue
                mses = batched_eval_f16(cci, cco, states[i], i, p, chunk_size=chunk_2)
                bidx = mses.argmin().item()
                if mses[bidx] < best_mse - 1e-7:
                    best_mse = mses[bidx].item()
                    best_ci = cci[bidx].clone()
                    best_co = cco[bidx].clone()
                    best = seq_to_pairs(best_ci, best_co)
                    improved = True
                    break

        if not improved:
            break

    return best, best_mse


def local_search_batched(pairs, X, p, max_rounds=15, chunk_2=256, chunk_3=100):
    """Best-improvement 2-way + 3-way fallback. Thorough for final polish."""
    best = list(pairs)
    best_mse = eval_mse(best, X, p)
    best_ci, best_co = pairs_to_seq(best)

    for rnd in range(max_rounds):
        states = compute_states(best, X)
        found_mse = best_mse
        found_ci = found_co = None

        # ── 2-way moves ──────────────────────────────────────
        for i in range(48):
            cand_ci, cand_co, n_cand = _build_2way_candidates(i, best_ci, best_co)
            if n_cand == 0:
                continue

            mses = batched_eval(cand_ci, cand_co, states[i], i, p, chunk_size=chunk_2)
            bidx = mses.argmin().item()
            if mses[bidx] < found_mse - 1e-9:
                found_mse = mses[bidx].item()
                found_ci = cand_ci[bidx].clone()
                found_co = cand_co[bidx].clone()

        # ── 3-way rotations (only if no 2-way found) ─────────
        if found_ci is None:
            for i in range(48):
                jj, kk = [], []
                for j in range(i + 1, 48):
                    for k in range(j + 1, 48):
                        jj.append(j); kk.append(k)
                if not jj:
                    continue
                jj_t = torch.tensor(jj, device=device)
                kk_t = torch.tensor(kk, device=device)
                n_cand = 2 * len(jj)

                cand_ci = best_ci.unsqueeze(0).expand(n_cand, -1).clone()
                cand_co = best_co.unsqueeze(0).expand(n_cand, -1).clone()

                d0 = torch.arange(0, n_cand, 2, device=device)
                d1 = torch.arange(1, n_cand, 2, device=device)

                cand_ci[d0, i] = best_ci[kk_t]
                cand_ci[d0, jj_t] = best_ci[i]
                cand_ci[d0, kk_t] = best_ci[jj_t]
                cand_co[d0, i] = best_co[kk_t]
                cand_co[d0, jj_t] = best_co[i]
                cand_co[d0, kk_t] = best_co[jj_t]
                cand_ci[d1, i] = best_ci[jj_t]
                cand_ci[d1, jj_t] = best_ci[kk_t]
                cand_ci[d1, kk_t] = best_ci[i]
                cand_co[d1, i] = best_co[jj_t]
                cand_co[d1, jj_t] = best_co[kk_t]
                cand_co[d1, kk_t] = best_co[i]

                mses = batched_eval(cand_ci, cand_co, states[i], i, p, chunk_size=chunk_3)
                bidx = mses.argmin().item()
                if mses[bidx] < found_mse - 1e-9:
                    found_mse = mses[bidx].item()
                    found_ci = cand_ci[bidx].clone()
                    found_co = cand_co[bidx].clone()

        if found_ci is not None:
            best_ci, best_co = found_ci, found_co
            best = seq_to_pairs(best_ci, best_co)
            best_mse = found_mse
        else:
            break

    return best, best_mse


def perturb(pairs, intensity):
    pairs = list(pairs)
    for _ in range(intensity):
        move = random.choice(['swap_order', 'swap_inp', 'swap_out', 'rotate3', 'shuffle_seg'])
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


# ── Load current best ────────────────────────────────────────
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
best_pairs = [(perm[i], perm[i + 1]) for i in range(0, 96, 2)]
best_mse = eval_mse(best_pairs, X_full, pred_full)
print(f"Starting MSE: {best_mse:.10f}")

# ── ILS with rotating subsample ──────────────────────────────
n_restarts = 500
SUB = 3000
print(f"\n{'='*60}")
print(f"ILS (fast): {n_restarts} restarts, subsample {SUB}")
print(f"{'='*60}")
t0 = time.time()
improvements = 0

for restart in range(n_restarts):
    idx = torch.randperm(len(X_full), device=device)[:SUB]
    X_sub = X_full[idx]
    pred_sub = pred_full[idx]

    # Build a tiny candidate pool each restart, then full-verify top candidates.
    candidates = []
    for _ in range(3):
        intensity = random.choices([1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
                                   weights=[18, 18, 14, 12, 10, 8, 7, 6, 5, 2])[0]
        pairs = perturb(best_pairs, intensity)

        # Periodically run short batched best-improvement to inject 3-way moves.
        if restart % 25 == 0:
            pairs, mse_sub = local_search_batched(
                pairs, X_sub, pred_sub, max_rounds=4, chunk_2=128, chunk_3=64
            )
        else:
            pairs, mse_sub = local_search_fast(pairs, X_sub, pred_sub, max_rounds=20)

        candidates.append((mse_sub, pairs, intensity))

    candidates.sort(key=lambda x: x[0])
    report_sub = candidates[0][0]

    if restart % 10 == 0:
        elapsed = time.time() - t0
        rate = (restart + 1) / elapsed if elapsed > 0 else 0
        eta = (n_restarts - restart - 1) / rate if rate > 0 else 0
        print(f"  [{restart:3d}/{n_restarts}] sub={report_sub:.6f}  "
              f"best={best_mse:.10f}  {elapsed:.0f}s  ETA:{eta:.0f}s  {improvements}imp")

    # Verify top-2 candidates on full data to reduce subsample noise.
    for mse_sub, pairs, intensity in candidates[:2]:
        full_mse = eval_mse(pairs, X_full, pred_full)
        if full_mse < best_mse - 1e-9:
            best_mse = full_mse
            best_pairs = pairs
            save(best_pairs)
            improvements += 1
            print(f"  ** Restart {restart}: NEW BEST {best_mse:.10f} "
                  f"(intensity={intensity}, sub={mse_sub:.6f}) **")
            break

elapsed = time.time() - t0
print(f"\nILS done: {elapsed:.0f}s, {improvements} improvements")

# ── Final full-data polish ────────────────────────────────────
print(f"\nFinal full-data polish (best-improvement + 3-way)...")
t1 = time.time()
best_pairs, best_mse = local_search_batched(best_pairs, X_full, pred_full,
                                            max_rounds=30, chunk_2=64, chunk_3=32)
save(best_pairs)
print(f"Final MSE: {best_mse:.10f} ({time.time()-t1:.0f}s)")
print(f"Total: {time.time()-t0:.0f}s")
