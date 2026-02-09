# Dropped Neural Net Puzzle - Findings

## Problem Structure

The neural network consists of:
- **97 pieces** (individual linear layers)
- **48 input layers** (48→96): These are the `inp` layers of Block modules
- **48 output layers** (96→48): These are the `out` layers of Block modules
- **1 LastLayer** (48→1): Piece 85 - the final output layer

## Block Structure

Each Block is a residual block:
```python
def forward(self, x):
    residual = x
    x = self.inp(x)      # 48 → 96
    x = self.activation(x)  # ReLU
    x = self.out(x)      # 96 → 48
    return residual + x
```

## Data Structure

- Input: 48 measurements (columns 0-47)
- Output: 1 prediction value
- `pred` column: The model's original predictions (what we need to match)
- `true` column: Ground truth values
- 10,000 data samples

## Solution Approach

Need to find a permutation of 97 positions where:
- Positions 0,2,4,...,94 (even) contain inp layers (48→96)
- Positions 1,3,5,...,95 (odd) contain out layers (96→48)
- Position 96 contains the LastLayer (piece 85)

This means finding:
1. Correct pairing of inp/out layers into 48 Blocks
2. Correct ordering of the 48 Blocks

## Layer Classification

### inp layers (48→96):
0, 1, 2, 3, 4, 5, 10, 13, 14, 15, 16, 18, 23, 27, 28, 31, 35, 37, 39, 41, 42, 43, 44, 45, 48, 49, 50, 56, 58, 59, 60, 61, 62, 64, 65, 68, 69, 73, 74, 77, 81, 84, 86, 87, 88, 91, 94, 95

### out layers (96→48):
6, 7, 8, 9, 11, 12, 17, 19, 20, 21, 22, 24, 25, 26, 29, 30, 32, 33, 34, 36, 38, 40, 46, 47, 51, 52, 53, 54, 55, 57, 63, 66, 67, 70, 71, 72, 75, 76, 78, 79, 80, 82, 83, 89, 90, 92, 93, 96

### LastLayer:
Piece 85 (48→1)

## Search Strategy

Using simulated annealing to find optimal pairing and ordering by minimizing MSE against the `pred` column.

## Results

### Attempt 1: Simulated Annealing (solve.py)
- Got stuck at MSE ~0.667
- Too slow, wrong approach

### Attempt 2: Hungarian + Greedy + Local Search (solve_v2.py)
- Hungarian algorithm for pairing: finds optimal 1-to-1 matching
- Greedy ordering: MSE = 1.38
- Local search (swap pairs): MSE = 0.231
- Still not near 0 - pairing might be wrong

### Attempt 3: Extended Local Search (solve_v3.py)
- Starting from v2 solution
- Added: out-layer swaps, inp-layer swaps, 3-way rotations
- MSE: 0.101504 (after 14 rounds, 361s)

### Attempt 4: Deep Local Search (solve_deep.py)
- Exhaustive 2-swaps and 3-swaps from current best
- No improvement found - stuck at local minimum 0.101504

### Attempt 5: Weight-based Matching (solve_match.py)
- Tried: ||out@inp - I||, trace, weight stats, spectral norms
- All much worse than current best (9-300+ MSE)
- Weight properties don't identify correct pairs

### Attempt 6: Simple Patterns (check_simple.py)
- Checked index-based patterns, offsets, modular patterns
- No simple arithmetic relationship between correct pairs
- Offset analysis: max 28 matches at offset 6 (need 48)

### Attempt 7: Projected Contribution Analysis (solve_structure.py)
Exploits the residual network structure. In first-order approximation, blocks
commute: `x_final ≈ x_0 + Σ_k residual_k(x_0)`. Each pair's contribution to
the final prediction can be precomputed by projecting through the last layer.

**Phase 1 — Three pairing methods tried:**
- **Least-squares + Hungarian**: MSE = 20.91 (poor — M matrix likely ill-conditioned)
- **Greedy projected matching**: MSE = 684.65 (terrible — greedy is suboptimal here)
- **Correlation + Hungarian**: MSE = 0.718 (best new pairing, but worse than previous best)

**Conclusion**: First-order approximation is too weak for this network. The residuals
are NOT small — blocks significantly modify the state, so they don't commute.
The pairing methods based on projecting through the last layer can't capture the
sequential dependencies.

**Phase 2 — Greedy ordering** on previous best improved MSE: 0.101504 → 0.098668

**Phase 3 — Local search** (2-way swaps + 3-way rotations): 0.098668 → **0.094918**

### Attempt 8: Multi-Strategy Escape (solve_phase2.py)
Multiple strategies to escape the local minimum:
1. **Leave-one-out analysis**: Identify which blocks are hurting predictions
   (skip each block, see if MSE decreases — suspicious blocks are likely mis-paired)
2. **Targeted 3-way search**: Exhaustive 3-way rearrangements focused on suspicious blocks
3. **Iterated Local Search (100 restarts)**: Random perturbations of varying intensity
   (2-15 moves) + fast local search using 2000-sample subsample
4. **Full greedy rebuild**: Try ALL available (inp, out) pairs at each position
   (forward and backward), using current best as proxy for remaining blocks

**Results (5945s total):**

| Strategy | Result |
|----------|--------|
| Leave-one-out | All blocks have positive delta (all helping). 4 suspicious (delta < 0.001) |
| Targeted 3-way | Marginal — 0.095 on full data (no real improvement) |
| **ILS (100 restarts)** | **0.095 → 0.027** — clear winner, kept finding new basins throughout |
| Greedy rebuild (fwd) | 0.107 raw → 0.032 after local search |
| Greedy rebuild (bwd) | 0.490 raw (much worse) |
| Final full-data local search | **0.0268** |

**Key finding**: ILS kept improving throughout 100 restarts (last improvement at restart 96),
suggesting more restarts would find even better solutions. The perturbation + local search
approach successfully escapes local minima that pure local search cannot.

### Attempt 9: Extended ILS (solve_ils_extended.py)
- 200 restarts, best-improvement + prefix caching, SUB=1000
- Small perturbations (mostly intensity 1-3)
- MSE: 0.027 → **0.0219** (11 improvements, 8267s)
- Improvements concentrated in low-intensity perturbations (1-2 moves)

### Attempt 10: 3-Way Cross-Swap + Rotation (solve_crossswap.py)
Tried two new 3-position move types:
- **Cross-swap(k,a,b)**: k gets (inp_a, out_b), a gets (inp_k, out_a), b gets (inp_b, out_k)
  — 103,776 moves per pass. **Found nothing** in 3 passes.
- **3-way rotations**: 34,592 moves per pass. **Found 2 improvements** that unlocked
  further 2-way gains.
- Full-data polish: 21 rounds of 2-way improvements
- MSE: 0.0207 → **0.0194**

**Critical finding — subsampling is misleading:**
Fast MSE (1000 samples) was 0.013 while full MSE was 0.023. The subsample overfits —
solutions that look good on 1000 samples don't generalize. Phase 3 (full data) found
21 rounds of improvements the subsample completely missed. **Must use full data for
accurate evaluation.**

### Attempt 11: Full-Data ILS (solve_ils_fulldata.py)
- 50 restarts, full 10K data, best-improvement + prefix caching
- Small perturbations (intensity 1-5)
- MSE: 0.0193 → **0.0189** (only 1 improvement in 50 restarts, 8981s)
- Deeply stuck — full-data search is accurate but too slow to explore broadly

### Attempt 12: Rotating Subsample ILS (solve_ils_rotating.py)
- Fresh random 3000 samples each restart (prevents overfitting to fixed subsample)
- Verify ALL candidates on full data (subsample MSE unreliable across rotations)
- More aggressive perturbations (intensity 2-15) to escape the deep basin
- 500 restarts
- **Too slow**: ~118s/restart (ETA ~16 hours). Bottleneck: sequential per-candidate
  eval_from calls — thousands of tiny CUDA kernels that don't saturate the GPU.

### Attempt 13: GPU-Batched ILS (solve_ils_fast.py)
- **GPU batching**: stacks all candidate moves per prefix position into single `bmm` calls
  instead of evaluating one candidate at a time. All 2-way moves for a given position i
  (~140 candidates) evaluated in one batched kernel launch.
- **Two search modes**:
  - `local_search_fast` (ILS restarts): first-improvement 2-way only, randomized position
    order, breaks as soon as any improving move is found → many cheap rounds instead of
    one expensive best-improvement scan. No 3-way (ILS perturbations handle escape).
  - `local_search_batched` (final polish): best-improvement 2-way + 3-way fallback for
    thorough convergence on full data.
- **Enhancements** (user-tuned):
  - `shuffle_seg` perturbation: randomly shuffles a segment of 3-8 consecutive blocks
  - Candidate pool: 3 perturbation+search per restart, verify top-2 on full data
  - Periodic `local_search_batched` every 25 restarts (injects 3-way moves)
  - More intensity options (added 1 and 20), max_rounds=20 for fast search
- Same rotating subsample + full-data verification approach
- **Results (250/500 restarts, ~3400s)**: MSE 0.0143 → **0.0089** (15 improvements)
  - ~13s/restart. Improvements concentrated at intensity 1-2.
  - Steady improvement throughout, last improvement at restart 199.
  - Still slow (~13s/restart) — room for further speed optimization.

### Attempt 14: Diagnostic + New Neighborhoods (solve_breakthrough.py)
Three-phase approach to break the plateau:

**Phase 1 — Leave-one-out analysis (~6s):**
- Skip each of 48 blocks, measure MSE change
- **No harmful blocks found** — all 48 contribute positively (delta > 0)
- Error is distributed, not concentrated in a few wrong blocks
- 4 "suspicious" blocks with small positive delta, but targeted search found nothing

**Phase 2 — New neighborhood local search:**
- **Or-opt (block relocation)**: Remove block from position i, insert at position j
  - Found 2 improvements that 2-way swaps couldn't reach
  - Or-opt is a genuinely different move — one relocation equals multiple swaps
- **2-opt (segment reversal)**: Reverse contiguous segments of length 3-10
  - No improvements found (ordering already well-optimized locally)
- **2-way and 3-way**: Standard fallback
  - Additional 2-way improvements unlocked after or-opt moves

**Phase 3 — Targeted exhaustive search on worst 6 blocks:**
- Tried all K! re-pairings and re-orderings among 6 most suspicious blocks
- **Found nothing** — confirms error is not concentrated in a small subset

**Results**: Starting MSE 0.0125 → **0.01229** (or-opt contribution)

### Attempt 15: FP16 + Or-opt ILS — SOLVED (solve_ils_fast.py v2)
Added FP16 and or-opt to the GPU-batched ILS solver:
- **FP16 (half-precision)**: All subsample evaluation uses FP16 weights/activations
  for ~2-4x throughput on RTX 3080 tensor cores. Final full-data verification
  remains FP32 for accuracy.
- **Or-opt in local search**: `local_search_fast` now does 2-way first-improvement →
  or-opt fallback. Or-opt relocates a block from position i to position j,
  shifting the intervening segment. ~94 candidates per position, GPU-batched.
- **Larger batch chunks**: FP16 halves memory → chunk_size doubled (256→512)

**Results (120/500 restarts, ~900s on RTX 3080):**

| Restart | MSE | Intensity |
|---------|-----|-----------|
| 76 | 0.01154 | 1 |
| 81 | 0.01058 | 1 |
| 83 | 0.01027 | 2 |
| 87 | 0.01026 | 2 |
| 94 | 0.01013 | 1 |
| 98 | 0.00789 | 5 |
| 100 | 0.00725 | 2 |
| 102 | 0.00646 | 2 |
| 106 | 0.00640 | 1 |
| 114 | 0.00603 | 1 |
| 117 | 0.00216 | 1 |
| 118 | 0.00121 | 3 |
| 119 | 0.00085 | 3 |
| **120** | **0.0000000000** | **2** |

**PUZZLE SOLVED.** MSE = 0.0 — the exact correct permutation was found.

The critical cascade happened at restarts 117-120: four consecutive improvements
in rapid succession (0.006 → 0.002 → 0.001 → 0.0008 → 0.0), suggesting the
solution was in a nearby basin that or-opt + low-intensity perturbations could reach.

### Current Best
- **MSE: 0.0000000000 (EXACT SOLUTION)**
- Permutation saved in solution.txt

## Key Insights

1. **Pairing is the bottleneck** — 48! possible pairings, local search stuck in basin
2. **First-order approximation fails** — residuals are too large for blocks to commute;
   projected contribution analysis gives MSE 0.7+ at best (vs 0.095 from local search)
3. **Ordering matters significantly** — greedy reordering alone improved MSE by ~3%
4. **The hint** ("Pths→Eqx, Pytrees, Trio") suggests: convert to JAX/Equinox for
   `vmap`/`jit` vectorized evaluation, use Trio for concurrent search. The key is
   **evaluation speed enabling wider search**, not a fundamentally different algorithm.
5. **ILS is the winning strategy** — random perturbations + local search escapes local
   minima that pure local search cannot.
6. **Subsampling is dangerous** — 1000-sample subsample diverges significantly from full
   data (0.013 vs 0.023). Must use full 10K samples for reliable evaluation. The
   subsample finds "illusory" improvements that don't transfer.
7. **Cross-swaps don't help** — 103K 3-way cross-swap moves found nothing (3 passes).
   The missing moves are NOT of this type.
8. **3-way rotations DO help** — found improvements that 2-way swaps couldn't, unlocking
   cascading 2-way improvements afterward.
9. **Or-opt (block relocation) helps** — found improvements that 2-way swaps couldn't.
   Relocating a block shifts a whole segment, equivalent to multiple swaps in one move.
10. **Error is distributed** — leave-one-out shows all 48 blocks contribute positively.
    No small subset of "wrong" blocks to fix. Targeted exhaustive search on worst 6
    blocks found nothing. Improvements come from global rearrangements, not local fixes.
11. **Speed unlocks solutions** — FP16 + or-opt solved the puzzle in 120 restarts (~900s)
    where FP32 2-way-only took 250 restarts (~3400s) to reach MSE 0.0089. The combination
    of faster evaluation (more restarts/time) and richer neighborhoods (or-opt) was decisive.
