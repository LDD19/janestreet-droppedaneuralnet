"""
Try to find correct pairs by analyzing weight matrix properties.
Hypothesis: correctly paired layers have specific mathematical relationships.
"""
import torch
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pieces
pieces = {}
inp_layers = []
out_layers = []
last_layer_idx = None

for i in range(97):
    piece = torch.load(f'pieces/piece_{i}.pth', weights_only=True, map_location='cpu')
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

pieces_gpu = {i: {k: v.to(device) for k, v in p.items()} for i, p in pieces.items()}
inp_weights = {idx: (pieces_gpu[idx]['weight'], pieces_gpu[idx]['bias']) for idx in inp_layers}
out_weights = {idx: (pieces_gpu[idx]['weight'], pieces_gpu[idx]['bias']) for idx in out_layers}
ll_w, ll_b = pieces_gpu[last_layer_idx]['weight'], pieces_gpu[last_layer_idx]['bias']

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

# Compute various matching scores
print("--- Computing matching scores ---")

# Score 1: Frobenius norm of product - identity
print("Score 1: ||out @ inp - I||")
score1 = np.zeros((48, 48))
I = torch.eye(48)
for i, inp_idx in enumerate(inp_layers):
    inp_w = pieces[inp_idx]['weight']  # (96, 48)
    for j, out_idx in enumerate(out_layers):
        out_w = pieces[out_idx]['weight']  # (48, 96)
        product = out_w @ inp_w  # (48, 48)
        score1[i, j] = (product - I).norm().item()

# Score 2: Trace of product
print("Score 2: trace(out @ inp)")
score2 = np.zeros((48, 48))
for i, inp_idx in enumerate(inp_layers):
    inp_w = pieces[inp_idx]['weight']
    for j, out_idx in enumerate(out_layers):
        out_w = pieces[out_idx]['weight']
        product = out_w @ inp_w
        score2[i, j] = -torch.trace(product).item()  # Negative because we minimize

# Score 3: Correlation of weight statistics
print("Score 3: weight stat correlation")
score3 = np.zeros((48, 48))
inp_stats = {}
out_stats = {}
for idx in inp_layers:
    w = pieces[idx]['weight'].flatten()
    inp_stats[idx] = (w.mean().item(), w.std().item(), w.min().item(), w.max().item())
for idx in out_layers:
    w = pieces[idx]['weight'].flatten()
    out_stats[idx] = (w.mean().item(), w.std().item(), w.min().item(), w.max().item())

for i, inp_idx in enumerate(inp_layers):
    for j, out_idx in enumerate(out_layers):
        # Correlation between stats
        s1 = np.array(inp_stats[inp_idx])
        s2 = np.array(out_stats[out_idx])
        score3[i, j] = np.sum((s1 - s2)**2)

# Score 4: Spectral properties
print("Score 4: spectral norm ratio")
score4 = np.zeros((48, 48))
for i, inp_idx in enumerate(inp_layers):
    inp_w = pieces[inp_idx]['weight']
    inp_norm = torch.linalg.norm(inp_w, ord=2).item()
    for j, out_idx in enumerate(out_layers):
        out_w = pieces[out_idx]['weight']
        out_norm = torch.linalg.norm(out_w, ord=2).item()
        score4[i, j] = abs(inp_norm - out_norm)

# Try each score with Hungarian
print("\n--- Testing different matching scores ---")

for name, score in [("Identity diff", score1), ("Neg trace", score2),
                     ("Stat diff", score3), ("Spectral diff", score4)]:
    row_ind, col_ind = linear_sum_assignment(score)
    pairs = [(inp_layers[i], out_layers[j]) for i, j in zip(row_ind, col_ind)]
    mse = evaluate(pairs)
    print(f"{name}: MSE = {mse:.6f}")

# Try combined scores
print("\n--- Trying combined scores ---")
for alpha in [0.1, 0.5, 1.0, 2.0]:
    combined = score1 + alpha * score3
    row_ind, col_ind = linear_sum_assignment(combined)
    pairs = [(inp_layers[i], out_layers[j]) for i, j in zip(row_ind, col_ind)]
    mse = evaluate(pairs)
    print(f"Identity + {alpha}*Stat: MSE = {mse:.6f}")

# Current best for comparison
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
current_pairs = [(perm[i], perm[i+1]) for i in range(0, 96, 2)]
current_mse = evaluate(current_pairs)
print(f"\nCurrent best: MSE = {current_mse:.6f}")
