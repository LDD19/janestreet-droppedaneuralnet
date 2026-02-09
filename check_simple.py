"""
Check simple pairing patterns.
"""
import torch
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

print("Sorted inp:", sorted(inp_layers))
print("Sorted out:", sorted(out_layers))

# Pattern 1: pair by sorted index (inp[0] with out[0], etc)
sorted_inp = sorted(inp_layers)
sorted_out = sorted(out_layers)
pairs1 = list(zip(sorted_inp, sorted_out))
mse1 = evaluate(pairs1)
print(f"\nPattern 1 (sorted index match): MSE = {mse1:.6f}")

# Pattern 2: pair by reverse sorted
pairs2 = list(zip(sorted_inp, sorted_out[::-1]))
mse2 = evaluate(pairs2)
print(f"Pattern 2 (reverse match): MSE = {mse2:.6f}")

# Pattern 3: pair by original order in file
pairs3 = list(zip(inp_layers, out_layers))
mse3 = evaluate(pairs3)
print(f"Pattern 3 (file order): MSE = {mse3:.6f}")

# Pattern 4: consecutive pairs (0,1), (2,3), etc in original numbering
# Wait, that doesn't work because 0 is inp and 1 is also inp...

# Let's look for patterns in numbering
print("\n--- Analyzing index patterns ---")
all_pieces = list(range(97))

# For each potential Block, check if inp and out indices are close
print("\nConsecutive index pairs (if they exist):")
for i in range(96):
    if i in inp_layers and i+1 in out_layers:
        print(f"  ({i}, {i+1}) - MATCH")
    elif i in out_layers and i+1 in inp_layers:
        print(f"  ({i+1}, {i}) - MATCH reversed")

# Look for modular pattern
print("\nChecking modular patterns:")
for mod in [2, 3, 4, 5, 6]:
    inp_mod = [i % mod for i in sorted_inp]
    out_mod = [o % mod for o in sorted_out]
    print(f"  mod {mod}: inp={set(inp_mod)}, out={set(out_mod)}")

# Check if any simple arithmetic relationship exists
print("\nPotential pair relationships:")
# For each inp, is there a consistent offset to its out?
for offset in range(-10, 11):
    count = 0
    pairs = []
    for inp in sorted_inp:
        out = inp + offset
        if out in out_layers:
            count += 1
            pairs.append((inp, out))
    if count > 5:
        print(f"  offset {offset}: {count} matches")
        if count == 48:  # All match!
            mse = evaluate(pairs)
            print(f"    MSE = {mse:.6f}")

# Current best
with open("solution.txt", "r") as f:
    perm = list(map(int, f.read().strip().split(",")))
current_pairs = [(perm[i], perm[i+1]) for i in range(0, 96, 2)]
current_mse = evaluate(current_pairs)
print(f"\nCurrent best: MSE = {current_mse:.6f}")
