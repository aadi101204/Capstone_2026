# WGAN-GP → WGAN-SN: Encryption Model Modifications

**Commit Reference:** `85e351f` (WGAN-GP baseline) → `30b29b3` (HEAD — "WGAN-GP-SN implemented")  
**Files Modified:** `main.py`, `src/models.py`, `src/utils.py`

---

## 1. Background & Motivation

The original encryption model was built on **WGAN-GP** (Wasserstein GAN with Gradient Penalty), a widely-used stabilisation technique for GAN training. While theoretically sound, WGAN-GP imposed a critical practical constraint on this codebase: it requires **double backpropagation** to compute the gradient of the critic's gradient w.r.t. its inputs.

PyTorch's CuDNN optimised LSTM kernels **do not support double backpropagation**. This forced `torch.backends.cudnn.enabled = False` when training on a GPU, eliminating virtually all GPU acceleration and causing catastrophic performance hangs during training — particularly painful given that the Critic itself is an LSTM.

The migration to **WGAN-SN** (WGAN with Spectral Normalisation) resolved this bottleneck by replacing the penalty-based Lipschitz constraint with an architectural one (spectral norm on weight matrices), removing the need for double backpropagation entirely and re-enabling CuDNN.

---

## 2. Summary of Changes

| Aspect | WGAN-GP (old) | WGAN-SN (new) |
|---|---|---|
| Lipschitz constraint | Gradient Penalty (explicit, computed) | Spectral Normalisation (embedded in weights) |
| Double backprop required | Yes | No |
| CuDNN | **Disabled** (→ slow/hangs on GPU) | **Enabled** (→ full GPU speed) |
| `gradient_penalty()` utility | Present in `src/utils.py` | **Removed** |
| `--lambda_gp` argument | Present in `main.py` | **Removed** |
| Critic FC layers | Plain `nn.Linear` | `nn.utils.spectral_norm(nn.Linear)` |
| Critic loss | `-(E[real] - E[fake]) + λ·GP` | `-(E[real] - E[fake])` |
| Generator branch layers | 2-layer LSTM + Tanh output + LeakyReLU seed | 3-layer LSTM + SIREN-style Sine seed + ELU + cumsum |
| Generator coupling | `nn.Conv1d` cross-node coupling + `BatchNorm1d` | **Removed** (direct stacking only) |
| Output scaling | `torch.sigmoid(BatchNorm(Conv1d(stacked)))` | `torch.sigmoid(stacked)` |

---

## 3. Detailed File-by-File Changes

### 3.1 `src/utils.py` — Removal of `gradient_penalty`

**Before (WGAN-GP):** `utils.py` contained a `gradient_penalty` function that:
1. Sampled a random interpolation `ε ∈ U[0,1]` between real and fake sequences.
2. Created `interpolated = ε·real + (1-ε)·fake` with `requires_grad=True`.
3. Ran the Critic on this interpolated point.
4. Computed `∂Critic/∂interpolated` via `torch.autograd.grad(create_graph=True, retain_graph=True)`.
5. Penalised deviation of the gradient norm from 1: `GP = E[(‖∇‖₂ − 1)²]`.

```python
# OLD: src/utils.py
def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    prob_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,      # ← requires double backprop
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
```

**After (WGAN-SN):** The entire `gradient_penalty` function was **deleted**. The `utils.py` now only contains `set_seed`.

---

### 3.2 `src/models.py` — Spectral Norm in Critic + Generator Redesign

#### 3.2.1 `LSTMCritic` — Spectral Normalisation on FC Layers

The Critic architecture is structurally identical (dual-stream: raw + differential LSTM), but the fully-connected head now wraps every `nn.Linear` layer with `nn.utils.spectral_norm`.

```python
# OLD: Plain Linear layers — no Lipschitz control in architecture
self.fc = nn.Sequential(
    nn.Linear(hidden_dim * 4, hidden_dim),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim, 1)
)
```

```python
# NEW: Spectral Norm enforces ‖W‖₂ ≤ 1, bounding the Critic's Lipschitz constant
self.fc = nn.Sequential(
    nn.utils.spectral_norm(nn.Linear(hidden_dim * 4, hidden_dim)),
    nn.LeakyReLU(0.2),
    nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
)
```

**What spectral norm does:** During each forward pass, PyTorch normalises the weight matrix `W` by its largest singular value `σ(W)`, ensuring `‖W‖₂ = 1`. This bounds the Lipschitz constant of the entire Critic network, which is the formal requirement for the Wasserstein distance estimation to be valid — achieved here without needing to explicitly compute any gradients of the Critic.

#### 3.2.2 `LSTMGeneratorBranch` — SIREN Seeding, 3-Layer LSTM, ELU + cumsum

The generator branch underwent a significant architectural rethink focused on producing more chaotically expressive sequences.

| Sub-component | Old (WGAN-GP) | New (WGAN-SN) |
|---|---|---|
| LSTM depth | 2 layers | 3 layers |
| Initial state activation | `LeakyReLU(0.2)` | `torch.sin(30 · x)` (SIREN) |
| Output block | `nn.Linear → Tanh` | `nn.Linear → ELU → nn.Linear` (delta) |
| Integration | None (direct output) | `torch.cumsum(deltas * 0.1, dim=1)` |

```python
# OLD: 2-layer LSTM, LeakyReLU seeded h0/c0, Tanh output
self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
self.fc_out = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
self.leaky_relu = nn.LeakyReLU(0.2)

def forward(self, z_slice):
    h0 = self.leaky_relu(self.init_h(z_slice)).unsqueeze(0).repeat(2, 1, 1).contiguous()
    c0 = self.leaky_relu(self.init_c(z_slice)).unsqueeze(0).repeat(2, 1, 1).contiguous()
    lstm_out, _ = self.lstm(dummy_input, (h0, c0))
    out = self.fc_out(lstm_out)
    return out.squeeze(-1)
```

```python
# NEW: 3-layer LSTM, SIREN sine-seeded h0/c0, ELU delta + cumsum integration
self.lstm = nn.LSTM(1, hidden_dim, num_layers=3, batch_first=True, dropout=0.1)
self.fc_delta = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ELU(),
    nn.Linear(hidden_dim // 2, 1)
)

def forward(self, z_slice):
    # SIREN: multiply by 30 to capture high-frequency chaotic features
    h0 = torch.sin(30 * self.init_h(z_slice)).unsqueeze(0).repeat(3, 1, 1).contiguous()
    c0 = torch.sin(30 * self.init_c(z_slice)).unsqueeze(0).repeat(3, 1, 1).contiguous()
    lstm_out, _ = self.lstm(dummy_input, (h0, c0))
    deltas = self.fc_delta(lstm_out).squeeze(-1) * 0.1
    seq = torch.cumsum(deltas, dim=1)   # Integrate increments
    return seq
```

**Key design decisions:**
- **SIREN seeding (sin(30·x)):** Sinusoidal Representation Networks (SIREN) are known to encode high-frequency signals; using `sin(30·x)` as the initial LSTM state maps noise into a rich initial dynamical regime, encouraging chaotic sensitivity to initial conditions.
- **ELU over Tanh/LeakyReLU:** ELU's exponential tails preserve negative information without saturating, better suited for representing trajectories with sharp fluctuations.
- **Cumulative sum integration:** Rather than directly outputting sequence values, the network predicts incremental *deltas* (scaled by 0.1 for stability), which are integration-accumulated. This mirrors how real chaotic trajectories evolve — as accumulated sensitive increments — rather than outputting independent values at each step.
- **3-layer LSTM:** Extra recurrent depth gives the generator more capacity for internal state dynamics needed to produce long-range chaotic correlations.

#### 3.2.3 `LSTMGenerator` — Removal of Cross-Node Coupling

```python
# OLD: Cross-node coupling via Conv1d + BatchNorm1d
self.coupling = nn.Conv1d(4, 4, kernel_size=3, padding=1, padding_mode='circular')
self.bn = nn.BatchNorm1d(4)

def forward(self, z):
    stacked = torch.stack([out1, out2, out3, out4], dim=1)
    coupled = self.coupling(stacked)
    normalized = self.bn(coupled)
    return torch.sigmoid(normalized)
```

```python
# NEW: Direct stack, no coupling or normalization
def forward(self, z):
    stacked = torch.stack([out1, out2, out3, out4], dim=1)
    return torch.sigmoid(stacked)  # Pure per-branch output
```

**Rationale:** The `Conv1d` coupling layer mixed dynamics between the 4 chaotic nodes. While this could add complexity, it also introduced learned cross-node correlations that *reduce* statistical independence between nodes — counterproductive for an encryption key generation use-case where maximum per-channel entropy is desired. `BatchNorm1d` further homogenised outputs across the batch, smoothing out chaotic fluctuations. Both were removed to preserve the raw chaos from each independent branch.

---

### 3.3 `main.py` — Training Loop & CuDNN Changes

#### Argument Changes

```python
# OLD: Had --lambda_gp parameter
parser.add_argument("--lambda_gp", type=int, default=10, help="Gradient penalty coefficient")

# NEW: --lambda_gp removed entirely
```

#### CuDNN Behaviour

```python
# OLD: CuDNN disabled — required for double backprop in WGAN-GP + LSTM
if device.type == 'cuda':
    torch.backends.cudnn.enabled = False
    print("CuDNN disabled to support Gradient Penalty with LSTMs.")
```

```python
# NEW: CuDNN enabled — SN removes the need for double backprop
if device.type == 'cuda':
    torch.backends.cudnn.enabled = True
    print("Using WGAN-SN (CuDNN Enabled). Gradient Penalty removed to fix performance hangs.")
```

#### Critic Loss Simplification

```python
# OLD: Wasserstein loss + weighted Gradient Penalty
gp = gradient_penalty(C, real_seq, fake_seq.detach(), device)
loss_C = -(torch.mean(outputs_real) - torch.mean(outputs_fake)) + args.lambda_gp * gp
```

```python
# NEW: Pure Wasserstein loss — Lipschitz already enforced by SN in model weights
# Standard WGAN Loss (Stability provided by Spectral Norm in models.py)
loss_C = -(torch.mean(outputs_real) - torch.mean(outputs_fake))
```

#### Import Changes

```python
# OLD
from src.utils import set_seed, gradient_penalty

# NEW — gradient_penalty import removed
from src.utils import set_seed
```

---

## 4. Impact on the Encryption Pipeline

The WGAN trains the `LSTMGenerator`, whose saved weights (`generator.pth`) are later used by `encryption.py` and `inference.py` to produce deterministic chaotic sequences (given a fixed noise seed). These sequences serve as the pixel-level XOR keys for encrypting image frames.

- **Training speed:** Eliminating the Gradient Penalty computation and re-enabling CuDNN provides a significant GPU speedup. The GP computation was proportional to the batch size and sequence length — for 192-step sequences, this was non-trivial.
- **Key quality:** The SIREN seeding + cumsum integration in the generator branches are expected to produce sequences with higher sensitivity to initial noise conditions, improving the cryptographic unpredictability of the generated keys.
- **Statistical independence:** Removing the cross-node Conv1d coupling ensures the 4 generated chaotic node sequences remain statistically independent, which is a desirable property for multi-channel encryption keys.

---

## 5. Files Changed (Quick Reference)

| File | Change Type | Description |
|---|---|---|
| `src/utils.py` | Modified | `gradient_penalty()` function removed |
| `src/models.py` | Modified | Spectral norm on Critic FC; SIREN+ELU+cumsum in Generator branch; Conv1d+BN removed from Generator |
| `main.py` | Modified | `--lambda_gp` arg removed; CuDNN enabled; GP call + import removed; Critic loss simplified |

---

*Document generated: 2026-02-21 | Commits compared: `85e351f` (WGAN-GP) → `30b29b3` (WGAN-SN)*
