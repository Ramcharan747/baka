# BAKA Architecture Specification

## Model Configuration (1.3B)

```
d_model:              2048
n_layers:             24
n_heads:              16
d_head:               128  (d_model / n_heads)
d_ffn:                8192 (4 * d_model)
d_memory:             256  (inner memory MLP dimension)
cms_levels:           4
cms_chunk_sizes:      [16, 256, 4096, 65536]
titans_chunk_size:    64
context_length:       8192
vocab_size:           32000
normalization:        RMSNorm
activation:           SiLU
position_encoding:    RoPE
weight_tying:         True (embedding + LM head share weights)
```

---

## Component 1: Delta Gradient Descent (DGD)

Source: Section 4.5, Equation 57 of the paper.

Standard gradient descent treats each sample independently:
```
W = W - lr * gradient
```

DGD incorporates the current weight state into the update:
```
lr' = lr / (1 + lr * ||x||²)
W_new = W_old * (I - lr' * x * xᵀ) - lr' * ∇L(W_old; x)
```

Expanded for L2 loss where target is v:
```
W_new = W_old * (I - lr' * k * kᵀ) - lr' * (W_old*k - v) * kᵀ
```

The term `(I - lr' * k * kᵀ)` is input-dependent weight decay.
When the current key k is relevant to stored memories in W,
those memories are partially decayed. This is structured forgetting.

**Implementation requirement:**
- Inputs k must be L2-normalized before DGD: k = k / ||k||₂
- Use float32 for DGD computation, cast result back to bfloat16
- This update runs INSIDE the forward pass, not in the outer optimizer

---

## Component 2: Self-Modifying Titans

Source: Section 8.1, Equations 83-88 of the paper.

Standard linear attention uses frozen projections:
```
k_t = x_t * W_k   (W_k frozen after training)
v_t = x_t * W_v   (W_v frozen after training)
```

Titans makes every projection a living memory module:
```
k_t   = M_k_{t-1}(x_t)       # M_k is a 2-layer MLP that updates
v_t   = M_v_{t-1}(x_t)       # M_v is a 2-layer MLP that updates
eta_t = M_eta_{t-1}(x_t)     # per-token learning rate
alpha_t = M_alpha_{t-1}(x_t) # per-token retention gate
q_t   = x_t * W_q            # ONLY q uses a frozen projection
```

Self-referential value generation (each memory generates its own targets):
```
v̂_k     = M_k_{t-1}(v_t)
v̂_v     = M_v_{t-1}(v_t)
v̂_eta   = M_eta_{t-1}(v_t)
v̂_alpha = M_alpha_{t-1}(v_t)
v̂_mem   = M_mem_{t-1}(v_t)
```

DGD update for each memory module (□ ∈ {k, v, eta, alpha, mem}):
```
M_□_t = M_□_{t-1} * (alpha_t * I - eta_t * k_t * k_tᵀ)
        - eta_t * (M_□_{t-1} * k_t - v̂_□) * k_tᵀ
```

Main sequence output:
```
y_t = M_mem_t(q_t)
```

**Memory module architecture (each M_□):**
```python
class MemoryMLP(nn.Module):
    def forward(self, x):
        return x + self.W2(F.silu(self.W1(x)))
    # W1: d_model -> d_memory (256)
    # W2: d_memory -> d_model
```

**Chunk-wise training (required for parallelization):**
```
Process sequence in chunks of size C=64
Within chunk: all memory states frozen (use states from end of prev chunk)
End of chunk: update ALL memory states using DGD
This allows full parallel computation within each chunk
```

**Initial states:**
All M_□_0 are learnable parameters (meta-learned by outer AdamW optimizer).
Reset to M_□_0 at the start of each new sequence.

---

## Component 3: Continuum Memory System (CMS)

Source: Section 7.1, Equations 70-71 of the paper.

Replaces the standard FFN block. Four MLPs at different update frequencies:

```
CMS(x_t) = MLP4(MLP3(MLP2(MLP1(x_t))))

MLP1: chunk_size=16    — fast, short-term memory
MLP2: chunk_size=256   — medium memory
MLP3: chunk_size=4096  — slow, long-term memory
MLP4: chunk_size=65536 — near-persistent, like pre-training weights
```

Each MLP_l is a standard 2-layer FFN:
```
MLP_l(x) = W2_l * SiLU(W1_l * x)
dimensions: d_model -> d_ffn -> d_model  (2048 -> 8192 -> 2048)
```

Update rule for MLP_l at step t:
```
if t % chunk_size_l == 0:
    accumulated_grad = sum of ∂L/∂θ_l over last chunk_size_l steps
    θ_l = θ_l - lr_l * accumulated_grad
else:
    θ_l unchanged
```

Learning rates per level (slower levels use smaller lr):
```
lr_1 = 1e-3  (fast, adapts quickly)
lr_2 = 1e-4
lr_3 = 1e-5
lr_4 = 1e-6  (slow, near-frozen)
```

**Why this prevents catastrophic forgetting:**
When MLP1 updates and forgets something, MLP2/3/4 may still retain it.
The slower layers act as a recovery mechanism across all timescales.

---

## Component 4: Full BAKA Block

One complete HOPE block:

```
Input: x  (shape: batch, seq_len, d_model)

x_norm = RMSNorm(x)
o = Titans(x_norm)        # Self-modifying sequence mixer
x = x + o                 # Residual

x_norm2 = RMSNorm(x)
y = CMS(x_norm2)           # Continuum memory feature transform
x = x + y                 # Residual

Output: x  (shape: batch, seq_len, d_model)
```

---

## Component 5: Full BAKA Model

```
tokens (batch, seq_len)
  -> Embedding (vocab_size=32000, d_model=2048)
  -> RoPE position encoding (applied to q in each Titans block)
  -> 24 x BAKABlock
  -> RMSNorm
  -> LM Head (d_model=2048, vocab_size=32000)
  -> logits (batch, seq_len, vocab_size)

Weight tying: Embedding.weight == LMHead.weight
```

---

## Two Gradient Flows — Critical

BAKA has two completely separate gradient flows that must NEVER be mixed:

**Outer loop (standard PyTorch autograd):**
- Optimizer: AdamW
- Loss: cross-entropy on next token prediction
- Updates: embedding, W_q, LM head, all initial memory states M_□_0
- Standard .backward() call

**Inner loop (DGD, runs inside forward pass):**
- Updates: M_k, M_v, M_eta, M_alpha, M_mem (current states, not initials)
- Updates: MLP1, MLP2, MLP3, MLP4 current states in CMS
- These are NOT tracked by outer autograd
- Use .detach() on all inner memory states
- Compute DGD updates manually using the equations above

Mixing these two will either crash training or silently produce wrong gradients.

---

## File Structure

```
baka/
├── model/
│   ├── config.py          ModelConfig dataclass with all hyperparameters
│   ├── dgd.py             dgd_update() function + DGDLayer class
│   ├── cms.py             CMSBlock class
│   ├── titans.py          SelfModifyingTitans class
│   ├── baka_block.py      BAKABlock class (Titans + CMS)
│   └── baka.py            BAKA model class (full model)
├── training/
│   ├── pretrain.py        Phase 1 training loop
│   ├── sft.py             Phase 2 supervised fine-tuning
│   └── grpo.py            Phase 3 GRPO reinforcement learning
├── data/
│   ├── download.py        Download FineWeb-Edu + datasets
│   └── pipeline.py        Tokenizer, chunking, DataLoader
└── eval/
    └── benchmark.py       GSM8K, ARC, HellaSwag, HumanEval evaluation
```

---

## Evaluation Targets

Run these benchmarks. Compare against published Llama-2 7B numbers.

```
GSM8K (math reasoning):       Llama-2 7B = 14.6%   → BAKA target: 20%+
ARC-Challenge (science):      Llama-2 7B = 53.7%   → BAKA target: 55%+
HellaSwag (commonsense):      Llama-2 7B = 77.2%   → BAKA target: 70%+
WinoGrande (pronoun):         Llama-2 7B = 69.2%   → BAKA target: 68%+
HumanEval (code):             Llama-2 7B = 12.8%   → BAKA target: 15%+
```

Beating Llama-2 7B on GSM8K at 1.3B parameters validates the core claim.

---

## Key Implementation Notes

1. All inputs to DGD must be L2-normalized
2. DGD runs in float32, results cast to bfloat16
3. Inner memory states use .detach() — not tracked by outer autograd
4. Initial memory states M_□_0 ARE tracked by outer autograd (they are nn.Parameters)
5. Chunk size for Titans = 64 tokens
6. CMS chunk sizes = [16, 256, 4096, 65536]
7. Gradient checkpointing required to fit 1.3B on T4 15GB
8. Save checkpoint to HuggingFace Hub every 500 steps
9. Tokenizer: use Qwen/Qwen3-0.6B tokenizer (vocab_size=32000, already proven)
10. RoPE applied only to the frozen q projection in Titans
