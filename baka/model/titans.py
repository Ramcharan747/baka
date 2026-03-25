import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from baka.model.config import BAKAConfig
    from baka.model.dgd import MemoryMLP, dgd_update
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baka.model.config import BAKAConfig
    from baka.model.dgd import MemoryMLP, dgd_update


def apply_rope(q, seq_len, chunk_offset=0):
    batch, chunk, dim = q.shape
    d_head = 128
    if dim % d_head != 0:
        d_head = dim
    n_heads = dim // d_head

    q = q.view(batch, chunk, n_heads, d_head)
    positions = torch.arange(chunk_offset, chunk_offset + chunk, dtype=torch.float32, device=q.device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2, dtype=torch.float32, device=q.device) / d_head))

    sincos = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(sincos).unsqueeze(0).unsqueeze(2)
    cos = torch.cos(sincos).unsqueeze(0).unsqueeze(2)

    q1, q2 = q[..., ::2], q[..., 1::2]
    q_rot = torch.empty_like(q)
    q_rot[..., ::2] = q1 * cos - q2 * sin
    q_rot[..., 1::2] = q1 * sin + q2 * cos
    return q_rot.view(batch, chunk, dim)


class DynamicMemoryModule(nn.Module):
    def __init__(self, d_model, d_memory):
        super().__init__()
        self.mlp = MemoryMLP(d_model, d_memory)
        self.W_base = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.orthogonal_(self.W_base, gain=0.01)
        self.W_current = None  # plain Python attribute, not a buffer

    def reset_state(self, batch_size):
        self.W_current = self.W_base.unsqueeze(0).repeat(batch_size, 1, 1).detach().clone()

    def update(self, x, v_hat_external, lr, chunk_size):
        """Update W_current via DGD using externally-provided v_hat."""
        k = F.normalize(x, p=2, dim=-1)
        new_W = dgd_update(self.W_current, k, v_hat_external, lr, chunk_size)
        self.W_current = new_W.detach()  # inner loop: must not flow into outer autograd

    def forward(self, q):
        return torch.bmm(q, self.W_current.transpose(1, 2))


class SelfModifyingTitans(nn.Module):
    def __init__(self, config: BAKAConfig):
        super().__init__()
        self.d_model = config.d_model
        self.chunk_size = config.titans_chunk_size
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        self.M_k = DynamicMemoryModule(self.d_model, config.d_memory)
        self.M_v = DynamicMemoryModule(self.d_model, config.d_memory)
        self.M_eta = DynamicMemoryModule(self.d_model, config.d_memory)
        self.M_alpha = DynamicMemoryModule(self.d_model, config.d_memory)
        self.M_mem = DynamicMemoryModule(self.d_model, config.d_memory)

        self.W_q = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        # self.W_q.weight.requires_grad = False  # EXPERIMENT: unfreezing W_q to test if trainable queries help

    def reset_state(self, batch_size: int):
        self.M_k.reset_state(batch_size)
        self.M_v.reset_state(batch_size)
        self.M_eta.reset_state(batch_size)
        self.M_alpha.reset_state(batch_size)
        self.M_mem.reset_state(batch_size)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        if self.M_k.W_current is None or self.M_k.W_current.size(0) != batch:
            self.reset_state(batch)

        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[:, i:i + self.chunk_size, :]
            c_len = chunk.shape[1]

            # --- Output: M_mem applied to RoPE'd queries ---
            q = self.W_q(chunk)
            q = apply_rope(q, seq_len, chunk_offset=i)
            y_t = self.M_mem(q)
            outputs.append(y_t)

            # --- Self-referential update (paper eq 83-88) ---
            lr = 1e-3

            # Step 1: compute v_t from M_v's MLP applied to chunk
            v_t = self.M_v.mlp(chunk)  # [batch, chunk_len, d_model]

            # Step 2: each memory generates its OWN v_hat from v_t
            v_hat_k = self.M_k.mlp(v_t)
            v_hat_v = self.M_v.mlp(v_t)
            v_hat_eta = self.M_eta.mlp(v_t)
            v_hat_alpha = self.M_alpha.mlp(v_t)
            v_hat_mem = self.M_mem.mlp(v_t)

            # Step 3 & 4: update each memory with its own v_hat
            self.M_k.update(chunk, v_hat_k, lr, c_len)
            self.M_v.update(chunk, v_hat_v, lr, c_len)
            self.M_eta.update(chunk, v_hat_eta, lr, c_len)
            self.M_alpha.update(chunk, v_hat_alpha, lr, c_len)
            self.M_mem.update(chunk, v_hat_mem, lr, c_len)

        return torch.cat(outputs, dim=1)


if __name__ == '__main__':
    config = BAKAConfig()
    config.d_model = 64
    config.n_heads = 4
    config.d_head = 16
    config.d_memory = 32
    config.titans_chunk_size = 64

    titans = SelfModifyingTitans(config).to(torch.bfloat16)
    x = torch.randn(2, 128, 64, dtype=torch.bfloat16)
    titans.reset_state(2)
    out = titans(x)
    assert out.shape == (2, 128, 64), f"Shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"

    # Verify memory PERSISTS across forward calls (no reset between them)
    state_before = titans.M_mem.W_current.clone()
    x2 = torch.randn(2, 128, 64, dtype=torch.bfloat16)
    out2 = titans(x2)  # second forward WITHOUT reset
    state_after = titans.M_mem.W_current
    assert not torch.allclose(state_before, state_after), "M_mem.W_current did not change between forward calls"
    print("PASS")
