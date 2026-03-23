import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryMLP(nn.Module):
    def __init__(self, d_model, dim_inner):
        super().__init__()
        self.W1 = nn.Linear(d_model, dim_inner, bias=False)
        self.W2 = nn.Linear(dim_inner, d_model, bias=False)
        
    def forward(self, x):
        return x + self.W2(F.silu(self.W1(x)))

def dgd_update(W, k, v_hat, lr, chunk_size):
    W_orig_dtype = W.dtype
    W_f32 = W.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_hat_f32 = v_hat.to(torch.float32)

    if k_f32.dim() == 3:
        # Vectorized: process all chunk tokens simultaneously
        # k_f32:     [batch, chunk, d_in]
        # v_hat_f32: [batch, chunk, d_out]
        # W_f32:     [batch, d_out, d_in]

        lr_prime = lr / (1.0 + lr * (k_f32 * k_f32).sum(-1, keepdim=True))
        # [batch, chunk, 1]

        W_k = torch.bmm(k_f32, W_f32.transpose(1, 2))
        # [batch, chunk, d_out]

        residual = W_k - v_hat_f32
        # [batch, chunk, d_out]

        # Outer products summed over chunk dimension
        decay = torch.einsum('bci,bcj->bij', lr_prime * W_k, k_f32)
        grad  = torch.einsum('bci,bcj->bij', lr_prime * residual, k_f32)
        W_f32 = W_f32 - decay - grad
    else:
        lr_prime = lr / (1.0 + lr * (k_f32 * k_f32).sum(-1, keepdim=True))
        W_k = (W_f32 @ k_f32.unsqueeze(-1)).squeeze(-1)
        residual = W_k - v_hat_f32

        term1 = W_k.unsqueeze(-1) * k_f32.unsqueeze(-2)
        term2 = residual.unsqueeze(-1) * k_f32.unsqueeze(-2)
        W_f32 = W_f32 - lr_prime.unsqueeze(-1) * (term1 + term2)

    return W_f32.to(W_orig_dtype)

if __name__ == '__main__':
    d_model = 64
    batch = 2
    chunk_size = 8
    lr = 1e-3
    W_init = torch.randn(batch, d_model, d_model, dtype=torch.bfloat16)
    k = F.normalize(torch.randn(batch, chunk_size, d_model, dtype=torch.bfloat16), p=2, dim=-1)
    v_hat = torch.randn(batch, chunk_size, d_model, dtype=torch.bfloat16)
    W_newdt = dgd_update(W_init, k, v_hat, lr, chunk_size)
    assert not torch.isnan(W_newdt).any()
    print("PASS")
