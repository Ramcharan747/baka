import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

try:
    from baka.model.config import BAKAConfig
    from baka.model.dgd import dgd_update, MemoryMLP
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baka.model.config import BAKAConfig
    from baka.model.dgd import dgd_update, MemoryMLP

class SelfModifyingTitans(nn.Module):
    def __init__(self, config: BAKAConfig):
        super().__init__()
        self.config = config
        self.chunk_size = config.titans_chunk_size
        self.d_model = config.d_model
        
        # Frozen query projection
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Living memory modules (MLPs)
        self.M_k = MemoryMLP(config.d_model, config.d_memory)
        self.M_v = MemoryMLP(config.d_model, config.d_memory)
        self.M_mem = MemoryMLP(config.d_model, config.d_memory)
        
        # Base memory state meta-learned by AdamW
        self.W_base = nn.Parameter(torch.empty(1, config.d_model, config.d_model))
        nn.init.orthogonal_(self.W_base)
        
        self.W_current = None
        
    def reset_state(self, batch_size):
        # CRITICAL FIX 1: Do NOT detach here. 
        # W_base must receive gradients from the very first chunk of the sequence.
        self.W_current = self.W_base.expand(batch_size, -1, -1).clone()
        
    def forward(self, x):
        b, seq_len, d = x.shape
        out = torch.zeros_like(x)
        
        if self.W_current is None or self.W_current.shape[0] != b:
            self.reset_state(b)
            
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size, :]
            
            # CRITICAL FIX 2: Truncated BPTT Boundary
            # Detach the state coming IN to the chunk to prevent OOM across the full sequence.
            # This breaks the graph between chunks, but preserves it INSIDE the chunk.
            curr_W = self.W_current.detach().requires_grad_(True)
            
            # 1. Dynamic Projections (MLPs)
            k_chunk = self.M_k(chunk)
            v_chunk = self.M_v(chunk)
            q_chunk = self.W_q(chunk)
            
            # 2. Self-Referential Targets
            v_hat = self.M_mem(v_chunk)
            
            # 3. L2 Normalization (Required by HOPE spec)
            k_norm = F.normalize(k_chunk, p=2, dim=-1)
            
            # 4. Delta Gradient Descent Update (Inner Loop)
            # CRITICAL FIX 3: Do NOT detach the result of dgd_update.
            # Gradients must flow backward from 'out_chunk' through 'new_W' into the MLPs.
            new_W = dgd_update(curr_W, k_norm, v_hat, lr=1e-3, chunk_size=chunk.size(1))
            
            # 5. Output Generation
            out_chunk = torch.bmm(q_chunk, new_W.transpose(1, 2))
            out[:, i:i+self.chunk_size, :] = out_chunk
            
            # 6. Save State for the next chunk
            self.W_current = new_W
            
        return out

if __name__ == '__main__':
    config = BAKAConfig()
    config.d_model = 64
    config.d_memory = 32
    titans = SelfModifyingTitans(config)
    x = torch.randn(2, 128, 64)
    titans.reset_state(2)
    out = titans(x)
    assert out.shape == x.shape
    print("PASS")