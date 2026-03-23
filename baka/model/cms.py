import sys, os
import torch
import torch.nn as nn
from typing import List, Tuple

try:
    from baka.model.config import BAKAConfig
    from baka.model.dgd import MemoryMLP
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baka.model.config import BAKAConfig
    from baka.model.dgd import MemoryMLP

class CMSBlock(nn.Module):
    def __init__(self, config: BAKAConfig):
        super().__init__()
        self.levels = config.cms_levels
        # Bug 5 integrated natively scaling by d_ffn instead of d_memory
        self.mlps = nn.ModuleList([
            MemoryMLP(config.d_model, config.d_ffn) for _ in range(self.levels)
        ])
        
        self.chunk_sizes = config.cms_chunk_sizes[:self.levels]
        self.lrs = config.cms_lr[:self.levels]
        self.step = 0
        
        for i in range(self.levels):
            self.register_buffer(f'grad_W1_{i}', torch.zeros_like(self.mlps[i].W1.weight))
            self.register_buffer(f'grad_W2_{i}', torch.zeros_like(self.mlps[i].W2.weight))

    def forward(self, x):
        # Bug 4 Fix: applies sequentially in chain mode
        out = x
        for mlp in self.mlps:
            out = mlp(out)
        return out

    def accumulate_gradients(self, grads: List[Tuple[torch.Tensor, torch.Tensor]]):
        for i, (g1, g2) in enumerate(grads):
            getattr(self, f'grad_W1_{i}').add_(g1)
            getattr(self, f'grad_W2_{i}').add_(g2)

    def update_if_scheduled(self, step: int):
        self.step = step
        for i in range(self.levels):
            if self.step > 0 and self.step % self.chunk_sizes[i] == 0:
                with torch.no_grad():
                    g1 = getattr(self, f'grad_W1_{i}')
                    g2 = getattr(self, f'grad_W2_{i}')
                    self.mlps[i].W1.weight.data -= self.lrs[i] * g1
                    self.mlps[i].W2.weight.data -= self.lrs[i] * g2
                    g1.zero_()
                    g2.zero_()

if __name__ == '__main__':
    config = BAKAConfig()
    config.d_model = 64
    config.d_ffn = 128
    cms = CMSBlock(config)
    x = torch.randn(2, 64)
    out = cms(x)
    assert out.shape == x.shape
    print("PASS")
