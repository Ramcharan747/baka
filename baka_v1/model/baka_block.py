import torch
import torch.nn as nn
from baka.model.config import BAKAConfig
try:
    from baka.model.titans import SelfModifyingTitans
    from baka.model.cms import CMSBlock
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baka.model.titans import SelfModifyingTitans
    from baka.model.cms import CMSBlock

class BAKABlock(nn.Module):
    def __init__(self, config: BAKAConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model)
        self.titans = SelfModifyingTitans(config)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.cms = CMSBlock(config)

    def forward(self, x):
        x = x + self.titans(self.norm1(x))
        x = x + self.cms(self.norm2(x))
        return x

if __name__ == '__main__':
    config = BAKAConfig()
    config.d_model = 64
    config.n_heads = 4
    config.d_head = 16
    config.d_ffn = 128
    block = BAKABlock(config).to(torch.bfloat16)
    x = torch.randn(2, 128, 64, dtype=torch.bfloat16)
    block.titans.reset_state(2)
    out = block(x)
    assert out.shape == x.shape
    print("PASS")
