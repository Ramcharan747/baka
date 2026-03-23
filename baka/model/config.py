import dataclasses
from typing import List

@dataclasses.dataclass
class BAKAConfig:
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_head: int = 128
    d_ffn: int = 8192
    d_memory: int = 256
    cms_levels: int = 4
    cms_chunk_sizes: List[int] = dataclasses.field(default_factory=lambda: [16, 256, 4096, 65536])
    cms_lr: List[float] = dataclasses.field(default_factory=lambda: [1e-3, 1e-4, 1e-5, 1e-6])
    titans_chunk_size: int = 64
    context_length: int = 8192
    vocab_size: int = 32000

    def estimated_params(self):
        vocab_params = self.vocab_size * self.d_model
        titans_params = 5 * (2 * self.d_model * self.d_memory) + (self.d_model * self.n_heads * self.d_head)
        cms_params = self.cms_levels * (2 * self.d_model * self.d_ffn)
        norm_params = 2 * self.d_model
        layer_params = titans_params + cms_params + norm_params
        return vocab_params + self.n_layers * layer_params

@dataclasses.dataclass
class BAKASmallConfig(BAKAConfig):
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_head: int = 32
    d_ffn: int = 1024
    d_memory: int = 32
    cms_levels: int = 4
    titans_chunk_size: int = 16
    cms_chunk_sizes: List[int] = dataclasses.field(default_factory=lambda: [16, 64, 256, 1024])

if __name__ == '__main__':
    config = BAKAConfig()
    print(f"Full config approximate parameter count: {config.estimated_params():,}")
    print("PASS")
