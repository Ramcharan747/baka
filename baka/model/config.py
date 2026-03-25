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
    vocab_size: int = 32000  # must use Llama tokenizer

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

@dataclasses.dataclass
class BAKA300MConfig(BAKAConfig):
    """300M parameter config for A100 training. Verified: 289,818,624 params."""
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64               # d_model // n_heads
    d_ffn: int = 3072              # 4 × d_model
    d_memory: int = 64
    cms_levels: int = 4
    titans_chunk_size: int = 64
    context_length: int = 2048
    vocab_size: int = 65536        # Llama-3 tokenizer

if __name__ == '__main__':
    for name, cls in [("BAKAConfig", BAKAConfig),
                      ("BAKASmallConfig", BAKASmallConfig),
                      ("BAKA300MConfig", BAKA300MConfig)]:
        cfg = cls()
        print(f"\n{name}:")
        print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
              f"n_heads={cfg.n_heads}, d_head={cfg.d_head}")
        print(f"  estimated_params = {cfg.estimated_params():,}")

    # Detailed breakdown for 300M
    cfg = BAKA300MConfig()
    vocab_params = cfg.vocab_size * cfg.d_model
    titans_params = 5 * (2 * cfg.d_model * cfg.d_memory) + (cfg.d_model * cfg.n_heads * cfg.d_head)
    cms_params = cfg.cms_levels * (2 * cfg.d_model * cfg.d_ffn)
    norm_params = 2 * cfg.d_model
    layer_params = titans_params + cms_params + norm_params

    print(f"\nBAKA300MConfig breakdown:")
    print(f"  vocab_params   = {vocab_params:,}")
    print(f"  titans/layer   = {titans_params:,}")
    print(f"  cms/layer      = {cms_params:,}")
    print(f"  norm/layer     = {norm_params:,}")
    print(f"  layer_total    = {layer_params:,} × {cfg.n_layers} = {layer_params * cfg.n_layers:,}")
    print(f"  TOTAL          = {cfg.estimated_params():,}")

    # VRAM estimate at batch=64, seq=2048, bf16
    B, S = 64, 2048
    model_gb = cfg.estimated_params() * 2 / 1e9                    # bf16
    optim_gb = cfg.estimated_params() * 12 / 1e9                   # AdamW states (fp32 copy + m + v)
    grad_gb = cfg.estimated_params() * 4 / 1e9                     # fp32 grads
    act_gb = B * S * cfg.d_model * cfg.n_layers * 2 * 2 / 1e9      # rough activation estimate w/ checkpointing
    w_curr_gb = 5 * B * cfg.d_model * cfg.d_model * 4 / 1e9 * cfg.n_layers  # W_current fp32
    cms_buf_gb = cfg.cms_levels * 2 * cfg.d_model * cfg.d_ffn * 4 / 1e9 * cfg.n_layers
    total_gb = model_gb + optim_gb + grad_gb + act_gb + w_curr_gb + cms_buf_gb

    print(f"\nVRAM estimate (batch={B}, seq={S}):")
    print(f"  Model (bf16):       {model_gb:.2f} GB")
    print(f"  Optimizer (AdamW):  {optim_gb:.2f} GB")
    print(f"  Gradients (fp32):   {grad_gb:.2f} GB")
    print(f"  Activations:        {act_gb:.2f} GB")
    print(f"  W_current (5 mems): {w_curr_gb:.2f} GB")
    print(f"  CMS grad buffers:   {cms_buf_gb:.2f} GB")
    print(f"  TOTAL:              {total_gb:.2f} GB")
    print(f"  A100 80GB free:     {80 - total_gb:.2f} GB")

    assert cfg.d_head == cfg.d_model // cfg.n_heads, "d_head must equal d_model // n_heads"
    assert cfg.estimated_params() == 289_818_624, f"Expected 289,818,624 got {cfg.estimated_params():,}"
    print("\nPASS")

