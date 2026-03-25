"""
BAKA 300M — Single A100 Training Script
Usage: python pretrain_1gpu.py --data pretrain_final.jsonl --resume
"""
import os
import sys
import math
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baka.model.config import BAKA300MConfig
from baka.model.baka import BAKA


# ---------------------------------------------------------------------------
# LR Schedule (validated — do not change)
# ---------------------------------------------------------------------------
def get_lr(step, total_steps, warmup_steps=2000, peak_lr=3e-4, min_lr=3e-5):
    """Cosine decay: warmup → peak → cosine decay to min_lr at exactly the last step."""
    if step < warmup_steps:
        return peak_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + (peak_lr - min_lr) * cosine


# ---------------------------------------------------------------------------
# Dataset — Pre-tokenizes JSONL to .bin cache, then memmaps
# ---------------------------------------------------------------------------
class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, seq_len, start_position=0):
        self.seq_len = seq_len
        cache_path = jsonl_path.replace('.jsonl', '.bin')

        if not os.path.exists(cache_path):
            print(f"Pre-tokenizing {jsonl_path} → {cache_path} ...")
            self._tokenize_to_bin(jsonl_path, cache_path)

        self.data = np.memmap(cache_path, dtype=np.uint32, mode='r')
        self.total_chunks = len(self.data) // (seq_len + 1)  # +1 for target shift
        self.start_position = start_position

        print(f"Dataset: {len(self.data):,} tokens, {self.total_chunks:,} chunks, "
              f"starting at position {start_position}")

    def _tokenize_to_bin(self, jsonl_path, cache_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        all_ids = []
        count = 0
        with open(jsonl_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get('text', '')
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
                count += 1
                if count % 100_000 == 0:
                    print(f"  Tokenized {count:,} documents, {len(all_ids):,} tokens...")

        print(f"  Total: {count:,} documents, {len(all_ids):,} tokens")
        arr = np.array(all_ids, dtype=np.uint32)
        arr.tofile(cache_path)
        print(f"  Saved to {cache_path} ({os.path.getsize(cache_path) / 1e9:.2f} GB)")

    def __len__(self):
        return self.total_chunks - self.start_position

    def __getitem__(self, idx):
        actual_idx = self.start_position + idx
        start = actual_idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ---------------------------------------------------------------------------
# Checkpoint save/load — CMS buffers included via state_dict buffers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, step, tokens_seen, data_position, loss, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
    torch.save({
        'step': step,
        'tokens_seen': tokens_seen,
        'data_position': data_position,
        'loss': loss,
        'model': model.state_dict(),        # includes CMS grad_W1_i, grad_W2_i buffers
        'optimizer': optimizer.state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path}")

    # Keep only last 3 checkpoints
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('ckpt_step')],
                   key=lambda f: int(f.split('step')[1].split('.')[0]))
    for old in ckpts[:-3]:
        os.remove(os.path.join(ckpt_dir, old))


def load_latest_checkpoint(model, optimizer, ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return 0, 0, 0
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('ckpt_step')],
                   key=lambda f: int(f.split('step')[1].split('.')[0]))
    if not ckpts:
        return 0, 0, 0

    path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {path}")
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])

    step = state['step']
    tokens_seen = state['tokens_seen']
    data_position = state['data_position']
    print(f"  Resumed: step={step}, tokens={tokens_seen/1e6:.1f}M, "
          f"data_position={data_position}")
    return step, tokens_seen, data_position


# ---------------------------------------------------------------------------
# Weight decay param groups
# ---------------------------------------------------------------------------
def get_param_groups(model, weight_decay=0.1):
    """Decay 2D+ params (linears), skip embeddings, norms, biases."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or 'embed' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BAKA 300M — single GPU training")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to pretrain_final.jsonl')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/300m',
                        help='Checkpoint directory')
    parser.add_argument('--total_steps', type=int, default=76_000,
                        help='Total training steps (10B tokens / 131072 tokens_per_step)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda')
    config = BAKA300MConfig()

    # Build model
    print(f"Building BAKA300M: {config.estimated_params():,} params")
    model = BAKA(config).to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    param_groups = get_param_groups(model)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1)

    # Resume
    start_step, tokens_seen, data_position = 0, 0, 0
    if args.resume:
        start_step, tokens_seen, data_position = load_latest_checkpoint(
            model, optimizer, args.ckpt_dir
        )

    # Dataset and DataLoader
    tokens_per_step = args.batch_size * config.context_length  # 64 * 2048 = 131,072
    dataset = PretrainDataset(args.data, config.context_length, start_position=data_position)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
    )

    # Training
    torch.backends.cudnn.benchmark = True
    model.train()
    step = start_step
    losses = []
    t_start = time.time()

    print(f"\nTraining: step {step} → {args.total_steps}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Data remaining: {len(dataset):,} chunks")
    print("=" * 80)

    for batch_x, batch_y in dataloader:
        # Reset Titans state
        for block in model.blocks:
            block.titans.reset_state(batch_x.shape[0])

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # LR schedule
        lr = get_lr(step, args.total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward + backward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(batch_x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   batch_y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # CMS scheduled updates
        for block in model.blocks:
            block.cms.update_if_scheduled(step)

        step += 1
        tokens_seen += tokens_per_step
        losses.append(loss.item())
        data_position += args.batch_size

        # Logging every 100 steps
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            elapsed = time.time() - t_start
            steps_done = step - start_step
            sps = steps_done / elapsed
            tokens_per_sec = sps * tokens_per_step
            eta_hours = (args.total_steps - step) / sps / 3600 if sps > 0 else 0

            # W_norm of M_mem in block 0
            w_norm = model.blocks[0].titans.M_mem.W_current.norm().item() if \
                     model.blocks[0].titans.M_mem.W_current is not None else 0.0

            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | "
                  f"Tokens: {tokens_seen/1e9:.2f}B | "
                  f"LR: {lr:.2e} | W_norm: {w_norm:.2f} | "
                  f"TPS: {tokens_per_sec/1e3:.1f}K | "
                  f"ETA: {eta_hours:.1f}h")

        # Checkpoint every 1000 steps
        if step % 1000 == 0:
            save_checkpoint(model, optimizer, step, tokens_seen,
                            data_position, losses[-1], args.ckpt_dir)

        if step >= args.total_steps:
            break

    # Final checkpoint
    if step > start_step:
        save_checkpoint(model, optimizer, step, tokens_seen,
                        data_position, losses[-1] if losses else 0.0, args.ckpt_dir)

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Training complete: {step} steps, {tokens_seen/1e9:.2f}B tokens, "
          f"{elapsed/3600:.1f}h")


if __name__ == '__main__':
    main()
