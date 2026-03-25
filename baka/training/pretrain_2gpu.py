"""
BAKA 300M — 2× A100 DDP Training Script
Usage: torchrun --nproc_per_node=2 pretrain_2gpu.py --data pretrain_final.jsonl --resume

NCCL env vars (set by torchrun or in SLURM script):
  MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
    def __init__(self, jsonl_path, seq_len, start_position=0,
                 tokenizer_name="meta-llama/Llama-3.2-1B"):
        self.seq_len = seq_len
        cache_path = jsonl_path.replace('.jsonl', '.bin')

        self._tokenizer_name = tokenizer_name
        if not os.path.exists(cache_path):
            # Only rank 0 tokenizes
            if dist.get_rank() == 0:
                print(f"Pre-tokenizing {jsonl_path} → {cache_path} ...")
                self._tokenize_to_bin(jsonl_path, cache_path, tokenizer_name)
            dist.barrier()  # wait for rank 0 to finish

        self.data = np.memmap(cache_path, dtype=np.uint32, mode='r')
        self.total_chunks = len(self.data) // (seq_len + 1)
        self.start_position = start_position

    def _tokenize_to_bin(self, jsonl_path, cache_path, tokenizer_name="meta-llama/Llama-3.2-1B"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
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

        total_tokens = len(all_ids)
        print(f"  Total: {count:,} documents, {total_tokens:,} tokens ({total_tokens/1e9:.2f}B)")
        assert total_tokens > 1_000_000, (
            f"Only {total_tokens:,} tokens — check JSONL file for corruption"
        )

        arr = np.array(all_ids, dtype=np.uint32)
        max_id = arr.max()
        print(f"  Max token ID: {max_id} (vocab_size: {tokenizer.vocab_size})")
        assert max_id < 65536, f"Token ID {max_id} exceeds vocab_size 65536!"

        arr.tofile(cache_path)
        print(f"  Saved: {count:,} docs, {total_tokens:,} tokens, "
              f"{os.path.getsize(cache_path)/1e9:.2f} GB")

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
    # Save the unwrapped model (DDP module)
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'step': step,
        'tokens_seen': tokens_seen,
        'data_position': data_position,
        'loss': loss,
        'model': model_state,
        'optimizer': optimizer.state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path}")

    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('ckpt_step')],
                   key=lambda f: int(f.split('step')[1].split('.')[0]))
    for old in ckpts[:-3]:
        os.remove(os.path.join(ckpt_dir, old))


def load_latest_checkpoint(model, optimizer, ckpt_dir):
    """Load checkpoint BEFORE wrapping in DDP."""
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
    return state['step'], state['tokens_seen'], state['data_position']


# ---------------------------------------------------------------------------
# Weight decay param groups
# ---------------------------------------------------------------------------
def get_param_groups(model, weight_decay=0.1):
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BAKA 300M — 2-GPU DDP training")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/300m')
    parser.add_argument('--total_steps', type=int, default=38_000,
                        help='Total steps (10B tokens / 262144 tokens_per_step)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size PER GPU')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-3.2-1B',
                        help='HuggingFace tokenizer name (fallback: huggyllama/llama-7b)')
    args = parser.parse_args()

    # DDP init
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    is_master = (rank == 0)

    if is_master:
        print(f"DDP: world_size={world_size}, backend=nccl")

    config = BAKA300MConfig()

    # Build model on GPU
    model = BAKA(config).to(device)
    if is_master:
        print(f"BAKA300M: {config.estimated_params():,} params")

    # Optimizer (before loading checkpoint)
    param_groups = get_param_groups(model)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1)

    # Resume BEFORE DDP wrapping
    start_step, tokens_seen, data_position = 0, 0, 0
    if args.resume:
        start_step, tokens_seen, data_position = load_latest_checkpoint(
            model, optimizer, args.ckpt_dir
        )

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank])

    # Dataset and DataLoader with DistributedSampler
    tokens_per_step = args.batch_size * config.context_length * world_size  # 64*2048*2 = 262,144
    dataset = PretrainDataset(args.data, config.context_length,
                              start_position=data_position,
                              tokenizer_name=args.tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
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

    if is_master:
        print(f"\nTraining: step {step} → {args.total_steps}")
        print(f"Tokens per step: {tokens_per_step:,} ({args.batch_size}×{config.context_length}×{world_size})")
        print("=" * 80)

    for batch_x, batch_y in dataloader:
        # Reset Titans state
        raw_model = model.module
        for block in raw_model.blocks:
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
        for block in raw_model.blocks:
            block.cms.update_if_scheduled(step)

        step += 1
        tokens_seen += tokens_per_step
        losses.append(loss.item())
        data_position += args.batch_size * world_size

        # Logging (master only)
        if step % 100 == 0 and is_master:
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            elapsed = time.time() - t_start
            steps_done = step - start_step
            sps = steps_done / elapsed
            tokens_per_sec = sps * tokens_per_step
            eta_hours = (args.total_steps - step) / sps / 3600 if sps > 0 else 0

            w_norm = raw_model.blocks[0].titans.M_mem.W_current.norm().item() if \
                     raw_model.blocks[0].titans.M_mem.W_current is not None else 0.0

            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | "
                  f"Tokens: {tokens_seen/1e9:.2f}B | "
                  f"LR: {lr:.2e} | W_norm: {w_norm:.2f} | "
                  f"TPS: {tokens_per_sec/1e3:.1f}K | "
                  f"ETA: {eta_hours:.1f}h")

        # Checkpoint (master only)
        if step % 1000 == 0 and is_master:
            save_checkpoint(model, optimizer, step, tokens_seen,
                            data_position, losses[-1], args.ckpt_dir)

        if step >= args.total_steps:
            break

    # Final
    if step > start_step and is_master:
        save_checkpoint(model, optimizer, step, tokens_seen,
                        data_position, losses[-1] if losses else 0.0, args.ckpt_dir)
        elapsed = time.time() - t_start
        print("=" * 80)
        print(f"Training complete: {step} steps, {tokens_seen/1e9:.2f}B tokens, "
              f"{elapsed/3600:.1f}h")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
