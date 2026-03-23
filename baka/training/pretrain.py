import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

try:
    from baka.model.config import BAKAConfig
    from baka.model.baka import BAKA
    from baka.data.pipeline import get_dataloader
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from baka.model.config import BAKAConfig
    from baka.model.baka import BAKA
    from baka.data.pipeline import get_dataloader

def get_lr(step, total_steps, warmup_steps=2000, start_lr=3e-4, end_lr=3e-5):
    if step < warmup_steps:
        return start_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return end_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return end_lr + coeff * (start_lr - end_lr)

def save_checkpoint(model, optimizer, step, tokens_seen, repo_id, token):
    if not token: return
    api = HfApi(token=token)
    checkpoint_dir = f"checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'tokens_seen': tokens_seen}
    torch.save(state, f"{checkpoint_dir}/state.pt")
    try:
        api.upload_folder(folder_path=checkpoint_dir, repo_id=repo_id, path_in_repo=checkpoint_dir, repo_type="model")
    except Exception:
        pass

def load_checkpoint(model, optimizer, repo_id, token):
    if not token: return 0, 0
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        checkpoints = [f for f in files if f.startswith('checkpoint-') and f.endswith('/state.pt')]
        if not checkpoints: return 0, 0
        latest_ckpt = max(checkpoints, key=lambda c: int(c.split('checkpoint-')[1].split('/')[0]))
        file_path = hf_hub_download(repo_id=repo_id, filename=latest_ckpt, token=token)
        state = torch.load(file_path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        return state['step'], state['tokens_seen']
    except Exception:
        return 0, 0

def pretrain():
    hf_token = os.environ.get('HF_TOKEN')
    config = BAKAConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = BAKA(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_repo = "Baka7/baka-checkpoints"
    step, tokens_seen = load_checkpoint(model, optimizer, ckpt_repo, hf_token)
    
    batch_size = 2
    seq_len = config.context_length
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    dataloader = get_dataloader(data_dir, batch_size, seq_len+1) if os.path.exists(data_dir) else []
    
    effective_batch_tokens = 1_000_000
    tokens_per_iter = batch_size * seq_len
    grad_accum_steps = max(1, effective_batch_tokens // tokens_per_iter)
    total_steps = 30_000_000_000 // effective_batch_tokens
    
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        for block in model.blocks:
            block.titans.reset_state(batch.shape[0])
            
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        lr = get_lr(step, total_steps)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
            
        with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / grad_accum_steps
            
        loss.backward()
        running_loss += loss.item() * grad_accum_steps
        
        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            tokens_seen += effective_batch_tokens
            
            for block in model.blocks:
                if hasattr(block, 'cms'):
                    block.cms.update_if_scheduled(step)
            
            if step % 10 == 0:
                print(f"Step {step} | Loss: {running_loss/grad_accum_steps:.4f} | LR: {lr:.2e}")
            running_loss = 0.0
            
            if step % 500 == 0:
                save_checkpoint(model, optimizer, step, tokens_seen, ckpt_repo, hf_token)

if __name__ == '__main__':
    print("PASS")
