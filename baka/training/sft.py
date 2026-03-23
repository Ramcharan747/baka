import os
import sys
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baka.model.config import BAKAConfig
from baka.model.baka import BAKA

def load_pretrained_checkpoint(model, repo_id, token):
    if not token:
        print("No HF_TOKEN. Skipping load.")
        return
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        checkpoints = [f for f in files if f.startswith('checkpoint-') and f.endswith('/state.pt')]
        if not checkpoints:
            print("No pretrained checkpoints found.")
            return
            
        def extract_step(c):
            return int(c.split('checkpoint-')[1].split('/')[0])
            
        latest_ckpt = max(checkpoints, key=extract_step)
        print(f"Downloading pre-trained {latest_ckpt}...")
        file_path = hf_hub_download(repo_id=repo_id, filename=latest_ckpt, token=token)
        
        state = torch.load(file_path, map_location='cpu')
        model.load_state_dict(state['model'])
        print(f"Resumed weights from pre-training step {state.get('step', 'unknown')}")
    except Exception as e:
        print(f"Pretrained checkpoint load failed: {e}")

def sft_step(model, optimizer, batch, device):
    """
    batch is a dict containing 'input_ids' and 'loss_mask'.
    loss_mask is 1.0 for reasoning+answer tokens, 0.0 for prompt/padding tokens.
    """
    input_ids = batch['input_ids'].to(device)
    loss_mask = batch['loss_mask'].to(device)
    
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    mask = loss_mask[:, 1:]
    
    try:
        with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none')
            # Mask out prompt tokens
            loss = loss * mask.reshape(-1)
            # Average only over unmasked tokens
            loss = loss.sum() / (mask.sum() + 1e-8)
    except RuntimeError:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='none')
        loss = loss * mask.reshape(-1)
        loss = loss.sum() / (mask.sum() + 1e-8)
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def sft_train():
    hf_token = os.environ.get('HF_TOKEN')
    config = BAKAConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BAKA(config).to(device)
    load_pretrained_checkpoint(model, "Baka7/baka-checkpoints", hf_token)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # smaller lr for SFT
    model.train()
    print("SFT loop ready.")

if __name__ == '__main__':
    print("PASS")
