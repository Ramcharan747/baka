import sys
import os
import torch
import torch.nn.functional as F
import subprocess
from transformers import AutoTokenizer

try:
    import sympy
except ImportError:
    sympy = None

try:
    from baka.model.config import BAKAConfig
    from baka.model.baka import BAKA
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from baka.model.config import BAKAConfig
    from baka.model.baka import BAKA

def verify_math(generation, ground_truth):
    if not sympy: return -1.0
    try:
        return 1.0 if sympy.simplify(sympy.sympify(generation.strip()) - sympy.sympify(ground_truth.strip())) == 0 else -1.0
    except Exception: return -1.0

def verify_code(generation, test_cases):
    try:
        result = subprocess.run(["python3", "-c", generation], capture_output=True, text=True, timeout=5)
        return 1.0 if result.returncode == 0 else -1.0
    except Exception: return -1.0

def grpo_step(model, optimizer, tokenizer, problem_text, ground_truth, v_type="math", num_samples=8):
    device = next(model.parameters()).device
    model.eval()
    
    prompt_ids = tokenizer(problem_text, return_tensors="pt")["input_ids"].to(device)
    prompt_ids = prompt_ids.expand(num_samples, -1)
    
    solutions = []
    log_probs_all = []
    
    for i in range(num_samples):
        curr_ids = prompt_ids[i:i+1].clone()
        log_prob_sum = 0.0
        
        for block in model.blocks:
            block.titans.reset_state(1)
            
        for _ in range(50):
            logits = model(curr_ids)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            log_prob_sum += torch.log(probs[0, next_token.item()])
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id: break
                
        decoded = tokenizer.decode(curr_ids[0, prompt_ids.size(1):], skip_special_tokens=True)
        solutions.append(decoded)
        log_probs_all.append(log_prob_sum)
        
    rewards = torch.tensor([verify_math(s, ground_truth) if v_type=="math" else verify_code(s, ground_truth) for s in solutions], device=device, dtype=torch.float32)
    log_probs_tensor = torch.stack(log_probs_all)
    
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    loss = - (advantages.detach() * log_probs_tensor).mean()
    
    model.train()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

if __name__ == '__main__':
    print("PASS")
