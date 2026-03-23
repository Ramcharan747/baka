import torch
import torch.nn as nn
from baka.model.config import BAKAConfig
try:
    from baka.model.baka_block import BAKABlock
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from baka.model.baka_block import BAKABlock
import torch.utils.checkpoint as checkpoint

class BAKA(nn.Module):
    def __init__(self, config: BAKAConfig):
        super().__init__()
        self.config = config
        
        # Word embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Stack of BAKA blocks
        self.blocks = nn.ModuleList([
            BAKABlock(config) for _ in range(config.n_layers)
        ])
        
        # Final norm
        self.norm = nn.RMSNorm(config.d_model)
        
        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed.weight
        
    def forward(self, tokens_ids):
        # tokens_ids: [batch, seq_len]
        x = self.embed(tokens_ids)
        
        for block in self.blocks:
            # Rule 6: Wrap each BAKABlock in torch.utils.checkpoint.checkpoint()
            if self.training:
                # To support gradient checkpointing properly
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
                
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
        
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.0):
        """
        prompt_ids: [batch, seq_len] tensor of ints
        generates max_new_tokens sequentially
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = prompt_ids if prompt_ids.size(1) <= self.config.context_length \
                       else prompt_ids[:, -self.config.context_length:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] # Get the logits for the next token
            
            # --- NEW: Repetition Penalty Logic ---
            if repetition_penalty != 1.0:
                # Apply penalty to tokens that have already been generated in this sequence
                for b in range(prompt_ids.size(0)):
                    unique_tokens = torch.unique(prompt_ids[b])
                    score = logits[b, unique_tokens]
                    # If logit is negative, multiply by penalty to make it more negative.
                    # If positive, divide by penalty to shrink it.
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[b, unique_tokens] = score
            # -------------------------------------
            
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            prompt_ids = torch.cat((prompt_ids, idx_next), dim=1)
            
        return prompt_ids


if __name__ == '__main__':
    from baka.model.config import BAKAConfig
    config = BAKAConfig()
    
    # Run param check
    model = BAKA(config)
    
    # Calculate exactly
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter count: {total_params:,}")
    # should be ~1.3B
    
    # Mock small pass
    config.n_layers = 1
    config.d_model = 64
    config.n_heads = 4
    config.d_head = 16
    config.d_memory = 32
    config.vocab_size = 500
    config.cms_levels = 4
    config.titans_chunk_size = 64
    
    small_model = BAKA(config)
    
    # Sanity check logit shape
    x = torch.randint(0, config.vocab_size, (2, 128))
    
    small_model.eval()
    logits = small_model(x)
    assert logits.shape == (2, 128, config.vocab_size), f"Logit shape mismatch: {logits.shape}"
    
    print("PASS")
