import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class StreamingTokenDataset(IterableDataset):
    def __init__(self, dataset_name, tokenizer_name, seq_len):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.eos_id = self.tokenizer.eos_token_id
        
        # streaming=True prevents downloading the massive dataset to local disk
        print(f"Initializing network stream for {self.dataset_name}...")
        self.dataset = load_dataset(self.dataset_name, name="sample-10BT", split="train", streaming=True)

    def __iter__(self):
        buffer = []
        # Continuously stream documents from HuggingFace
        for example in self.dataset:
            # Tokenize the text (add_special_tokens=False so we can manually control EOS)
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=False)
            
            # Append the EOS token to strictly separate documents
            tokens.append(self.eos_id)
            buffer.extend(tokens)
            
            # Yield perfectly sized chunks for the model's context window
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)

def get_dataloader(batch_size, seq_len):
    """
    Returns an infinite stream of tokens. 
    seq_len should be context_length + 1 (to create x and y shifted targets).
    """
    dataset = StreamingTokenDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer_name="Qwen/Qwen2.5-0.5B",
        seq_len=seq_len
    )
    
    # num_workers=0 is required for streaming IterableDatasets in basic Colab setups 
    # to prevent duplicate data streams across threads.
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

if __name__ == '__main__':
    # Quick sanity check test
    print("Testing streaming dataloader...")
    loader = get_dataloader(batch_size=2, seq_len=129)
    for i, batch in enumerate(loader):
        print(f"Batch {i} shape: {batch.shape}")
        if i == 2:
            break
    print("PASS")