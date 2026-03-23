import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BAKADataset(Dataset):
    def __init__(self, data_dir, seq_len=8192):
        self.seq_len = seq_len
        self.data_files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith('.bin')
        ]
        
        # Memory map all files to avoid loading gigabytes into RAM
        self.mmaps = []
        self.file_lengths = []
        self.total_chunks = 0
        
        for f in self.data_files:
            m = np.memmap(f, dtype=np.uint16, mode='r')
            num_tokens = len(m)
            # Each chunk is seq_len tokens
            num_chunks = num_tokens // seq_len
            
            if num_chunks > 0:
                self.mmaps.append(m)
                self.file_lengths.append(num_chunks)
                self.total_chunks += num_chunks
                
        # To find which file a chunk belongs to efficiently
        self.cumulative_lengths = np.cumsum(self.file_lengths)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        # Find which file this chunk belongs to
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # Calculate local chunk index inside that file
        if file_idx == 0:
            local_chunk_idx = idx
        else:
            local_chunk_idx = idx - self.cumulative_lengths[file_idx - 1]
            
        start_idx = local_chunk_idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Read the slice from memmap
        # We cast to int64 directly for torch embedding compatibility
        chunk = self.mmaps[file_idx][start_idx:end_idx].astype(np.int64)
        
        # Returns chunks of 8192 tokens
        return torch.from_numpy(chunk)


def get_dataloader(data_dir, batch_size, seq_len=8192, num_workers=4, prefetch_factor=2):
    dataset = BAKADataset(data_dir, seq_len)
    
    # DataLoader with proper shuffling and prefetching
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True, # prefetching to GPU
        drop_last=True
    )
    return loader


if __name__ == '__main__':
    # Sanity check
    os.makedirs('baka/data/cache', exist_ok=True)
    # create dummy bin file with roughly 20000 tokens
    dummy = np.arange(20000, dtype=np.uint16)
    dummy_path = 'baka/data/cache/dummy_test.bin'
    with open(dummy_path, 'wb') as f:
        f.write(dummy.tobytes())
        
    loader = get_dataloader('baka/data/cache', batch_size=2, seq_len=8192, num_workers=0)
    
    try:
        for batch in loader:
            assert batch.shape == (2, 8192), f"Batch shape expected (2, 8192) got {batch.shape}"
            break
        print("PASS")
    finally:
        # cleanup dummy
        os.remove(dummy_path)
