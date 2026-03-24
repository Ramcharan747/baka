import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler


class BAKADataset(Dataset):
    def __init__(self, data_dir, seq_len=8192):
        self.seq_len = seq_len
        self.data_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.bin')
        ])

        self.mmaps = []
        self.file_lengths = []
        self.total_chunks = 0

        for f in self.data_files:
            fsize = os.path.getsize(f)
            if fsize == 0:
                continue
            m = np.memmap(f, dtype=np.uint16, mode='r')
            num_tokens = len(m)
            num_chunks = num_tokens // seq_len

            if num_chunks > 0:
                self.mmaps.append(m)
                self.file_lengths.append(num_chunks)
                self.total_chunks += num_chunks

        self.cumulative_lengths = np.cumsum(self.file_lengths)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')

        if file_idx == 0:
            local_chunk_idx = idx
        else:
            local_chunk_idx = idx - self.cumulative_lengths[file_idx - 1]

        start_idx = local_chunk_idx * self.seq_len
        end_idx = start_idx + self.seq_len

        chunk = self.mmaps[file_idx][start_idx:end_idx].astype(np.int64)
        return torch.from_numpy(chunk)


class SequentialSampler:
    """Walks through data in order. Tracks position across sessions via JSON file."""

    def __init__(self, total_size, state_file):
        self.total_size = total_size
        self.state_file = state_file
        self.start_offset = 0

        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.start_offset = state.get('next_index', 0)

        remaining = total_size - self.start_offset
        if remaining <= 0:
            print("WARNING: All data consumed! Wrapping to start.")
            self.start_offset = 0
            remaining = total_size

        print(f"Data position: {self.start_offset:,} / {total_size:,} "
              f"({self.start_offset/total_size*100:.1f}% used, {remaining:,} remaining)")

        self.current = self.start_offset

    def __iter__(self):
        self.current = self.start_offset
        while self.current < self.total_size:
            yield self.current
            self.current += 1

    def __len__(self):
        return self.total_size - self.start_offset

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                'next_index': self.current,
                'total_size': self.total_size
            }, f)
        pct = self.current / self.total_size * 100
        print(f"Saved position {self.current:,} / {self.total_size:,} ({pct:.1f}%)")


def get_dataloader(data_dir, batch_size, seq_len=8192, num_workers=4, prefetch_factor=2):
    dataset = BAKADataset(data_dir, seq_len)

    state_file = os.path.join(data_dir, 'sampler_state.json')
    sampler = SequentialSampler(len(dataset), state_file)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True
    )
    loader.sequential_sampler = sampler
    return loader


def calculate_shards_needed(state_file, steps, batch_size, seq_len, tokens_per_shard=250_000_000):
    """Calculate which shard numbers are needed for the next N steps."""
    start_chunk = 0
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            start_chunk = json.load(f).get('next_index', 0)

    chunks_per_shard = tokens_per_shard // seq_len
    chunks_needed = steps * batch_size
    start_shard = start_chunk // chunks_per_shard
    end_shard = (start_chunk + chunks_needed) // chunks_per_shard

    shards = list(range(start_shard, end_shard + 1))
    print(f"Position: chunk {start_chunk:,}")
    print(f"Need {chunks_needed:,} chunks for {steps} steps")
    print(f"Shards needed: {shards}")
    return shards


if __name__ == '__main__':
    os.makedirs('baka/data/cache', exist_ok=True)
    dummy = np.arange(20000, dtype=np.uint16)
    dummy_path = 'baka/data/cache/dummy_test.bin'
    with open(dummy_path, 'wb') as f:
        f.write(dummy.tobytes())

    loader = get_dataloader('baka/data/cache', batch_size=2, seq_len=8192, num_workers=0)

    try:
        for batch in loader:
            assert batch.shape == (2, 8192), f"Batch shape expected (2, 8192) got {batch.shape}"
            break
        loader.sequential_sampler.save_state()
        print("PASS")
    finally:
        os.remove(dummy_path)
        state_f = 'baka/data/cache/sampler_state.json'
        if os.path.exists(state_f):
            os.remove(state_f)
