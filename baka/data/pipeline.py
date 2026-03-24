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

        # Memory map all files to avoid loading gigabytes into RAM
        self.mmaps = []
        self.file_lengths = []
        self.total_chunks = 0

        for f in self.data_files:
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


class NoRepeatSampler(Sampler):
    """Sampler that tracks which indices have been used across sessions.
    Saves state to a JSON file so training can resume without repeating data."""

    def __init__(self, total_size, state_file):
        self.total_size = total_size
        self.state_file = state_file
        self.used_indices = set()

        # Load previously used indices
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.used_indices = set(state.get('used_indices', []))
                print(f"Loaded {len(self.used_indices):,} used indices from {state_file}")

        remaining = total_size - len(self.used_indices)
        print(f"Data tracker: {len(self.used_indices):,} used / {total_size:,} total / {remaining:,} remaining")

        if remaining <= 0:
            print("WARNING: All data has been seen! Resetting tracker.")
            self.used_indices = set()

    def __iter__(self):
        # Get unused indices, shuffle them
        all_indices = set(range(self.total_size))
        remaining = list(all_indices - self.used_indices)
        np.random.shuffle(remaining)

        for idx in remaining:
            self.used_indices.add(idx)
            yield idx

    def __len__(self):
        return self.total_size - len(self.used_indices)

    def save_state(self):
        """Call this after training to save which indices were consumed."""
        with open(self.state_file, 'w') as f:
            json.dump({
                'used_indices': list(self.used_indices),
                'total_size': self.total_size
            }, f)
        print(f"Saved {len(self.used_indices):,} used indices to {self.state_file}")


def get_dataloader(data_dir, batch_size, seq_len=8192, num_workers=4, prefetch_factor=2):
    dataset = BAKADataset(data_dir, seq_len)

    state_file = os.path.join(data_dir, 'sampler_state.json')
    sampler = NoRepeatSampler(len(dataset), state_file)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True
    )
    # Attach sampler for easy access to save state
    loader.no_repeat_sampler = sampler
    return loader


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
        # Save state after training
        loader.no_repeat_sampler.save_state()
        print("PASS")
    finally:
        os.remove(dummy_path)
        state_f = 'baka/data/cache/sampler_state.json'
        if os.path.exists(state_f):
            os.remove(state_f)
