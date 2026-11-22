"""Dimension-aware batch sampling with equal batch counts for Accelerate/DDP."""

import random

from torch.utils.data import BatchSampler


class BucketBatchSampler(BatchSampler):
    """Shuffle within each image-size bucket and pad whole batches for DDP."""

    batch_size = None
    drop_last = False

    def __init__(self, dataset, batch_size, seed=42, num_processes=1):
        self.dataset, self.requested_batch_size = dataset, batch_size
        self.seed, self.num_processes, self.epoch = seed, num_processes, 0
        self.sampler = range(len(dataset))

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for members in self.dataset.bucket_indices:
            indices = list(members)
            rng.shuffle(indices)
            batches.extend(indices[i:i + self.requested_batch_size]
                           for i in range(0, len(indices), self.requested_batch_size))
        return self._pad_batches(batches, rng)

    def _pad_batches(self, batches, rng):
        rng.shuffle(batches)
        if not batches:
            raise ValueError('训练DataLoader为空')
        original = list(batches)
        while len(batches) % self.num_processes:
            batches.append(original[(len(batches) - len(original)) % len(original)])
        return batches

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())
