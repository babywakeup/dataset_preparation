"""Epoch-scheduled direction mixing while preserving image-size buckets."""

import random

from .bucket import BucketBatchSampler


class ScheduledMixBucketBatchSampler(BucketBatchSampler):
    """Keep each epoch's sample count and bucket sizes while changing AtoB share.

    The sampler builds one deterministic global batch plan. Accelerate shards
    that plan once, after DataLoader preparation; no rank-local slicing is done
    here. Direction-specific buckets that have no examples from the other side
    remain fixed, and the requested/achieved ratios can differ in that case.
    """

    def __init__(self, dataset, batch_size, seed=42, num_processes=1,
                 atob_ratio_start=0.5, atob_ratio_end=0.25, total_epochs=1):
        super().__init__(dataset, batch_size, seed, num_processes)
        if not 0 <= atob_ratio_start <= 1 or not 0 <= atob_ratio_end <= 1:
            raise ValueError('AtoB比例必须在[0,1]范围内')
        if total_epochs < 1:
            raise ValueError('total_epochs必须为正整数')
        if len(dataset.direction_ids) != len(dataset):
            raise ValueError('Dataset方向标签数量与样本数不一致')
        self.atob_ratio_start = atob_ratio_start
        self.atob_ratio_end = atob_ratio_end
        self.total_epochs = total_epochs

    def set_total_epochs(self, total_epochs):
        if total_epochs < 1:
            raise ValueError('total_epochs必须为正整数')
        self.total_epochs = total_epochs

    def ratio_for_epoch(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        progress = min(max(epoch, 0) / max(self.total_epochs - 1, 1), 1.)
        return self.atob_ratio_start + (self.atob_ratio_end - self.atob_ratio_start) * progress

    def _counts_per_bucket(self, ratio, rng):
        grouped = []
        total = 0
        min_atob = max_atob = 0
        for members in self.dataset.bucket_indices:
            by_direction = {'a_to_b': [], 'b_to_a': []}
            for index in members:
                direction = self.dataset.direction_ids[index]
                if direction not in by_direction:
                    raise ValueError(f'未知训练方向: {direction}')
                by_direction[direction].append(index)
            grouped.append(by_direction)
            count = len(members)
            if not count:
                continue
            total += count
            if by_direction['a_to_b'] and by_direction['b_to_a']:
                max_atob += count
            elif by_direction['a_to_b']:
                min_atob += count
                max_atob += count
            else:
                min_atob += 0
        if total == 0:
            raise ValueError('训练DataLoader为空')

        counts = []
        for by_direction in grouped:
            n_atob, n_btoa = len(by_direction['a_to_b']), len(by_direction['b_to_a'])
            count = n_atob + n_btoa
            counts.append(count if n_atob and not n_btoa else 0 if n_btoa and not n_atob
                          else round(count * ratio))
        desired = min(max(round(total * ratio), min_atob), max_atob)
        difference = desired - sum(counts)
        flexible = [i for i, by_direction in enumerate(grouped)
                    if by_direction['a_to_b'] and by_direction['b_to_a']]
        rng.shuffle(flexible)
        for bucket_index in flexible:
            capacity = (len(grouped[bucket_index]['a_to_b']) +
                        len(grouped[bucket_index]['b_to_a']) - counts[bucket_index])
            if difference > 0:
                adjustment = min(difference, capacity)
                counts[bucket_index] += adjustment
                difference -= adjustment
            elif difference < 0:
                adjustment = min(-difference, counts[bucket_index])
                counts[bucket_index] -= adjustment
                difference += adjustment
            if difference == 0:
                break
        return grouped, counts

    @staticmethod
    def _draw(pool, count, rng):
        if count <= len(pool):
            return rng.sample(pool, count)
        return [rng.choice(pool) for _ in range(count)]

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        ratio = self.ratio_for_epoch()
        grouped, atob_counts = self._counts_per_bucket(ratio, rng)
        batches = []
        for by_direction, n_atob in zip(grouped, atob_counts):
            n_btoa = len(by_direction['a_to_b']) + len(by_direction['b_to_a']) - n_atob
            chosen = self._draw(by_direction['a_to_b'], n_atob, rng)
            chosen.extend(self._draw(by_direction['b_to_a'], n_btoa, rng))
            rng.shuffle(chosen)
            batches.extend(chosen[i:i + self.requested_batch_size]
                           for i in range(0, len(chosen), self.requested_batch_size))
        result = self._pad_batches(batches, rng)
        sampled_indices = [index for batch in result for index in batch]
        sampled_atob = sum(self.dataset.direction_ids[index] == 'a_to_b' for index in sampled_indices)
        self.last_mix = {'epoch': self.epoch, 'requested_atob_ratio': ratio,
                         'atob_samples': sampled_atob, 'total_samples': len(sampled_indices),
                         'achieved_atob_ratio': sampled_atob / len(sampled_indices)}
        return result
