"""Offline two-rank acceptance worker; artifacts stay in the supplied temp output."""
import json
import os
import sys
from pathlib import Path

from train_flux3_recolor_lora import main, parse_args, DreamBoothDataset, BucketBatchSampler
from safetensors.torch import save_file

args = parse_args(sys.argv[1:])
result = main(args)
rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
output = Path(args.output_dir) / 'rank-artifacts'
output.mkdir(exist_ok=True)
parameters = {n: p.detach().cpu().contiguous() for n, p in result['model'].named_parameters() if p.requires_grad}
if result.get('text_encoder') is not None:
    parameters.update({f'text_encoder.{n}': p.detach().cpu().contiguous()
                       for n, p in result['text_encoder'].named_parameters()})
save_file(parameters, str(output / f'rank-{rank}.safetensors'))
split = json.loads((Path(args.output_dir) / 'split.json').read_text())
from datasets import load_from_disk
config = json.loads((Path(args.output_dir) / 'recolor_config.json').read_text())
dataset = DreamBoothDataset(args, load_from_disk(config['dataset_path'])['train'])
sampler = BucketBatchSampler(dataset, args.train_batch_size, args.seed, world)
shard = [dataset.sample_ids[index] for batch in sampler.batches()[rank::world] for index in batch]
(output / f'rank-{rank}.json').write_text(json.dumps(shard))
