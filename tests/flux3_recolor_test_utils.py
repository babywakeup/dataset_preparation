"""Offline random FLUX.3/Qwen/VAE fixtures; never access the Hub."""
from pathlib import Path


def make_tiny_repository(root):
    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
    from transformers import (PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(12)
    root = Path(root)
    AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), layers_per_block=1, latent_channels=2,
                      norm_num_groups=4, mid_block_add_attention=False).save_pretrained(root / 'vae')
    Flux3Transformer2DModel(in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=1, joint_attention_dim=16, timestep_channels=8,
        axes_dims_rope=(4, 4, 4, 4), mlp_ratio=2).save_pretrained(root / 'transformer')
    Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        text_config={'vocab_size': 8, 'hidden_size': 16, 'intermediate_size': 16, 'num_hidden_layers': 1,
                     'num_attention_heads': 2, 'num_key_value_heads': 2, 'head_dim': 8,
                     'rope_parameters': {'rope_type': 'default', 'rope_theta': 1000000., 'mrope_section': [1, 1, 2]}},
        vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 16, 'num_heads': 2,
                       'out_hidden_size': 16, 'deepstack_visual_indexes': []},
        image_token_id=2, video_token_id=3, vision_start_token_id=4, vision_end_token_id=5,
    )).save_pretrained(root / 'text_encoder')
    vocab = {'<unk>': 0, '</s>': 1, '<|image_pad|>': 2, '<|video_pad|>': 3,
             '<|vision_start|>': 4, '<|vision_end|>': 5, 'recolor': 6, 'red': 7}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single='$A </s>', special_tokens=[('</s>', 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', eos_token='</s>', pad_token='</s>')
    Qwen3VLProcessor(tokenizer=tokenizer, image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen3VLVideoProcessor(), chat_template="{{ messages[0]['content'] }}").save_pretrained(root / 'tokenizer')
    return root


def make_dataset(root, count=7):
    from PIL import Image
    from flux3_recolor import CONDITION_FOLDERS, TARGET_FOLDER
    root = Path(root)
    for folder in (TARGET_FOLDER, *CONDITION_FOLDERS, '主颜色B'):
        (root / folder).mkdir(parents=True)
        for i in range(count):
            Image.new('RGB', (32, 32), (i * 25, 32, 80)).save(root / folder / f'{i}.png')
    return root


def make_training_dataset(args, names):
    """Prepare metadata fixtures explicitly, then call the real lazy Dataset."""
    from datasets import Dataset
    from build_flux3_recolor_dataset import generate_samples
    from flux3_recolor.config import read_json
    from train_flux3_recolor_lora import DreamBoothDataset
    captions = read_json(args.caption_json) if args.caption_json else None
    rows = []
    for index, direction in enumerate(args.training_directions):
        rows.extend(generate_samples(args.instance_data_dir, direction, names, index,
                                     prompt=args.instance_prompt, captions=captions))
    return DreamBoothDataset(args, Dataset.from_list(rows))
