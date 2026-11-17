#!/usr/bin/env python
"""Compare a trainable text encoder + LoRA with the original fixed-text trainer.

Accepts the same arguments as train_flux3_recolor_lora.py. Original prompt encodes
only initialize one shared trainable [1,L,H] feature; target color stays image-only.
The original training entry remains the fixed-text baseline.
"""
import sys
import train_flux3_recolor_lora as trainer


def main(input_args=None):
    argv = list(sys.argv[1:] if input_args is None else input_args)
    if '--train_text_encoder' not in argv:
        argv.append('--train_text_encoder')
    args = trainer.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = 'outputs/flux3_recolor_text_encoder'
    return trainer.main(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc
