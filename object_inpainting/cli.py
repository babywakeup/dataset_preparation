import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import build_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build object-reference inpainting pairs from images and instance masks."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workflow", type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--reference-resolution", type=int, default=512)
    parser.add_argument("--dilation-radius", type=int, default=4)
    parser.add_argument("--reference-padding", type=float, default=0.1)
    parser.add_argument("--min-object-fraction", type=float, default=0.01)
    parser.add_argument("--max-object-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=16)
    args = parser.parse_args()
    config = PipelineConfig(
        resolution=args.resolution,
        reference_resolution=args.reference_resolution,
        dilation_radius=args.dilation_radius,
        reference_padding=args.reference_padding,
        min_object_fraction=args.min_object_fraction,
        max_object_fraction=args.max_object_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    summary = build_dataset(args.input_manifest, args.output_dir, config, args.preview_count, args.workflow)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
