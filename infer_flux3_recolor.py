#!/usr/bin/env python
"""Run three-reference FLUX.3 recoloring with a local LoRA adapter."""

from flux3_recolor.inference import main


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
