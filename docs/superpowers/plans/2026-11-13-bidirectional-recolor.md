# Bidirectional Recolor Dataset Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in the current workspace.

**Goal:** Merge B→A and A→B tasks so synthetic B images also provide supervision.

**Architecture:** Keep source-level train/validation splits, then expand directional records in the shared Dataset. Central task definitions drive strict file inspection, preprocessing, identity caches, configuration, validation and inference. Default to both directions equally; explicit single B→A preserves legacy data and checkpoint behavior.

**Tech Stack:** Python, Pillow, PyTorch, Diffusers, PEFT, Accelerate, unittest.

**Spec:** User-approved two-subdataset proposal in this conversation: gray/B/colorA→A and gray/A/colorB→B; same three-image prompt roles, separate validation/loss and cache identities, no source leakage, new output directory.

## Constraints and review focus

- No raw image rewrites, downloads, existing output changes or real GPU training.
- Keep original top-level adapter folder metadata for backward compatibility; persist complete subdataset definitions separately.
- Missing/mispaired B swatches must fail before model loading; include them in the dataset fingerprint.
- Old one-direction checkpoints can resume only with explicit one-direction selection and unchanged data/config.
- Every source belongs to one split before both directions expand; grouped source variants remain together.
- Validation must render both directions and log distinct TensorBoard tags, while old single-direction paths continue working.
- Compute directional loss from per-sample errors, not by assigning mixed-batch mean loss to both directions.

## Tasks

- [x] Add failing tests using distinct A/B pixels and swatches, directional cache round trips, source splits, missing/changed B swatches, two-direction validation and resume.
- [x] Implement task constants/specs and selected-folder inspection; expand Dataset records and stable direction identities.
- [x] Add per-sample loss support, direction statistics and strict checkpoint metadata/default normalization.
- [x] Extend validation/inference and tracker paths for per-direction outputs, preserving old single-direction paths.
- [x] Run focused RED/GREEN and CPU integration/regression checks; inspect real dataset without rewriting it.
- [x] Update user guide, architecture navigation and iteration handoff with actual results and limitations.

## Execution evidence

- Initial RED: default dataset returned4 instead of8 repeated directional records; inspector lacked directions argument. Integration initially lacked subdatasets metadata.
- GREEN: 61 CPU regressions passed, then final7 directional checks passed after console/help-only refinements.
- Review caught generated.png source directory being matched by recursive artifact search. Real validation reproduced IsADirectoryError; filter now requires is_file, and the integrated directional test holds out this exact filename.
- Real check_dataset:1754 sources, five paired folders; train1579/validation175, expanded3158/350 records at repeats1. No raw/output/model changes.
- Two-process CPU Accelerate/DDP acceptance passed: both directions, disjoint rank sample identities, synchronized parameters, validation continuation, exact uninterrupted/resumed adapter comparison.
- Ruling: preserve old B→A sample identities and interpret absent direction metadata as B→A for explicit single-direction resume; new training defaults to both and cannot silently resume old single-direction semantics.
