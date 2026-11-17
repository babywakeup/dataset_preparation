# Training / Pipeline No-Grad Comparison

**Goal:** User-requested standalone Python script checks the actual no-LoRA training path against the native pipeline at identical noisy target and sigma, and runs real weights on GPU1 after user stopped training.

**Design:** Reuse Dataset/collate/encode_latents/flow_matching_loss unchanged. Reproduce the helper's RNG draw order via fork_rng rather than replacing its noise/sampler. Keep native RGB/VAE/text/position blocks and disable only prompt upsampling and CFG. Supply inverse-shift sigma so the normal scheduler applies shift once. Capture real Transformer inputs and predictions without canonicalizing VAE layout; separately replay exact inputs to isolate arithmetic from preprocessing differences. Persist per-stage tensor differences, training/native prediction losses, one-step latent differences, no-grad/no-LoRA counts and optional tensors.

**Constraints:** GPU1 only; no optimizer, no adapter, no backward, no existing training data/checkpoint changes. Initial GPU1 memory was busy; user explicitly stopped training and authorized GPU, recheck showed empty. Test per-step conditional velocity rather than asserting final guided multi-step image equals a training scalar.

- [x] RED two CPU tests on missing script.
- [x] Implement standalone compare_flux3_training_pipeline.py and difference-report tests.
- [x] CPU random tiny repository: both tests pass; actual differences preserved, matched-input replay passes.
- [x] Run real sample A→B and B→A on GPU1 with BF16 and record results honestly.
- [x] Review/report any discrepancy; run relevant regression and update guide/handoff.

**User steering:** Official example specifies original FP32 VAE weights, BF16 other components, no global autocast. Default audit pipeline now uses the official local factory and load_components precision; the production training path remains unchanged as the baseline. Preserve optional same-VAE/training-autocast reproduction mode.

**Evidence:** Eight CPU regressions passed; real GPU1 six cases completed and saved under outputs/flux3_pipeline_compare_verified. Identical inputs plus identical context replay exactly; native FP32 VAE and autocast differences prevent asserting equivalence. Manual loss differs at most5.96e-8. Reviewer found no important code issues. No optimizer/backward/checkpoint/data changes; GPU1 released after process exit.
