# Model Specifications and Runtime Performance

> Measured on NVIDIA GeForce RTX 3090 (24GB), single GPU, batch size 64.
> Models: ResNet-101 backbone + CustomEgoposeCascadedRefinementHead_enhanced (hmd_info_size=12, heatmap_decoder='efficient').

## Parameter Counts

| Component | Single-Stage | Cascaded |
|-----------|-------------|----------|
| Backbone (ResNet-101) | 42.50M | 42.50M |
| Head | 12.68M | 13.04M |
| **Total** | **55.18M** | **55.54M** |
| Stage 2 delta | — | **0.37M** |

### Stage 2 Parameter Breakdown (0.37M)

| Module | Parameters | Description |
|--------|-----------|-------------|
| `spatial_proj` | 131,136 | Per-joint backbone feature projection (2048→64) |
| `refinement_mlp` | 225,027 | Shared per-joint refinement network (355→256→3) |
| `pose_encoder_net` | 6,272 | Coarse pose context encoder (48→128) |
| `kin_encoder` | 3,904 | Kinematic feature encoder (60→64) |
| `hmd_encoder_stage2` | 416 | HMD encoder for Stage 2 (12→32) |
| **Total** | **366,755** | +0.66% over Single-Stage |

## Checkpoint Sizes

| Model | File Size |
|-------|----------|
| Cascaded + GBH (best ep8) | 216 MB |
| Single-stage (no-hmd, best ep9) | 214 MB |

## Training Throughput

Measured from training logs (epoch 1, last 20 iterations after warmup, single RTX 3090, batch=64):

| Model | Time/Iter | Samples/s |
|-------|----------|-----------|
| Single-stage | 0.221 s | 289.6 |
| Cascaded | 0.225 s | 284.4 |
| **Overhead** | +0.004 s (+1.8%) | -5.2 (-1.8%) |

## Inference FPS

### End-to-End (`model.test_step`)

Full pipeline: data preprocessing → backbone → head forward → decode (Stage 1 + Stage 2) → result packing.
Measured on real test data (H5 cache), 50 batches (3,200 samples) after 5 warmup batches. Single RTX 3090, batch=64.

| Model | FPS | ms/batch |
|-------|-----|----------|
| Single-stage | 216.2 | 296.0 |
| Cascaded | 207.8 | 307.9 |
| **Overhead** | **-8.4 (-3.9%)** | **+11.9 (+4.0%)** |

### Forward-Only (backbone + heatmap)

`torch.no_grad()`, 200 iterations, batch=64, synthetic input. This is the neural network compute portion only.

| Model | FPS | ms/batch |
|-------|-----|----------|
| Single-stage | 640.0 | 100.0 |
| Cascaded | 643.2 | 99.5 |

The ~430 FPS gap between forward-only and end-to-end comes from: data preprocessing (~40ms), heatmap codec decode (~20ms), Stage 1 encoder + pose decoder (~15ms), Stage 2 grid sampling + refinement MLP (~12ms for cascaded only), and result packing/CPU transfer (~20ms).

## For Paper Table

```
| Spec                  | Single-Stage | Cascaded (Ours) |
|-----------------------|-------------|-----------------|
| Total parameters      | 55.18M      | 55.54M          |
| Stage 2 parameters    | --          | 0.37M           |
| Checkpoint size       | 214 MB      | 216 MB          |
| Training throughput   | 289.6 sam/s | 284.4 sam/s     |
| Inference FPS (E2E)   | ~216        | ~208            |
```

GPU: NVIDIA GeForce RTX 3090 (24GB), single GPU.

## Previous Table Values (INCORRECT)

| Metric | Old (wrong) | New (verified) | Issue |
|--------|------------|----------------|-------|
| Total params (SS) | ~48.4M | **55.18M** | Off by ~7M (backbone was underestimated) |
| Total params (Casc) | ~48.8M | **55.54M** | Same issue |
| Checkpoint (SS) | 220 MB | **214 MB** | Minor |
| Checkpoint (Casc) | 221 MB | **216 MB** | Minor |
| Training throughput | 144.5 it/s | **289.6 samples/s** | Unit was samples/s not it/s; values also wrong |
| Inference FPS (E2E) | ~294 / ~290 | **~216 / ~208** | Previously unverified; end-to-end is lower than claimed |
