# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMPose is OpenMMLab's pose estimation toolbox built on PyTorch. This fork extends MMPose with egocentric (first-person) pose estimation for HMD (Head-Mounted Display) applications, including dual-backbone architectures and segmentation/depth prediction.

## Common Commands

### Training
```bash
# Single GPU
python tools/train.py <config_path>

# Multi-GPU (distributed)
bash tools/dist_train.sh <config_path> <num_gpus>

# With options
python tools/train.py <config> --work-dir <dir> --amp --resume
```

### Testing/Evaluation
```bash
# Single GPU
python tools/test.py <config_path> <checkpoint_path>

# Multi-GPU
bash tools/dist_test.sh <config_path> <checkpoint_path> <num_gpus>
```

### Running Tests
```bash
pytest tests/
```

### Custom Dataset Cache (Performance Optimization)
```bash
# Build annotation-only H5 cache (fast metadata loading)
python tools/dataset_converters/build_egopose_h5cache.py \
    --data-root /path/to/dataset --num-workers 4

# Build H5 cache WITH embedded images (eliminates disk I/O bottleneck)
# Images are preprocessed to 256x256, stored uncompressed for NVMe SSD
python tools/dataset_converters/build_egopose_h5cache_with_images.py \
    --data-root /path/to/dataset \
    --output /mnt/nvme/h5cache/train_cache_with_images.h5 \
    --num-workers 8
```

**Cache location (Linux)**: `/mnt/dataset_vol/h5cache/` (NVMe SSD recommended)
- `train_cache_with_images.h5` (39GB, 210k samples)
- `val_cache_with_images.h5` (2.8GB, 15k samples)
- `test_cache_with_images.h5` (15GB, 115k samples)

## Architecture Overview

### Registry Pattern
All components use MMEngine's registry system. To add custom components:
1. Implement class inheriting from base (e.g., `BaseDataset`, `BaseLoss`)
2. Decorate with `@REGISTRY.register_module()`
3. Reference by type name in config files

Key registries: `MODELS`, `DATASETS`, `KEYPOINT_CODECS`, `TRANSFORMS`, `METRICS`, `HOOKS`

### Configuration System
Configs use inheritance via `_base_` list. Override any setting by redefining it.
```python
_base_ = ['path/to/base_config.py']
model = dict(type='CustomModel', ...)  # Overrides base
```

### Core Module Structure
- `mmpose/models/pose_estimators/` - Top-level models (TopdownPoseEstimator, etc.)
- `mmpose/models/backbones/` - Feature extractors (ResNet, HRNet)
- `mmpose/models/heads/` - Prediction heads (heatmap, regression)
- `mmpose/datasets/datasets/` - Dataset implementations
- `mmpose/codecs/` - Keypoint encoding/decoding (heatmaps ↔ coordinates)
- `mmpose/evaluation/metrics/` - Evaluation metrics

### Data Flow
1. Config loaded → Components built from registry
2. Dataset + Transforms → DataLoader
3. Model: backbone(s) → neck → head → loss
4. Codec: encodes GT keypoints to heatmaps, decodes predictions back

## Custom Extensions (This Fork)

### Custom Components Pattern
All custom files follow `custom_*.py` naming and extend base classes:

| Component | Location | Base Class |
|-----------|----------|------------|
| Datasets | `mmpose/datasets/datasets/body3d/custom_*.py` | `BaseDataset` |
| Models | `mmpose/models/pose_estimators/custom_*.py` | `TopdownPoseEstimator` |
| Heads | `mmpose/models/heads/heatmap_heads/custom_*.py` | `HeatmapHead` |
| Codecs | `mmpose/codecs/custom_*.py` | `BaseKeypointCodec` |
| Metrics | `mmpose/evaluation/metrics/custom_*.py` | `BaseMetric` |
| Losses | `mmpose/models/losses/*.py` | - |
| Visualizers | `mmpose/visualization/custom_*.py` | `PoseLocalVisualizer` |

### Key Custom Features
- **Dual Backbone**: `Custom_TopdownPoseEstimator` accepts `backbone` + `backbone2`
- **HMD Info**: Custom heads process HMD (head/hand) 9-dim position data
- **Confidence-Weighted Fusion**: `ConfidenceWeightedHMDHead` - dynamically adjusts HMD/visual weights using heatmap confidence
- **H5 Caching**: `H5CachedEgoposeDataset` for fast loading (replaces JSON parsing)
  - Annotation-only cache: fast metadata loading
  - **Image-embedded cache**: includes 256x256 images (resolves NTFS bottleneck)
- **Seg+Depth**: `*_seg_depth` variants predict segmentation and depth maps
- **3D Visualizer**: `CustomPose3dLocalVisualizer_xregopose_v2` for simplified 3D pose visualization

### HMD Info Structure (9-dim)
```python
hmd_info = [
    right_hand_local,   # (3,) right hand local coordinates (relative to head)
    left_hand_local,    # (3,) left hand local coordinates
    hand_distance,      # (1,) distance between both hands
    right_distance,     # (1,) head-to-right-hand distance
    left_distance       # (1,) head-to-left-hand distance
]
```

### Custom Configs
Located in `my_code/custom_config/` (see `my_code/custom_config/README.md` for details):

| Config | Head Type | Features | Results |
|--------|-----------|----------|---------|
| `HMD_xregopose_single_coco_full_config.py` | `CustomxRegoposeBaselinel1` | **Single COCO baseline** | **41.37mm 🏆** |
| `HMD_xregopose_h5cache_coco_mpii_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | Dual COCO+MPII | 43.26mm |
| `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | `CustomxRegoposeBaselinel1_multi_backbone_v2` | Dual + Progressive Warmup | 45.93mm |
| `HMD_xregopose_single_lifting_config.py` | `CustomEgoposeLiftingHead` | Soft-argmax 2D→3D Lifting | 45.92mm |
| `HMD_xregopose_lifting_backbone_fusion_config.py` | `CustomEgoposeLiftingBackboneFusionHead` | Lifting + Backbone Fusion | 105.18mm ❌ |
| `HMD_xregopose_spatial_lifting_full_config.py` | `CustomEgoposeSpatialLiftingHead` | Grid Sampling Spatial Depth | pending experiment |
| `HMD_xregopose_efficient_decoder_full_config.py` | `CustomxRegoposeBaselinel1` | EfficientHeatmapDecoder (40M→1.35M) | 45.06mm |
| `HMD_xregopose_attention_lifting_full_config.py` | `CustomEgoposeAttentionLiftingHead` | Attention Lifting (Cross-Attn) | 45.43mm |
| `HMD_xregopose_vit_lifting_full_config.py` | `CustomEgoposeViTLiftingHead` | ViT-Style Lifting v1-v3 | 45.34mm |
| `HMD_xregopose_decoupled_full_config.py` | `CustomEgoposeDecoupledHead` | **Upper-Lower Decoupled (new)** | pending experiment |

### Experiment Results Summary (2026-01-27)

| Model | Full Body MPJPE | Upper Body | Lower Body | vs Baseline |
|-------|-----------------|------------|------------|-------------|
| **Single COCO (Baseline)** | **41.37mm** 🏆 | 29.42mm | 53.31mm | - |
| ViT Lifting v3 | 45.34mm | **23.49mm** ⭐ | 67.19mm | +3.97mm |
| Attention Lifting | 45.43mm | - | - | +4.06mm |
| EfficientHeatmapDecoder | 45.06mm | - | - | +3.69mm |
| Dual COCO+MPII | 43.26mm | - | - | +1.89mm |
| Single Lifting | 45.92mm | - | - | +4.55mm |

**Key finding**: ViT v3 achieves Upper Body 23.49mm (best!), but Lower Body 67.19mm (worst)
→ **Upper-Lower Decoupled** model attempting to combine the strengths of both

**Goal**: achieve below 41mm

**Next experiment**: Upper-Lower Decoupled Head (ViT v3 Upper + Baseline Lower)
- Expected results: ~38.40mm (combining Upper 23.49mm + Lower 53.31mm)

### Document Structure

| File | Purpose |
|------|---------|
| `EXPERIMENT_RESULTS.md` | **Experiment results and analysis** (per-epoch details, comprehensive analysis) |
| `DUAL_BACKBONE_EXPERIMENT_PLAN.md` | Experiment plan and per-phase implementation status |
| `IMPROVEMENT_IDEAS.md` | Improvement ideas (EfficientDecoder, Attention Lifting, etc.) |
| `SPATIAL_DEPTH_EXTRACTION_IDEAS.md` | Spatial information preserving depth extraction methods (Grid Sampling, etc.) |
| `VIT_STYLE_LIFTING_IDEA.md` | ViT-Style Lifting ideas (Learnable Queries, Self-Attention) |
| `NEXT_MODEL_IDEAS.md` | **Next model ideas** (Upper-Lower Decoupled and 4 other options) |
| `PROBLEM.md` | Code modification requirements (Metric squeeze bug, etc.) |
| `README.md` | Config descriptions and Smoke Test rules |

## Key Files Reference

| Purpose | File |
|---------|------|
| Training entry | `tools/train.py` |
| Testing entry | `tools/test.py` |
| Registry definitions | `mmpose/registry.py` |
| Default runtime | `configs/_base_/default_runtime.py` |
| H5 cache builder (annotation) | `tools/dataset_converters/build_egopose_h5cache.py` |
| H5 cache builder (with images) | `tools/dataset_converters/build_egopose_h5cache_with_images.py` |
| **Experiment results** | `my_code/custom_config/EXPERIMENT_RESULTS.md` |
| **Experiment plan** | `my_code/custom_config/DUAL_BACKBONE_EXPERIMENT_PLAN.md` |
| **Improvement ideas** | `my_code/custom_config/IMPROVEMENT_IDEAS.md` |
| **Spatial Depth methods** | `my_code/custom_config/SPATIAL_DEPTH_EXTRACTION_IDEAS.md` |
| **Code modification notes** | `my_code/custom_config/PROBLEM.md` |
| Config README | `my_code/custom_config/README.md` |

### Head Files

| Head | File | Purpose |
|------|------|---------|
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | Single backbone baseline (🏆 41.37mm) |
| `CustomxRegoposeBaselinel1_multi_backbone` | `custom_egopose_baselinel1_head_multi_backbone.py` | Dual backbone |
| `CustomxRegoposeBaselinel1_multi_backbone_v2` | `custom_egopose_baselinel1_head_multi_backbone_v2.py` | Dual + Warmup |
| `CustomEgoposeLiftingHead` | `custom_egopose_lifting_head.py` | Soft-argmax 2D→3D Lifting |
| `CustomEgoposeLiftingBackboneFusionHead` | `custom_egopose_lifting_backbone_fusion_head.py` | Lifting + Backbone Fusion |
| `CustomEgoposeSpatialLiftingHead` | `custom_egopose_spatial_lifting_head.py` | Grid Sampling based Spatial Depth |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting (Cross-Attn) |
| `CustomEgoposeViTLiftingHead` | `custom_egopose_vit_lifting_head.py` | ViT-Style Lifting v1-v3 |
| `CustomEgoposeDecoupledHead` | `custom_egopose_decoupled_head.py` | **Upper-Lower Decoupled (new)** |

### Smoke Test Rules

**Purpose**: quickly verify that the training pipeline of a new implementation runs correctly

**Config settings**:
- `max_epochs = 1~2` (short training)
- `checkpoint = None` (no weight saving)
- Use `_small` suffix in config filename

**Dataset paths**:
- Small: `train_small_1k.h5` / `val_small_500.h5`
- Full: `train_cache_with_images.h5` / `test_cache_with_images.h5`

### Key Structural Issues and Solutions

| Issue | Cause | Solution Direction |
|-------|-------|--------------------|
| HeatmapDecoder 40M params | `linear3: 2048→18432` | EfficientHeatmapDecoder (Conv-based, 1.35M) |
| GAP spatial information loss | Global Average Pooling | Per-joint feature extraction via Grid Sampling |
| Depth Ambiguity | Heatmap is a 2D probability distribution | Preserve depth cues via backbone features |
| Metric squeeze bug | `squeeze()` removes all dimensions | `squeeze(dim=1)` removes only specific dimension |

## Multi-Server Environment

This project runs simultaneously on the **3090 server** (code editing/push) and the **4090 server** (training/pull).

### Server Configuration

| Item | 3090 Server | 4090 Server |
|------|-------------|-------------|
| GPU | 2x RTX 3090 | 2x RTX 4090 |
| Role | Code editing, git push | Training execution, git pull |
| Dataset | H5 image-embedded cache | Annotation-only H5 + disk images |
| CLAUDE.md | Original (tracked by git) | `--assume-unchanged` (local only) |

### Mandatory Code Writing Rules

**`.view()` is prohibited → must use `.reshape()` instead**

`.view()` causes non-contiguous tensor RuntimeError in 4090 DDP distributed training.
It does not occur on 3090, but `.reshape()` must be used in all new code for 4090 compatibility.

```python
# BAD - RuntimeError in 4090 DDP
hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)
pose_3d = pose_3d.view(-1, 16, 3)

# GOOD - safe on both servers
hmd_tokens = self.hmd_embed(hmd_info).reshape(B, 3, -1)
pose_3d = pose_3d.reshape(-1, 16, 3)
```

### 4090 Data Pipeline Differences

SIGSEGV occurs on 4090 when combining H5 embedded images + DDP + multiprocessing.
Workaround uses annotation-only H5 + preprocessed disk images.

| Item | 3090 (H5 image cache) | 4090 (disk images) |
|------|----------------------|---------------------|
| Pipeline | `LoadImageFromH5Cache` | `LoadImage` |
| `use_cached_images` | `True` | `False` |
| `img_path_replace` | None | `{'/Dataset/': '/Dataset_256/'}` |
| VisualizationHook | `H5CacheVisualizationHook` | `PoseVisualizationHook` |

### Config Writing Guide (Cross-Server Compatible)

When writing a new config, adding per-server branching allows the same config to be used on both servers:
```python
import platform, socket

_hostname = socket.gethostname()
_is_4090_server = (_hostname == '4090_server_hostname')  # Replace with actual hostname

if _is_4090_server:
    _pipeline_load = dict(type='LoadImage')
    _use_cached_images = False
    _img_path_replace = {'/Dataset/': '/Dataset_256/'}
    _vis_hook_type = 'PoseVisualizationHook'
else:
    _pipeline_load = dict(type='LoadImageFromH5Cache')
    _use_cached_images = True
    _img_path_replace = None
    _vis_hook_type = 'H5CacheVisualizationHook'
```

### 4090 Local-Only Files (not tracked by git)

| File | Purpose |
|------|---------|
| `4090_TRAINING_GUIDE.md` | 4090 training guide |
| `ENVIRONMENT_SETUP_TROUBLESHOOTING.md` | Environment setup troubleshooting |
| `tools/dataset_converters/preprocess_egopose_images.py` | Image preprocessing script |
| `my_code/custom_config/HMD_xregopose_dist_test_config.py` | Distributed training test config |

### 4090 Locally Modified Files (not tracked by git)

| File | Modification Details |
|------|----------------------|
| `custom_egopose_dataset.py` | Windows hardcoded paths → relative paths |
| `custom_egopose_dataset_h5cache.py` | Added `img_path_replace` parameter |
| `mmpose/datasets/transforms/__init__.py` | Added `LoadImageFromH5Cache` import |
| `HMD_xregopose_vit_lifting_full_config.py` | 4090 data paths/pipeline |
| `tools/train.py` | Unconfirmed modifications |

### CLAUDE.md Management

- **Original editing**: performed only on the 3090 server (tracked by git)
- **4090**: `git update-index --assume-unchanged CLAUDE.md` is configured
- Local server environment info is added at the top of 4090's CLAUDE.md
- When syncing the latest CLAUDE.md on 4090:
  ```bash
  git update-index --no-assume-unchanged CLAUDE.md
  git pull
  git update-index --assume-unchanged CLAUDE.md
  ```

## External Documentation

### MMEngine (Core Framework)
- **Documentation**: https://mmengine.readthedocs.io/en/latest/index.html
- Key topics:
  - [Registry](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html) - Component registration system
  - [Config](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) - Configuration file system
  - [Runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html) - Training/testing loop
  - [Hook](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html) - Training hooks
  - [Data Transform](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_transform.html) - Data pipeline transforms
  - [Visualization](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html) - Visualization backends

### MMPose (Pose Estimation)
- **Documentation**: https://mmpose.readthedocs.io/en/latest/overview.html
- Key topics:
  - [User Guides](https://mmpose.readthedocs.io/en/latest/user_guides/index.html) - Training, testing, inference
  - [Codecs](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html#step3-codec) - Keypoint encoding/decoding
  - [Model Zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html) - Pretrained models
  - [Dataset Preparation](https://mmpose.readthedocs.io/en/latest/dataset_zoo/index.html) - Dataset formats
  - [Custom Dataset](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html) - Adding new datasets
  - [Custom Model](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_models.html) - Adding new models

### MMDeploy (Model Deployment)
- **Documentation**: https://mmdeploy.readthedocs.io/en/latest/
- **MMPose Deployment**: https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpose.html
- Key topics:
  - [Installation](https://mmdeploy.readthedocs.io/en/latest/get_started.html) - Setup guide
  - [Model Conversion](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html) - PyTorch → Backend conversion
  - [Supported Backends](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/index.html) - ONNX Runtime, TensorRT, OpenVINO, etc.
  - [SDK Integration](https://mmdeploy.readthedocs.io/en/latest/03-benchmark/how_to_evaluate_a_model.html) - C++/Python inference SDK
- Supported backends: ONNX Runtime, TensorRT, OpenVINO, NCNN, TVM, CoreML
- Deployment config naming: `pose-detection_{backend}-{precision}_{static|dynamic}_{shape}.py`

## Model Deployment (Custom)

### Convert Custom EgoPose Model
```bash
# Using custom conversion script
python my_code/deploy/convert_egopose_model.py \
    --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
    --checkpoint work_dirs/HMD_xregopose_h5cache/best_*.pth \
    --output-dir deploy_models/egopose_onnx \
    --backend onnxruntime
```

### Run Inference with Deployed Model
```bash
python my_code/deploy/inference_deployed.py \
    --model-dir deploy_models/egopose_onnx \
    --image path/to/image.jpg \
    --output-dir output_deploy
```
