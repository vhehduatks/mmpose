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
- **Confidence-Weighted Fusion**: `ConfidenceWeightedHMDHead` - heatmap confidence로 HMD/visual 가중치 동적 조절
- **H5 Caching**: `H5CachedEgoposeDataset` for fast loading (replaces JSON parsing)
  - Annotation-only cache: 빠른 메타데이터 로딩
  - **Image-embedded cache**: 256x256 이미지 포함 (NTFS 병목 해결)
- **Seg+Depth**: `*_seg_depth` variants predict segmentation and depth maps
- **3D Visualizer**: `CustomPose3dLocalVisualizer_xregopose_v2` for simplified 3D pose visualization

### HMD Info Structure (9-dim)
```python
hmd_info = [
    right_hand_local,   # (3,) 오른손 로컬 좌표 (머리 기준)
    left_hand_local,    # (3,) 왼손 로컬 좌표
    hand_distance,      # (1,) 양손 간 거리
    right_distance,     # (1,) 머리-오른손 거리
    left_distance       # (1,) 머리-왼손 거리
]
```

### Custom Configs
Located in `my_code/custom_config/` (see `my_code/custom_config/README.md` for details):

| Config | Head Type | 특징 | 결과 |
|--------|-----------|------|------|
| `HMD_xregopose_single_coco_full_config.py` | `CustomxRegoposeBaselinel1` | **Single COCO baseline** | **41.37mm 🏆** |
| `HMD_xregopose_h5cache_coco_mpii_config.py` | `CustomxRegoposeBaselinel1_multi_backbone` | Dual COCO+MPII | 43.26mm |
| `HMD_xregopose_h5cache_coco_mpii_warmup_10ep_config.py` | `CustomxRegoposeBaselinel1_multi_backbone_v2` | Dual + Progressive Warmup | 45.93mm |
| `HMD_xregopose_single_lifting_config.py` | `CustomEgoposeLiftingHead` | Soft-argmax 2D→3D Lifting | 45.92mm |
| `HMD_xregopose_lifting_backbone_fusion_config.py` | `CustomEgoposeLiftingBackboneFusionHead` | **Lifting + Backbone Fusion** | 실험 대기 |

### 실험 결과 요약 (2026-01-21)

| Model | Full Body MPJPE | vs Baseline |
|-------|-----------------|-------------|
| **Single COCO (Baseline)** | **41.37mm** 🏆 | - |
| Dual COCO+MPII | 43.26mm | +1.89mm ❌ |
| Dual Warmup v2 | 45.93mm | +4.56mm ❌ |
| Single Lifting | 45.92mm | +4.55mm ❌ |

**목표**: 41mm 이하 달성

**다음 실험**: Lifting + Backbone Fusion (역할 분리: 2D coords → gradient 차단, backbone → depth cues)

### 문서 구조

| 파일 | 용도 |
|------|------|
| `EXPERIMENT_RESULTS.md` | **실험 결과 및 분석** (epoch별 상세, 종합 분석) |
| `DUAL_BACKBONE_EXPERIMENT_PLAN.md` | 실험 계획 및 Phase별 구현 상태 |
| `DUAL_BACKBONE_IMPROVEMENT_IDEAS.md` | 개선 아이디어 (Attention Lifting 등) |
| `README.md` | Config 설명 |

## Key Files Reference

| Purpose | File |
|---------|------|
| Training entry | `tools/train.py` |
| Testing entry | `tools/test.py` |
| Registry definitions | `mmpose/registry.py` |
| Default runtime | `configs/_base_/default_runtime.py` |
| H5 cache builder (annotation) | `tools/dataset_converters/build_egopose_h5cache.py` |
| H5 cache builder (with images) | `tools/dataset_converters/build_egopose_h5cache_with_images.py` |
| **실험 결과** | `my_code/custom_config/EXPERIMENT_RESULTS.md` |
| **실험 계획** | `my_code/custom_config/DUAL_BACKBONE_EXPERIMENT_PLAN.md` |
| **개선 아이디어** | `my_code/custom_config/DUAL_BACKBONE_IMPROVEMENT_IDEAS.md` |
| Config README | `my_code/custom_config/README.md` |

### Head 파일

| Head | File | 용도 |
|------|------|------|
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | Single backbone baseline (🏆 41.37mm) |
| `CustomxRegoposeBaselinel1_multi_backbone` | `custom_egopose_baselinel1_head_multi_backbone.py` | Dual backbone |
| `CustomxRegoposeBaselinel1_multi_backbone_v2` | `custom_egopose_baselinel1_head_multi_backbone_v2.py` | Dual + Warmup |
| `CustomEgoposeLiftingHead` | `custom_egopose_lifting_head.py` | Soft-argmax 2D→3D Lifting |
| `CustomEgoposeLiftingBackboneFusionHead` | `custom_egopose_lifting_backbone_fusion_head.py` | **Lifting + Backbone Fusion (신규)** |

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
