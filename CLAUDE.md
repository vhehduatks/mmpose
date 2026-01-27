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
| `HMD_xregopose_lifting_backbone_fusion_config.py` | `CustomEgoposeLiftingBackboneFusionHead` | Lifting + Backbone Fusion | 105.18mm ❌ |
| `HMD_xregopose_spatial_lifting_full_config.py` | `CustomEgoposeSpatialLiftingHead` | Grid Sampling Spatial Depth | 실험 대기 |
| `HMD_xregopose_efficient_decoder_full_config.py` | `CustomxRegoposeBaselinel1` | EfficientHeatmapDecoder (40M→1.35M) | 45.06mm |
| `HMD_xregopose_attention_lifting_full_config.py` | `CustomEgoposeAttentionLiftingHead` | Attention Lifting (Cross-Attn) | 45.43mm |
| `HMD_xregopose_vit_lifting_full_config.py` | `CustomEgoposeViTLiftingHead` | ViT-Style Lifting v1-v3 | 45.34mm |
| `HMD_xregopose_decoupled_full_config.py` | `CustomEgoposeDecoupledHead` | **Upper-Lower Decoupled (신규)** | 실험 대기 |

### 실험 결과 요약 (2026-01-27)

| Model | Full Body MPJPE | Upper Body | Lower Body | vs Baseline |
|-------|-----------------|------------|------------|-------------|
| **Single COCO (Baseline)** | **41.37mm** 🏆 | 29.42mm | 53.31mm | - |
| ViT Lifting v3 | 45.34mm | **23.49mm** ⭐ | 67.19mm | +3.97mm |
| Attention Lifting | 45.43mm | - | - | +4.06mm |
| EfficientHeatmapDecoder | 45.06mm | - | - | +3.69mm |
| Dual COCO+MPII | 43.26mm | - | - | +1.89mm |
| Single Lifting | 45.92mm | - | - | +4.55mm |

**핵심 발견**: ViT v3는 Upper Body 23.49mm (최고!), Lower Body 67.19mm (최악)
→ **Upper-Lower Decoupled** 모델로 두 장점 결합 시도 중

**목표**: 41mm 이하 달성

**다음 실험**: Upper-Lower Decoupled Head (ViT v3 Upper + Baseline Lower)
- 예상 결과: ~38.40mm (Upper 23.49mm + Lower 53.31mm 결합)

### 문서 구조

| 파일 | 용도 |
|------|------|
| `EXPERIMENT_RESULTS.md` | **실험 결과 및 분석** (epoch별 상세, 종합 분석) |
| `DUAL_BACKBONE_EXPERIMENT_PLAN.md` | 실험 계획 및 Phase별 구현 상태 |
| `IMPROVEMENT_IDEAS.md` | 개선 아이디어 (EfficientDecoder, Attention Lifting 등) |
| `SPATIAL_DEPTH_EXTRACTION_IDEAS.md` | 공간 정보 보존 Depth 추출 방법 (Grid Sampling 등) |
| `VIT_STYLE_LIFTING_IDEA.md` | ViT-Style Lifting 아이디어 (Learnable Queries, Self-Attention) |
| `NEXT_MODEL_IDEAS.md` | **다음 모델 아이디어** (Upper-Lower Decoupled 등 4가지 옵션) |
| `PROBLEM.md` | 코드 수정 필요 사항 (Metric squeeze 버그 등) |
| `README.md` | Config 설명 및 Smoke Test 규칙 |

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
| **개선 아이디어** | `my_code/custom_config/IMPROVEMENT_IDEAS.md` |
| **Spatial Depth 방법** | `my_code/custom_config/SPATIAL_DEPTH_EXTRACTION_IDEAS.md` |
| **코드 수정 사항** | `my_code/custom_config/PROBLEM.md` |
| Config README | `my_code/custom_config/README.md` |

### Head 파일

| Head | File | 용도 |
|------|------|------|
| `CustomxRegoposeBaselinel1` | `custom_egopose_baselinel1_head.py` | Single backbone baseline (🏆 41.37mm) |
| `CustomxRegoposeBaselinel1_multi_backbone` | `custom_egopose_baselinel1_head_multi_backbone.py` | Dual backbone |
| `CustomxRegoposeBaselinel1_multi_backbone_v2` | `custom_egopose_baselinel1_head_multi_backbone_v2.py` | Dual + Warmup |
| `CustomEgoposeLiftingHead` | `custom_egopose_lifting_head.py` | Soft-argmax 2D→3D Lifting |
| `CustomEgoposeLiftingBackboneFusionHead` | `custom_egopose_lifting_backbone_fusion_head.py` | Lifting + Backbone Fusion |
| `CustomEgoposeSpatialLiftingHead` | `custom_egopose_spatial_lifting_head.py` | Grid Sampling 기반 Spatial Depth |
| `CustomEgoposeAttentionLiftingHead` | `custom_egopose_attention_lifting_head.py` | Attention Lifting (Cross-Attn) |
| `CustomEgoposeViTLiftingHead` | `custom_egopose_vit_lifting_head.py` | ViT-Style Lifting v1-v3 |
| `CustomEgoposeDecoupledHead` | `custom_egopose_decoupled_head.py` | **Upper-Lower Decoupled (신규)** |

### Smoke Test 규칙

**목적**: 새 구현체의 훈련 파이프라인이 정상 동작하는지 빠르게 확인

**Config 설정**:
- `max_epochs = 1~2` (짧은 훈련)
- `checkpoint = None` (가중치 저장 안함)
- Config 파일명에 `_small` suffix 사용

**Dataset 경로**:
- Small: `train_small_1k.h5` / `val_small_500.h5`
- Full: `train_cache_with_images.h5` / `test_cache_with_images.h5`

### 주요 구조적 문제 및 해결 방향

| 문제 | 원인 | 해결 방향 |
|------|------|----------|
| HeatmapDecoder 40M params | `linear3: 2048→18432` | EfficientHeatmapDecoder (Conv 기반, 1.35M) |
| GAP 공간 정보 손실 | Global Average Pooling | Grid Sampling으로 관절별 feature 추출 |
| Depth Ambiguity | Heatmap은 2D 확률 분포 | Backbone feature로 depth cues 보존 |
| Metric squeeze 버그 | `squeeze()` 전체 차원 제거 | `squeeze(dim=1)` 특정 차원만 제거 |

## Multi-Server Environment

이 프로젝트는 **3090 서버**(코드 수정/push)와 **4090 서버**(훈련/pull)에서 동시 운영 중.

### 서버 구성

| 항목 | 3090 서버 | 4090 서버 |
|------|-----------|-----------|
| GPU | 2x RTX 3090 | 2x RTX 4090 |
| 역할 | 코드 수정, git push | 훈련 실행, git pull |
| Dataset | H5 image-embedded cache | Annotation-only H5 + 디스크 이미지 |
| CLAUDE.md | 원본 (git 반영) | `--assume-unchanged` (로컬 전용) |

### 코드 작성 필수 규칙

**`.view()` 사용 금지 → 반드시 `.reshape()` 사용**

4090 DDP 분산훈련에서 `.view()`가 non-contiguous tensor RuntimeError 발생.
3090에서는 발생하지 않지만 4090 호환을 위해 모든 새 코드에서 `.reshape()` 사용 필수.

```python
# BAD - 4090 DDP에서 RuntimeError 발생
hmd_tokens = self.hmd_embed(hmd_info).view(B, 3, -1)
pose_3d = pose_3d.view(-1, 16, 3)

# GOOD - 양쪽 서버 모두 안전
hmd_tokens = self.hmd_embed(hmd_info).reshape(B, 3, -1)
pose_3d = pose_3d.reshape(-1, 16, 3)
```

### 4090 데이터 파이프라인 차이

4090에서 H5 embedded images + DDP + multiprocessing 조합 시 SIGSEGV 발생.
annotation-only H5 + 전처리된 디스크 이미지로 우회.

| 항목 | 3090 (H5 image cache) | 4090 (디스크 이미지) |
|------|----------------------|---------------------|
| Pipeline | `LoadImageFromH5Cache` | `LoadImage` |
| `use_cached_images` | `True` | `False` |
| `img_path_replace` | 없음 | `{'/Dataset/': '/Dataset_256/'}` |
| VisualizationHook | `H5CacheVisualizationHook` | `PoseVisualizationHook` |

### Config 작성 가이드 (Cross-Server 호환)

새 config 작성 시, 서버별 분기를 넣으면 양쪽에서 동일 config 사용 가능:
```python
import platform, socket

_hostname = socket.gethostname()
_is_4090_server = (_hostname == '4090서버호스트명')  # 실제 호스트명으로 교체

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

### 4090 로컬 전용 파일 (git 미반영)

| 파일 | 용도 |
|------|------|
| `4090_TRAINING_GUIDE.md` | 4090 훈련 가이드 |
| `ENVIRONMENT_SETUP_TROUBLESHOOTING.md` | 환경 설치 트러블슈팅 |
| `tools/dataset_converters/preprocess_egopose_images.py` | 이미지 전처리 스크립트 |
| `my_code/custom_config/HMD_xregopose_dist_test_config.py` | 분산훈련 테스트 config |

### 4090 로컬 수정 파일 (git 미반영)

| 파일 | 수정 내용 |
|------|----------|
| `custom_egopose_dataset.py` | Windows 경로 하드코딩 → 상대경로 |
| `custom_egopose_dataset_h5cache.py` | `img_path_replace` 파라미터 추가 |
| `mmpose/datasets/transforms/__init__.py` | `LoadImageFromH5Cache` import 추가 |
| `HMD_xregopose_vit_lifting_full_config.py` | 4090용 데이터 경로/파이프라인 |
| `tools/train.py` | 미확인 수정 |

### CLAUDE.md 관리

- **원본 수정**: 3090 서버에서만 수행 (git 반영)
- **4090**: `git update-index --assume-unchanged CLAUDE.md` 설정됨
- 4090 CLAUDE.md 상단에 로컬 서버 환경 정보 추가되어 있음
- 4090에서 최신 CLAUDE.md 동기화 필요 시:
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
