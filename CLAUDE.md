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
# Build H5 cache for fast dataset loading (65k samples: 7min → 1sec)
python tools/dataset_converters/build_egopose_h5cache.py \
    --data-root F:/ego_cam_dataset/Train --num-workers 4
```

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
- **HMD Info**: Custom heads process HMD (head/hand) position data
- **H5 Caching**: `H5CachedEgoposeDataset` for fast loading (replaces JSON parsing)
- **Seg+Depth**: `*_seg_depth` variants predict segmentation and depth maps
- **3D Visualizer**: `CustomPose3dLocalVisualizer_xregopose_v2` for simplified 3D pose visualization

### Custom Configs
Located in `my_code/custom_config/`:
- `HMD_xregopose_h5cache_config.py` - Fast-loading H5 cached dataset
- `HMD_xregopose_imple_config_2backbone.py` - Dual backbone training
- `Segdepth_xregopose_imple_config_baseline.py` - With seg/depth prediction

## Key Files Reference

| Purpose | File |
|---------|------|
| Training entry | `tools/train.py` |
| Testing entry | `tools/test.py` |
| Registry definitions | `mmpose/registry.py` |
| Default runtime | `configs/_base_/default_runtime.py` |
| H5 cache builder | `tools/dataset_converters/build_egopose_h5cache.py` |

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
