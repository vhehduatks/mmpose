"""
EgoPose Model Deployment Module

This module provides tools for converting and deploying custom EgoPose models
to various inference backends (ONNX Runtime, TensorRT, OpenVINO).

Reference:
    - MMDeploy: https://mmdeploy.readthedocs.io/en/latest/
    - MMPose Deployment: https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmpose.html

Usage:
    # Convert model
    python my_code/deploy/convert_egopose_model.py \
        --config my_code/custom_config/HMD_xregopose_h5cache_config.py \
        --checkpoint work_dirs/HMD_xregopose_h5cache/best_*.pth \
        --output-dir deploy_models/egopose_onnx

    # Run inference
    python my_code/deploy/inference_deployed.py \
        --model-dir deploy_models/egopose_onnx \
        --image path/to/image.jpg
"""

from .convert_egopose_model import (
    export_simplified_model,
    convert_to_onnx,
    convert_to_tensorrt,
)

__all__ = [
    'export_simplified_model',
    'convert_to_onnx',
    'convert_to_tensorrt',
]
