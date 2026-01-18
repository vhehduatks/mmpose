"""
Deployment Configurations for Custom EgoPose Model

This module provides deployment configs for various backends.
"""

# ONNX Runtime config
onnxruntime_config = dict(
    backend='onnxruntime',
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['image'],
        output_names=['keypoints_2d', 'keypoints_3d', 'heatmaps'],
        input_shape=[1, 3, 256, 256],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'keypoints_2d': {0: 'batch_size'},
            'keypoints_3d': {0: 'batch_size'},
            'heatmaps': {0: 'batch_size'},
        },
    ),
    codebase='mmpose',
    task='PoseDetection',
)

# TensorRT config (FP16)
tensorrt_fp16_config = dict(
    backend='tensorrt',
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['image'],
        output_names=['keypoints_2d', 'keypoints_3d', 'heatmaps'],
        input_shape=[1, 3, 256, 256],
    ),
    tensorrt_config=dict(
        fp16_mode=True,
        max_workspace_size=1 << 30,  # 1GB
    ),
    codebase='mmpose',
    task='PoseDetection',
)

# TensorRT config (INT8)
tensorrt_int8_config = dict(
    backend='tensorrt',
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['image'],
        output_names=['keypoints_2d', 'keypoints_3d', 'heatmaps'],
        input_shape=[1, 3, 256, 256],
    ),
    tensorrt_config=dict(
        fp16_mode=True,
        int8_mode=True,
        max_workspace_size=1 << 30,
    ),
    codebase='mmpose',
    task='PoseDetection',
)

# OpenVINO config
openvino_config = dict(
    backend='openvino',
    onnx_config=dict(
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['image'],
        output_names=['keypoints_2d', 'keypoints_3d', 'heatmaps'],
        input_shape=[1, 3, 256, 256],
    ),
    codebase='mmpose',
    task='PoseDetection',
)


def get_deploy_config(backend='onnxruntime', precision='fp32'):
    """Get deployment config for specified backend.

    Args:
        backend: 'onnxruntime', 'tensorrt', or 'openvino'
        precision: 'fp32', 'fp16', or 'int8'

    Returns:
        Deployment config dict
    """
    if backend == 'onnxruntime':
        return onnxruntime_config
    elif backend == 'tensorrt':
        if precision == 'int8':
            return tensorrt_int8_config
        else:
            return tensorrt_fp16_config
    elif backend == 'openvino':
        return openvino_config
    else:
        raise ValueError(f"Unknown backend: {backend}")
