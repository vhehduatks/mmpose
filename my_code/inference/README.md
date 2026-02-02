# Inference Scripts

This folder contains scripts for running inference with trained XR EgoPose models.

## Scripts

### inference_egopose.py

Main inference script for XR EgoPose models. Supports single image inference and batch inference from dataset.

**Usage:**

```bash
# Single image inference
python my_code/inference/inference_egopose.py \
    --config my_code/custom_config/HMD_xregopose_single_coco_full_config.py \
    --checkpoint work_dirs/HMD_xregopose_single_coco_full/best_*.pth \
    --input path/to/image.png \
    --output output_inference/

# Inference on dataset samples (with proper HMD info)
python my_code/inference/inference_egopose.py \
    --config my_code/custom_config/HMD_xregopose_single_coco_full_config.py \
    --checkpoint work_dirs/HMD_xregopose_single_coco_full/best_*.pth \
    --dataset-root /path/to/dataset \
    --num-samples 10 \
    --output output_inference/

# Inference on folder of images
python my_code/inference/inference_egopose.py \
    --config my_code/custom_config/HMD_xregopose_single_coco_full_config.py \
    --checkpoint work_dirs/HMD_xregopose_single_coco_full/best_*.pth \
    --input /path/to/image_folder/ \
    --num-samples 20 \
    --output output_inference/
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config file | Required |
| `--checkpoint` | Path to checkpoint file | Required |
| `--input` | Input image path or folder | None |
| `--dataset-root` | Dataset root for batch inference | None |
| `--num-samples` | Number of samples to process | 5 |
| `--output` | Output directory | `output_inference` |
| `--device` | Device (cuda:0, cpu) | `cuda:0` |
| `--show` | Show visualization window | False |

**Output:**

- Visualization images with predicted 2D/3D poses
- Comparison with ground truth (when available)

## Related: Deploy Scripts

For model deployment (ONNX export, TensorRT), see `my_code/deploy/`:

- `convert_egopose_model.py` - Convert model to ONNX
- `inference_deployed.py` - Run inference with deployed model
- `inference_egopose_fc.py` - Inference with FC (fully connected) exported model
- `inference_full3d.py` - Full 3D inference pipeline
