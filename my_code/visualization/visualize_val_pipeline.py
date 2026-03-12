"""
Visualize Mo2Cap2 Validation Pipeline Stages

Shows how images transform through each stage:
1. Raw image (LoadImage)
2. After CenterCrop
3. After TopdownAffine (256x256)
4. After normalize (data_preprocessor)
5. After denormalize (visualization hook)

Usage:
    python my_code/visualization/visualize_val_pipeline.py --index 0 --output output_pipeline_vis
    python my_code/visualization/visualize_val_pipeline.py --index 2744 --output output_pipeline_vis  # weipeng_studio
    python my_code/visualization/visualize_val_pipeline.py --img-path /path/to/image.jpg --output output_pipeline_vis
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mmengine.registry import init_default_scope


def visualize_pipeline_stages(
    img_path: Optional[str] = None,
    dataset_index: Optional[int] = None,
    output_dir: str = 'output_pipeline_vis',
    config_path: str = 'my_code/custom_config/HMD_mo2cap2_cascaded_both_from_ground_config.py',
    show_stats: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Visualize how an image transforms through the validation pipeline.

    Args:
        img_path: Path to a specific image file
        dataset_index: Index in the combined test dataset (0-5645)
        output_dir: Directory to save visualization outputs
        config_path: Path to config file for pipeline definition
        show_stats: Whether to print statistics for each stage

    Returns:
        Dictionary of stage_name -> image array
    """
    init_default_scope('mmpose')

    from mmpose.registry import DATASETS, TRANSFORMS
    from mmengine.config import Config

    os.makedirs(output_dir, exist_ok=True)

    # Test data paths
    test_root = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet'

    # Determine image path
    if img_path is None and dataset_index is not None:
        # Load from dataset
        sequences = ['olek_outdoor', 'weipeng_studio']
        seq_sizes = [2744, 2902]

        if dataset_index < seq_sizes[0]:
            seq = sequences[0]
            local_idx = dataset_index
        else:
            seq = sequences[1]
            local_idx = dataset_index - seq_sizes[0]

        seq_dir = os.path.join(test_root, seq)
        img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])
        img_path = os.path.join(seq_dir, img_files[local_idx])

        print(f"Dataset index {dataset_index} -> {seq}[{local_idx}]")

    if img_path is None:
        raise ValueError("Either img_path or dataset_index must be provided")

    print(f"Image path: {img_path}")

    # Store all stages
    stages = {}

    # Stage 1: Raw image (LoadImage)
    print("\n" + "="*60)
    print("Stage 1: Raw Image (LoadImage)")
    print("="*60)

    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    stages['1_raw'] = raw_img.copy()
    _print_stats(raw_img, "Raw", show_stats)

    # Stage 2: After CenterCrop
    print("\n" + "="*60)
    print("Stage 2: After Mo2Cap2CenterCrop (128px margins)")
    print("="*60)

    from mmpose.datasets.transforms.center_crop import Mo2Cap2CenterCrop
    center_crop = Mo2Cap2CenterCrop(margin_left=128, margin_right=128)

    results = {'img': raw_img.copy(), 'img_shape': raw_img.shape[:2], 'ori_shape': raw_img.shape[:2]}
    results = center_crop.transform(results)
    cropped_img = results['img']

    stages['2_centercrop'] = cropped_img.copy()
    _print_stats(cropped_img, "CenterCrop", show_stats)

    # Stage 3: After TopdownAffine
    print("\n" + "="*60)
    print("Stage 3: After TopdownAffine (256x256)")
    print("="*60)

    from mmpose.datasets.transforms.common_transforms import GetBBoxCenterScale
    from mmpose.datasets.transforms.topdown_transforms import TopdownAffine

    # Add required fields for GetBBoxCenterScale
    results['bbox'] = np.array([[0, 0, cropped_img.shape[1], cropped_img.shape[0]]], dtype=np.float32)
    results['bbox_score'] = np.ones(1, dtype=np.float32)

    get_bbox = GetBBoxCenterScale(padding=1.0)
    results = get_bbox.transform(results)

    affine = TopdownAffine(input_size=(256, 256))
    results = affine.transform(results)
    affine_img = results['img']

    stages['3_affine'] = affine_img.copy()
    _print_stats(affine_img, "TopdownAffine", show_stats)

    # Stage 4: After normalize (data_preprocessor simulation)
    print("\n" + "="*60)
    print("Stage 4: After Normalize (data_preprocessor)")
    print("="*60)

    # BGR to RGB, then normalize
    img_rgb = affine_img[:, :, ::-1].copy()
    img_float = img_rgb.astype(np.float32)

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_normalized = (img_float - mean) / std

    # For visualization, scale to 0-255
    img_norm_vis = ((img_normalized - img_normalized.min()) /
                    (img_normalized.max() - img_normalized.min() + 1e-8) * 255).astype(np.uint8)
    img_norm_vis = img_norm_vis[:, :, ::-1]  # RGB to BGR for saving

    stages['4_normalized'] = img_norm_vis.copy()
    _print_stats(img_normalized, "Normalized (float)", show_stats, is_float=True)

    # Stage 5: After denormalize (visualization hook simulation)
    print("\n" + "="*60)
    print("Stage 5: After Denormalize (visualization hook)")
    print("="*60)

    img_denorm = img_normalized * std + mean
    img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
    img_denorm_bgr = img_denorm[:, :, ::-1].copy()  # RGB to BGR

    stages['5_denormalized'] = img_denorm_bgr.copy()
    _print_stats(img_denorm_bgr, "Denormalized", show_stats)

    # Stage 6: Fallback comparison (what happens if fallback triggers)
    print("\n" + "="*60)
    print("Stage 6: Fallback (direct resize without CenterCrop)")
    print("="*60)

    fallback_img = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    stages['6_fallback'] = fallback_img.copy()
    _print_stats(fallback_img, "Fallback", show_stats)

    # Calculate difference between proper processing and fallback
    diff = np.abs(img_denorm_bgr.astype(float) - fallback_img.astype(float))
    print(f"  Difference (denorm vs fallback): Mean={diff.mean():.2f}, Max={diff.max():.2f}")

    # Create combined visualization
    print("\n" + "="*60)
    print("Creating combined visualization...")
    print("="*60)

    combined = _create_combined_visualization(stages, img_path)

    # Save outputs
    basename = os.path.splitext(os.path.basename(img_path))[0]

    # Save individual stages
    for stage_name, img in stages.items():
        out_path = os.path.join(output_dir, f'{basename}_{stage_name}.jpg')
        cv2.imwrite(out_path, img)

    # Save combined
    combined_path = os.path.join(output_dir, f'{basename}_combined.jpg')
    cv2.imwrite(combined_path, combined)

    print(f"\nSaved outputs to: {output_dir}")
    print(f"Combined visualization: {combined_path}")

    return stages


def _print_stats(img: np.ndarray, stage_name: str, show: bool, is_float: bool = False):
    """Print image statistics."""
    if not show:
        return

    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Min/Max: {img.min():.2f} / {img.max():.2f}")
    print(f"  Mean: {img.mean():.2f}")

    if not is_float and len(img.shape) == 3 and img.shape[2] == 3:
        print(f"  Channel means (BGR): B={img[:,:,0].mean():.2f}, G={img[:,:,1].mean():.2f}, R={img[:,:,2].mean():.2f}")


def _create_combined_visualization(stages: Dict[str, np.ndarray], img_path: str) -> np.ndarray:
    """Create a combined visualization with all stages."""

    # Target size for each panel
    panel_size = (256, 256)

    # Resize all images to panel size
    panels = []
    labels = [
        '1. Raw',
        '2. CenterCrop',
        '3. Affine 256x256',
        '4. Normalized',
        '5. Denormalized',
        '6. Fallback'
    ]

    for (stage_name, img), label in zip(stages.items(), labels):
        # Resize if needed
        if img.shape[:2] != panel_size:
            panel = cv2.resize(img, panel_size, interpolation=cv2.INTER_LINEAR)
        else:
            panel = img.copy()

        # Add label
        cv2.putText(panel, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add shape info
        h, w = img.shape[:2]
        cv2.putText(panel, f'{w}x{h}', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Add mean value
        cv2.putText(panel, f'mean={img.mean():.1f}', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        panels.append(panel)

    # Arrange in 2 rows x 3 columns
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])
    combined = np.vstack([row1, row2])

    # Add title
    title_bar = np.ones((40, combined.shape[1], 3), dtype=np.uint8) * 50
    title = f"Val Pipeline: {os.path.basename(img_path)}"
    cv2.putText(title_bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.vstack([title_bar, combined])

    return combined


def visualize_batch(
    start_index: int = 0,
    count: int = 5,
    output_dir: str = 'output_pipeline_vis',
    include_both_sequences: bool = True,
):
    """
    Visualize multiple images from the dataset.

    Args:
        start_index: Starting dataset index
        count: Number of images to visualize
        output_dir: Output directory
        include_both_sequences: If True, also visualize from weipeng_studio
    """
    indices = list(range(start_index, start_index + count))

    if include_both_sequences:
        # Add samples from weipeng_studio (starts at 2744)
        weipeng_start = 2744
        indices.extend(range(weipeng_start, weipeng_start + count))

    for idx in indices:
        try:
            print(f"\n{'#'*60}")
            print(f"# Processing index {idx}")
            print(f"{'#'*60}")
            visualize_pipeline_stages(dataset_index=idx, output_dir=output_dir)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Mo2Cap2 validation pipeline stages')
    parser.add_argument('--index', type=int, default=None, help='Dataset index (0-5645)')
    parser.add_argument('--img-path', type=str, default=None, help='Path to specific image')
    parser.add_argument('--output', type=str, default='output_pipeline_vis', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Visualize batch of images')
    parser.add_argument('--batch-start', type=int, default=0, help='Batch start index')
    parser.add_argument('--batch-count', type=int, default=3, help='Number of images per sequence')

    args = parser.parse_args()

    if args.batch:
        visualize_batch(
            start_index=args.batch_start,
            count=args.batch_count,
            output_dir=args.output,
        )
    elif args.index is not None:
        visualize_pipeline_stages(dataset_index=args.index, output_dir=args.output)
    elif args.img_path is not None:
        visualize_pipeline_stages(img_path=args.img_path, output_dir=args.output)
    else:
        # Default: visualize first image from each sequence
        print("No arguments provided. Visualizing sample from each sequence...")
        visualize_pipeline_stages(dataset_index=0, output_dir=args.output)  # olek_outdoor
        visualize_pipeline_stages(dataset_index=2744, output_dir=args.output)  # weipeng_studio
