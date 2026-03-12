"""
Visualize Mo2Cap2 Pipeline Processing

Utility functions to compare train/test images through the preprocessing pipeline.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from typing import List, Optional, Tuple


def build_transforms(use_circular_crop: bool = True, radius_ratio: float = 0.85):
    """Build preprocessing transforms.

    Args:
        use_circular_crop: Whether to include CircularCrop transform.
        radius_ratio: Radius ratio for CircularCrop.

    Returns:
        List of transform functions.
    """
    import sys
    sys.path.insert(0, '/home/hyeonghwan/github/mmpose')

    from mmengine.registry import init_default_scope
    init_default_scope('mmpose')

    from mmpose.datasets.transforms import (
        LoadImage, CircularCrop, GetBBoxCenterScale, TopdownAffine
    )

    transforms = [LoadImage()]

    if use_circular_crop:
        transforms.append(CircularCrop(
            radius_ratio=radius_ratio,
            soft_edge=True,
            edge_width=0.1
        ))

    transforms.extend([
        GetBBoxCenterScale(padding=1.0),
        TopdownAffine(input_size=(256, 256)),
    ])

    return transforms


def process_image(img_path: str, transforms: list) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single image through the pipeline.

    Args:
        img_path: Path to image file.
        transforms: List of transform functions.

    Returns:
        Tuple of (original_image, processed_image).
    """
    # Load original
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w = orig_img.shape[:2]

    # Create sample dict
    sample = {
        'img_path': img_path,
        'bbox': np.array([[0, 0, w, h]], dtype=np.float32),
        'bbox_score': np.array([1.0], dtype=np.float32),
    }

    # Apply transforms
    results = sample.copy()
    for t in transforms:
        results = t(results)

    return orig_img, results['img']


def visualize_test_images(
    test_data_root: str = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet',
    num_samples: int = 6,
    radius_ratio: float = 0.85,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize test images with and without CircularCrop.

    Args:
        test_data_root: Path to test data directory.
        num_samples: Number of samples to visualize.
        radius_ratio: Radius ratio for CircularCrop.
        output_path: Path to save the figure. If None, auto-generate.
        show: Whether to display the figure.
    """
    # Build transforms
    transforms_no_crop = build_transforms(use_circular_crop=False)
    transforms_with_crop = build_transforms(use_circular_crop=True, radius_ratio=radius_ratio)

    # Collect test image paths
    test_paths = []
    for seq in ['olek_outdoor', 'weipeng_studio']:
        img_dir = os.path.join(test_data_root, seq)
        if os.path.exists(img_dir):
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            for img_file in img_files[:num_samples // 2]:
                test_paths.append(os.path.join(img_dir, img_file))

    test_paths = test_paths[:num_samples]

    # Process images
    originals, no_crop, with_crop = [], [], []
    for img_path in test_paths:
        orig, proc_no = process_image(img_path, transforms_no_crop)
        _, proc_with = process_image(img_path, transforms_with_crop)
        originals.append(orig)
        no_crop.append(proc_no)
        with_crop.append(proc_with)

    # Create figure
    n = len(test_paths)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle(f'Mo2Cap2 Test Images Pipeline (CircularCrop radius_ratio={radius_ratio})', fontsize=14)

    for i in range(n):
        axes[0, i].imshow(originals[i])
        axes[0, i].set_title(f'Original\n{originals[i].shape[1]}x{originals[i].shape[0]}')
        axes[0, i].axis('off')

        axes[1, i].imshow(no_crop[i].astype(np.uint8))
        axes[1, i].set_title('No CircularCrop')
        axes[1, i].axis('off')

        axes[2, i].imshow(with_crop[i].astype(np.uint8))
        axes[2, i].set_title('With CircularCrop')
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('No Crop', fontsize=12)
    axes[2, 0].set_ylabel('With Crop', fontsize=12)

    plt.tight_layout()

    if output_path is None:
        output_path = f'output_mo2cap2_vis/pipeline_comparison_r{radius_ratio}.png'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    if show:
        plt.show()
    plt.close()


def visualize_train_vs_test(
    train_data_root: str = '/mnt/sdb2/mo2cap2_dataset/training_data',
    test_data_root: str = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet',
    num_samples: int = 4,
    radius_ratio: float = 0.85,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Compare train and test images after pipeline processing.

    Args:
        train_data_root: Path to training data (H5 chunks).
        test_data_root: Path to test data directory.
        num_samples: Number of samples per dataset.
        radius_ratio: Radius ratio for CircularCrop on test images.
        output_path: Path to save the figure.
        show: Whether to display the figure.
    """
    # Build transforms
    transforms_train = build_transforms(use_circular_crop=False)
    transforms_test = build_transforms(use_circular_crop=True, radius_ratio=radius_ratio)

    # Load train images from H5
    train_processed = []
    chunk_files = sorted([f for f in os.listdir(train_data_root) if f.startswith('chunk_') and f.endswith('.h5')])

    # Save temp images from H5 to process through pipeline
    import tempfile
    temp_dir = tempfile.mkdtemp()

    for chunk_file in chunk_files[:2]:
        chunk_path = os.path.join(train_data_root, chunk_file)
        with h5py.File(chunk_path, 'r') as f:
            images = f['images'][:]
            for i in range(min(num_samples, len(images))):
                img = images[i]
                if img.shape[0] == 3:  # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))

                # Save temp file
                temp_path = os.path.join(temp_dir, f'train_{len(train_processed)}.png')
                cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Process through pipeline
                _, processed = process_image(temp_path, transforms_train)
                train_processed.append(processed)

                if len(train_processed) >= num_samples:
                    break
        if len(train_processed) >= num_samples:
            break

    # Load and process test images
    test_processed = []
    for seq in ['olek_outdoor', 'weipeng_studio']:
        img_dir = os.path.join(test_data_root, seq)
        if os.path.exists(img_dir):
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            for img_file in img_files[:num_samples // 2]:
                img_path = os.path.join(img_dir, img_file)
                _, processed = process_image(img_path, transforms_test)
                test_processed.append(processed)
                if len(test_processed) >= num_samples:
                    break
        if len(test_processed) >= num_samples:
            break

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)

    # Create figure
    n = max(len(train_processed), len(test_processed))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(f'Train vs Test (Test with CircularCrop r={radius_ratio})', fontsize=14)

    for i in range(n):
        if i < len(train_processed):
            axes[0, i].imshow(train_processed[i].astype(np.uint8))
            axes[0, i].set_title(f'Train {i+1}')
        axes[0, i].axis('off')

        if i < len(test_processed):
            axes[1, i].imshow(test_processed[i].astype(np.uint8))
            axes[1, i].set_title(f'Test {i+1}')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Train', fontsize=12)
    axes[1, 0].set_ylabel('Test', fontsize=12)

    plt.tight_layout()

    if output_path is None:
        output_path = f'output_mo2cap2_vis/train_vs_test_r{radius_ratio}.png'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    if show:
        plt.show()
    plt.close()


def test_radius_ratios(
    test_data_root: str = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet',
    ratios: List[float] = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Test different radius ratios for CircularCrop.

    Args:
        test_data_root: Path to test data directory.
        ratios: List of radius ratios to test.
        output_path: Path to save the figure.
        show: Whether to display the figure.
    """
    # Get one test image
    for seq in ['olek_outdoor', 'weipeng_studio']:
        img_dir = os.path.join(test_data_root, seq)
        if os.path.exists(img_dir):
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            if img_files:
                test_img_path = os.path.join(img_dir, img_files[0])
                break

    # Process with different ratios
    results = []
    for r in ratios:
        transforms = build_transforms(use_circular_crop=(r < 1.0), radius_ratio=r)
        _, processed = process_image(test_img_path, transforms)
        results.append((r, processed))

    # Create figure
    n = len(ratios)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle('CircularCrop Radius Ratio Comparison', fontsize=14)

    for i, (r, img) in enumerate(results):
        axes[i].imshow(img.astype(np.uint8))
        axes[i].set_title(f'r={r}')
        axes[i].axis('off')

    plt.tight_layout()

    if output_path is None:
        output_path = 'output_mo2cap2_vis/radius_ratio_comparison.png'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Mo2Cap2 pipeline')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'compare', 'radius'],
                        help='Visualization mode')
    parser.add_argument('--radius', type=float, default=0.85,
                        help='CircularCrop radius ratio')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of samples to visualize')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure')
    args = parser.parse_args()

    if args.mode == 'test':
        visualize_test_images(
            radius_ratio=args.radius,
            num_samples=args.num_samples,
            show=not args.no_show
        )
    elif args.mode == 'compare':
        visualize_train_vs_test(
            radius_ratio=args.radius,
            num_samples=args.num_samples,
            show=not args.no_show
        )
    elif args.mode == 'radius':
        test_radius_ratios(show=not args.no_show)
