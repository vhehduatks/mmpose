"""
Compare Mo2Cap2 Training vs Validation Image Preprocessing

Compares:
1. H5 training image (already 256x256 in chunk)
2. Val image processed through CenterCrop → Resize (simulating val pipeline)
3. Val image just resized (without CenterCrop)

Purpose: Verify if train and val images have same preprocessing
"""

import os
import sys
import cv2
import h5py
import numpy as np
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_h5_training_image(chunk_idx: int = 1, local_idx: int = 0) -> np.ndarray:
    """Load a training image from H5 chunk.

    Returns:
        BGR image (H, W, C) format
    """
    chunk_path = f'/mnt/sdb2/mo2cap2_dataset/training_data/mo2cap2_chunk_{chunk_idx:04d}.hdf5'

    with h5py.File(chunk_path, 'r') as hf:
        # Images are (N, C, H, W) RGB format
        img = hf['Images'][local_idx]  # (3, 256, 256)

    # Convert CHW RGB → HWC BGR
    img = img.transpose(1, 2, 0)  # (256, 256, 3)
    img_bgr = img[:, :, ::-1].copy()  # RGB → BGR

    return img_bgr


def load_val_image_with_centercrop(img_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load val image and apply CenterCrop + Resize (proper val pipeline).

    Returns:
        BGR image (H, W, C) format
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {img_path}")

    h, w = img.shape[:2]

    # Apply CenterCrop (128px margins)
    margin_left = 128
    margin_right = 128
    if w > margin_left + margin_right:
        img = img[:, margin_left:w-margin_right].copy()

    # Resize to target
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img


def load_val_image_no_crop(img_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load val image with direct resize (no CenterCrop).

    Returns:
        BGR image (H, W, C) format
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {img_path}")

    # Direct resize without cropping
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    return img


def analyze_image(img: np.ndarray, name: str) -> dict:
    """Analyze image statistics."""
    h, w = img.shape[:2]

    # Corner values (check for vignetting/masking)
    corner_size = 10
    corners = {
        'top_left': img[:corner_size, :corner_size].mean(),
        'top_right': img[:corner_size, -corner_size:].mean(),
        'bottom_left': img[-corner_size:, :corner_size].mean(),
        'bottom_right': img[-corner_size:, -corner_size:].mean(),
    }

    # Center values
    ch, cw = h // 2, w // 2
    center = img[ch-corner_size:ch+corner_size, cw-corner_size:cw+corner_size].mean()

    stats = {
        'name': name,
        'shape': img.shape,
        'min': img.min(),
        'max': img.max(),
        'mean': img.mean(),
        'std': img.std(),
        'corners': corners,
        'corner_avg': np.mean(list(corners.values())),
        'center': center,
        'center_to_corner_ratio': center / max(np.mean(list(corners.values())), 1e-6),
    }

    return stats


def print_stats(stats: dict):
    """Print image statistics."""
    print(f"\n{'='*60}")
    print(f"  {stats['name']}")
    print(f"{'='*60}")
    print(f"  Shape: {stats['shape']}")
    print(f"  Min/Max: {stats['min']:.2f} / {stats['max']:.2f}")
    print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    print(f"  Corner values:")
    for k, v in stats['corners'].items():
        print(f"    {k}: {v:.2f}")
    print(f"  Corner avg: {stats['corner_avg']:.2f}")
    print(f"  Center value: {stats['center']:.2f}")
    print(f"  Center/Corner ratio: {stats['center_to_corner_ratio']:.2f}")


def create_comparison_visualization(
    train_img: np.ndarray,
    val_crop_img: np.ndarray,
    val_nocrop_img: np.ndarray,
    output_path: str
):
    """Create side-by-side comparison visualization."""

    # Calculate differences
    diff_crop = cv2.absdiff(train_img, val_crop_img)
    diff_nocrop = cv2.absdiff(train_img, val_nocrop_img)

    # Amplify differences for visibility
    diff_crop_vis = (diff_crop * 3).clip(0, 255).astype(np.uint8)
    diff_nocrop_vis = (diff_nocrop * 3).clip(0, 255).astype(np.uint8)

    # Create panels
    panel_h, panel_w = 256, 256

    def add_label(img: np.ndarray, label: str, sublabel: str = '') -> np.ndarray:
        img = img.copy()
        cv2.putText(img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if sublabel:
            cv2.putText(img, sublabel, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        return img

    # Row 1: Original images
    train_panel = add_label(train_img, 'Train (H5)', f'mean={train_img.mean():.1f}')
    val_crop_panel = add_label(val_crop_img, 'Val (CenterCrop)', f'mean={val_crop_img.mean():.1f}')
    val_nocrop_panel = add_label(val_nocrop_img, 'Val (No Crop)', f'mean={val_nocrop_img.mean():.1f}')

    row1 = np.hstack([train_panel, val_crop_panel, val_nocrop_panel])

    # Row 2: Difference images
    diff_crop_panel = add_label(diff_crop_vis, 'Train vs CenterCrop', f'MAE={diff_crop.mean():.2f}')
    diff_nocrop_panel = add_label(diff_nocrop_vis, 'Train vs NoCrop', f'MAE={diff_nocrop.mean():.2f}')
    empty_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Add verdict
    if diff_crop.mean() < diff_nocrop.mean():
        verdict_text = 'Train matches CenterCrop'
        verdict_color = (0, 255, 0)  # Green
    else:
        verdict_text = 'Train matches NoCrop'
        verdict_color = (0, 0, 255)  # Red

    cv2.putText(empty_panel, 'VERDICT:', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(empty_panel, verdict_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, verdict_color, 1)
    cv2.putText(empty_panel, f'CenterCrop MAE: {diff_crop.mean():.2f}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(empty_panel, f'NoCrop MAE: {diff_nocrop.mean():.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    row2 = np.hstack([diff_crop_panel, diff_nocrop_panel, empty_panel])

    # Combine
    combined = np.vstack([row1, row2])

    # Add title
    title_bar = np.ones((40, combined.shape[1], 3), dtype=np.uint8) * 50
    cv2.putText(title_bar, 'Mo2Cap2 Train vs Val Preprocessing Comparison', (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.vstack([title_bar, combined])

    cv2.imwrite(output_path, combined)
    print(f"\nSaved comparison to: {output_path}")

    return diff_crop.mean(), diff_nocrop.mean()


def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare Mo2Cap2 train vs val preprocessing')
    parser.add_argument('--train-chunk', type=int, default=1, help='H5 chunk index for training')
    parser.add_argument('--train-sample', type=int, default=0, help='Sample index within chunk')
    parser.add_argument('--output', type=str, default='output_preprocessing_compare', help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("="*70)
    print("Mo2Cap2 Train vs Val Preprocessing Comparison")
    print("="*70)

    # Load training image from H5
    print(f"\nLoading training image: chunk {args.train_chunk}, sample {args.train_sample}")
    train_img = load_h5_training_image(args.train_chunk, args.train_sample)
    train_stats = analyze_image(train_img, f'Train (H5 chunk {args.train_chunk}, sample {args.train_sample})')
    print_stats(train_stats)

    # Save training image
    cv2.imwrite(os.path.join(args.output, 'train_h5_image.jpg'), train_img)

    # Load a test image for val pipeline comparison
    test_root = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet'
    sequences = ['olek_outdoor', 'weipeng_studio']

    for seq in sequences:
        seq_dir = os.path.join(test_root, seq)
        img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])

        if len(img_files) == 0:
            continue

        # Use first image from each sequence
        val_img_path = os.path.join(seq_dir, img_files[0])

        print(f"\n{'#'*70}")
        print(f"# Comparing with: {seq}")
        print(f"{'#'*70}")

        # Val with CenterCrop (proper pipeline)
        val_crop_img = load_val_image_with_centercrop(val_img_path)
        val_crop_stats = analyze_image(val_crop_img, f'Val ({seq}) - CenterCrop')
        print_stats(val_crop_stats)

        # Val without CenterCrop (direct resize)
        val_nocrop_img = load_val_image_no_crop(val_img_path)
        val_nocrop_stats = analyze_image(val_nocrop_img, f'Val ({seq}) - No Crop')
        print_stats(val_nocrop_stats)

        # Create comparison visualization
        output_path = os.path.join(args.output, f'compare_{seq}.jpg')
        mae_crop, mae_nocrop = create_comparison_visualization(
            train_img, val_crop_img, val_nocrop_img, output_path
        )

        print(f"\n  COMPARISON RESULTS for {seq}:")
        print(f"    Train vs CenterCrop MAE: {mae_crop:.2f}")
        print(f"    Train vs NoCrop MAE: {mae_nocrop:.2f}")

        # Save individual images
        cv2.imwrite(os.path.join(args.output, f'val_{seq}_centercrop.jpg'), val_crop_img)
        cv2.imwrite(os.path.join(args.output, f'val_{seq}_nocrop.jpg'), val_nocrop_img)

    # Final analysis: check if training images have circular mask (fisheye vignetting)
    print("\n" + "="*70)
    print("CIRCULAR MASK ANALYSIS (Fisheye Vignetting)")
    print("="*70)

    # Check if corners are significantly darker than center
    if train_stats['center_to_corner_ratio'] > 5:
        print(f"  Training images appear to have CIRCULAR MASK")
        print(f"  Center/Corner brightness ratio: {train_stats['center_to_corner_ratio']:.2f}")
        print(f"  This is expected for fisheye cameras with vignetting")
    else:
        print(f"  No significant circular mask detected")
        print(f"  Center/Corner brightness ratio: {train_stats['center_to_corner_ratio']:.2f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Findings:
1. Training images in H5 are ALREADY 256x256
2. Val images go through: LoadImage → CenterCrop (128px) → TopdownAffine (256x256)

Potential Preprocessing Differences:
- If Train images were created WITH CenterCrop: MAE with CenterCrop < MAE without
- If Train images were created WITHOUT CenterCrop: MAE with CenterCrop > MAE without

Note: The original Mo2Cap2 dataset provides pre-processed 256x256 images in HDF5 format.
The CenterCrop in val pipeline may have been introduced later for this MMPose integration.
""")


if __name__ == '__main__':
    main()
