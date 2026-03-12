"""
Calculate Mo2Cap2 Dataset Normalization Statistics

Computes per-channel mean and std for:
1. Training set (H5 chunks - 530k images)
2. Test set (JPG files - 5.6k images)

Usage:
    python my_code/visualization/calculate_mo2cap2_normalization.py
    python my_code/visualization/calculate_mo2cap2_normalization.py --sample-ratio 0.1  # Use 10% of data
"""

import os
import sys
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def calculate_train_stats(
    data_root: str = '/mnt/sdb2/mo2cap2_dataset/training_data',
    sample_ratio: float = 1.0,
    chunk_pattern: str = 'mo2cap2_chunk_'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and std for Mo2Cap2 training set from H5 chunks.

    Images in H5 are stored as (N, C, H, W) in RGB format.

    Args:
        data_root: Path to training data directory
        sample_ratio: Fraction of data to use (1.0 = all)
        chunk_pattern: Pattern for chunk files

    Returns:
        mean: (3,) array of per-channel means (RGB order)
        std: (3,) array of per-channel stds (RGB order)
    """
    print("=" * 70)
    print("Calculating Training Set Statistics")
    print("=" * 70)

    # Find all chunk files
    chunk_files = sorted([
        os.path.join(data_root, f)
        for f in os.listdir(data_root)
        if f.startswith(chunk_pattern) and f.endswith('.hdf5')
    ])

    if len(chunk_files) == 0:
        raise ValueError(f"No chunk files found in {data_root}")

    print(f"Found {len(chunk_files)} chunk files")

    # Sample chunks if ratio < 1
    if sample_ratio < 1.0:
        n_chunks = max(1, int(len(chunk_files) * sample_ratio))
        chunk_indices = np.linspace(0, len(chunk_files) - 1, n_chunks, dtype=int)
        chunk_files = [chunk_files[i] for i in chunk_indices]
        print(f"Sampling {len(chunk_files)} chunks ({sample_ratio*100:.1f}%)")

    # Use Welford's online algorithm for numerical stability
    n_pixels = 0
    mean = np.zeros(3, dtype=np.float64)
    M2 = np.zeros(3, dtype=np.float64)  # Sum of squared differences

    for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
        with h5py.File(chunk_path, 'r') as hf:
            # Images: (N, C, H, W) RGB uint8
            images = hf['Images'][:]  # (1000, 3, 256, 256)

            # Convert to float and reshape to (N*H*W, C)
            N, C, H, W = images.shape
            images = images.astype(np.float64)
            images = images.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, 3)

            # Welford's algorithm for batch update
            for pixel in images:
                n_pixels += 1
                delta = pixel - mean
                mean += delta / n_pixels
                delta2 = pixel - mean
                M2 += delta * delta2

    # Final variance and std
    variance = M2 / n_pixels
    std = np.sqrt(variance)

    print(f"\nTraining Set Statistics (RGB, 0-255 scale):")
    print(f"  Total pixels: {n_pixels:,}")
    print(f"  Mean: R={mean[0]:.3f}, G={mean[1]:.3f}, B={mean[2]:.3f}")
    print(f"  Std:  R={std[0]:.3f}, G={std[1]:.3f}, B={std[2]:.3f}")

    return mean, std


def calculate_train_stats_fast(
    data_root: str = '/mnt/sdb2/mo2cap2_dataset/training_data',
    sample_ratio: float = 0.1,
    chunk_pattern: str = 'mo2cap2_chunk_'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast calculation using batch processing (less memory efficient but faster).

    Args:
        data_root: Path to training data directory
        sample_ratio: Fraction of data to use
        chunk_pattern: Pattern for chunk files

    Returns:
        mean: (3,) array of per-channel means (RGB order)
        std: (3,) array of per-channel stds (RGB order)
    """
    print("=" * 70)
    print("Calculating Training Set Statistics (Fast Method)")
    print("=" * 70)

    # Find all chunk files
    chunk_files = sorted([
        os.path.join(data_root, f)
        for f in os.listdir(data_root)
        if f.startswith(chunk_pattern) and f.endswith('.hdf5')
    ])

    if len(chunk_files) == 0:
        raise ValueError(f"No chunk files found in {data_root}")

    print(f"Found {len(chunk_files)} chunk files")

    # Sample chunks
    if sample_ratio < 1.0:
        n_chunks = max(1, int(len(chunk_files) * sample_ratio))
        chunk_indices = np.linspace(0, len(chunk_files) - 1, n_chunks, dtype=int)
        chunk_files = [chunk_files[i] for i in chunk_indices]
        print(f"Sampling {len(chunk_files)} chunks ({sample_ratio*100:.1f}%)")

    # Collect channel sums and squared sums
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
        with h5py.File(chunk_path, 'r') as hf:
            # Images: (N, C, H, W) RGB uint8
            images = hf['Images'][:].astype(np.float64)  # (1000, 3, 256, 256)

            N, C, H, W = images.shape
            n_pixels += N * H * W

            # Sum per channel
            for c in range(C):
                channel_sum[c] += images[:, c, :, :].sum()
                channel_sq_sum[c] += (images[:, c, :, :] ** 2).sum()

    # Calculate mean and std
    mean = channel_sum / n_pixels
    variance = (channel_sq_sum / n_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    print(f"\nTraining Set Statistics (RGB, 0-255 scale):")
    print(f"  Total pixels: {n_pixels:,}")
    print(f"  Mean: R={mean[0]:.3f}, G={mean[1]:.3f}, B={mean[2]:.3f}")
    print(f"  Std:  R={std[0]:.3f}, G={std[1]:.3f}, B={std[2]:.3f}")

    return mean, std


def calculate_test_stats(
    data_root: str = '/mnt/sdb2/mo2cap2_dataset/test_data/TestSet',
    apply_centercrop: bool = True,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and std for Mo2Cap2 test set.

    Applies same preprocessing as val pipeline:
    - CenterCrop (128px margins)
    - Resize to 256x256

    Args:
        data_root: Path to test data directory
        apply_centercrop: Whether to apply CenterCrop before calculating
        target_size: Target size for resize

    Returns:
        mean: (3,) array of per-channel means (RGB order)
        std: (3,) array of per-channel stds (RGB order)
    """
    print("=" * 70)
    print(f"Calculating Test Set Statistics (CenterCrop={apply_centercrop})")
    print("=" * 70)

    sequences = ['olek_outdoor', 'weipeng_studio']

    # Collect channel sums and squared sums
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for seq in sequences:
        seq_dir = os.path.join(data_root, seq)
        if not os.path.exists(seq_dir):
            print(f"Warning: {seq_dir} not found, skipping")
            continue

        img_files = sorted([
            f for f in os.listdir(seq_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

        print(f"Processing {seq}: {len(img_files)} images")

        for img_file in tqdm(img_files, desc=f"  {seq}"):
            img_path = os.path.join(seq_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Convert BGR to RGB
            img = img[:, :, ::-1].astype(np.float64)

            h, w = img.shape[:2]

            # Apply CenterCrop if requested
            if apply_centercrop:
                margin_left = 128
                margin_right = 128
                if w > margin_left + margin_right:
                    img = img[:, margin_left:w-margin_right, :]

            # Resize to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

            H, W, C = img.shape
            n_pixels += H * W

            # Sum per channel
            for c in range(C):
                channel_sum[c] += img[:, :, c].sum()
                channel_sq_sum[c] += (img[:, :, c] ** 2).sum()

    # Calculate mean and std
    mean = channel_sum / n_pixels
    variance = (channel_sq_sum / n_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    print(f"\nTest Set Statistics (RGB, 0-255 scale):")
    print(f"  Total pixels: {n_pixels:,}")
    print(f"  Mean: R={mean[0]:.3f}, G={mean[1]:.3f}, B={mean[2]:.3f}")
    print(f"  Std:  R={std[0]:.3f}, G={std[1]:.3f}, B={std[2]:.3f}")

    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Calculate Mo2Cap2 normalization statistics')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='Fraction of training data to sample (default: 0.1)')
    parser.add_argument('--full', action='store_true',
                        help='Use full training data (overrides --sample-ratio)')
    parser.add_argument('--train-only', action='store_true', help='Only calculate train stats')
    parser.add_argument('--test-only', action='store_true', help='Only calculate test stats')

    args = parser.parse_args()

    sample_ratio = 1.0 if args.full else args.sample_ratio

    results = {}

    # Calculate training stats
    if not args.test_only:
        train_mean, train_std = calculate_train_stats_fast(sample_ratio=sample_ratio)
        results['train'] = {'mean': train_mean, 'std': train_std}

    # Calculate test stats
    if not args.train_only:
        test_mean, test_std = calculate_test_stats(apply_centercrop=True)
        results['test'] = {'mean': test_mean, 'std': test_std}

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Dataset-Specific Normalization Values")
    print("=" * 70)

    print("\nImageNet defaults (for reference):")
    print("  mean = [123.675, 116.28, 103.53]")
    print("  std  = [58.395, 57.12, 57.375]")

    if 'train' in results:
        print(f"\nMo2Cap2 TRAINING set (RGB order):")
        print(f"  mean = [{results['train']['mean'][0]:.3f}, {results['train']['mean'][1]:.3f}, {results['train']['mean'][2]:.3f}]")
        print(f"  std  = [{results['train']['std'][0]:.3f}, {results['train']['std'][1]:.3f}, {results['train']['std'][2]:.3f}]")

    if 'test' in results:
        print(f"\nMo2Cap2 TEST set (RGB order, with CenterCrop):")
        print(f"  mean = [{results['test']['mean'][0]:.3f}, {results['test']['mean'][1]:.3f}, {results['test']['mean'][2]:.3f}]")
        print(f"  std  = [{results['test']['std'][0]:.3f}, {results['test']['std'][1]:.3f}, {results['test']['std'][2]:.3f}]")

    # Generate config snippet
    print("\n" + "=" * 70)
    print("CONFIG SNIPPET (copy to your config file)")
    print("=" * 70)

    if 'train' in results and 'test' in results:
        print("""
# Mo2Cap2 dataset-specific normalization
# Calculated from actual dataset statistics

# Option 1: Use training set statistics (recommended for training)
MO2CAP2_TRAIN_MEAN = [{:.3f}, {:.3f}, {:.3f}]
MO2CAP2_TRAIN_STD = [{:.3f}, {:.3f}, {:.3f}]

# Option 2: Use test set statistics (for test-only evaluation)
MO2CAP2_TEST_MEAN = [{:.3f}, {:.3f}, {:.3f}]
MO2CAP2_TEST_STD = [{:.3f}, {:.3f}, {:.3f}]

# Option 3: Use combined statistics (average of train and test)
MO2CAP2_COMBINED_MEAN = [{:.3f}, {:.3f}, {:.3f}]
MO2CAP2_COMBINED_STD = [{:.3f}, {:.3f}, {:.3f}]

# Apply in data_preprocessor:
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=MO2CAP2_TRAIN_MEAN,  # or MO2CAP2_COMBINED_MEAN
        std=MO2CAP2_TRAIN_STD,    # or MO2CAP2_COMBINED_STD
        type='PoseDataPreprocessor'
    ),
    ...
)
""".format(
            results['train']['mean'][0], results['train']['mean'][1], results['train']['mean'][2],
            results['train']['std'][0], results['train']['std'][1], results['train']['std'][2],
            results['test']['mean'][0], results['test']['mean'][1], results['test']['mean'][2],
            results['test']['std'][0], results['test']['std'][1], results['test']['std'][2],
            (results['train']['mean'][0] + results['test']['mean'][0]) / 2,
            (results['train']['mean'][1] + results['test']['mean'][1]) / 2,
            (results['train']['mean'][2] + results['test']['mean'][2]) / 2,
            (results['train']['std'][0] + results['test']['std'][0]) / 2,
            (results['train']['std'][1] + results['test']['std'][1]) / 2,
            (results['train']['std'][2] + results['test']['std'][2]) / 2,
        ))


if __name__ == '__main__':
    main()
