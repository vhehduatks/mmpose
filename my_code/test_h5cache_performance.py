#!/usr/bin/env python
"""
Test script to compare dataset loading performance:
- Original CustomEgoposeDataset (JSON parsing)
- H5CachedEgoposeDataset (HDF5 cache)

Usage:
    python my_code/test_h5cache_performance.py --data-root F:/ego_cam_dataset/Train

Expected results:
    - Original: ~5-10 minutes for 65k samples
    - H5 Cached: ~3-5 seconds for 65k samples
"""

import argparse
import os
import sys
import time

# Add mmpose to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Test H5 cache performance')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--test-original', action='store_true',
                        help='Also test original dataset (slow!)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Force rebuild H5 cache')
    return parser.parse_args()


def test_h5cached_dataset(data_root: str, rebuild_cache: bool = False):
    """Test H5CachedEgoposeDataset loading time."""
    from mmpose.datasets.datasets.body3d import H5CachedEgoposeDataset

    print('\n' + '=' * 60)
    print('Testing H5CachedEgoposeDataset')
    print('=' * 60)

    cache_file = os.path.join(data_root, 'annotations_cache.h5')
    cache_exists = os.path.exists(cache_file)

    if cache_exists and not rebuild_cache:
        print(f'Cache file exists: {cache_file}')
        print(f'Cache size: {os.path.getsize(cache_file) / (1024*1024):.2f} MB')
    else:
        print('Cache will be built on first load...')

    # Simple pipeline for testing
    pipeline = [
        dict(type='LoadImage'),
        dict(padding=1.0, type='GetBBoxCenterScale'),
        dict(input_size=(256, 256), type='TopdownAffine'),
        dict(type='PackPoseInputs'),
    ]

    start_time = time.time()

    dataset = H5CachedEgoposeDataset(
        data_mode='topdown',
        data_root=data_root,
        rebuild_cache=rebuild_cache,
        pipeline=pipeline,
    )

    load_time = time.time() - start_time

    print(f'\nResults:')
    print(f'  - Number of samples: {len(dataset)}')
    print(f'  - Loading time: {load_time:.2f} seconds')
    print(f'  - Time per sample: {load_time / len(dataset) * 1000:.4f} ms')

    # Test accessing a few samples
    print('\nTesting sample access...')
    start_time = time.time()
    for i in [0, len(dataset) // 2, len(dataset) - 1]:
        sample = dataset[i]
        print(f'  - Sample {i}: img_path={sample["inputs"].shape if hasattr(sample.get("inputs", {}), "shape") else "N/A"}')
    access_time = time.time() - start_time
    print(f'  - Access time for 3 samples: {access_time:.4f} seconds')

    return load_time, len(dataset)


def test_original_dataset(data_root: str):
    """Test original CustomEgoposeDataset loading time."""
    from mmpose.datasets.datasets.body3d import CustomEgoposeDataset

    print('\n' + '=' * 60)
    print('Testing Original CustomEgoposeDataset (This will take a while...)')
    print('=' * 60)

    pipeline = [
        dict(type='LoadImage'),
        dict(padding=1.0, type='GetBBoxCenterScale'),
        dict(input_size=(256, 256), type='TopdownAffine'),
        dict(type='PackPoseInputs'),
    ]

    start_time = time.time()

    dataset = CustomEgoposeDataset(
        data_mode='topdown',
        data_root=data_root,
        pipeline=pipeline,
    )

    load_time = time.time() - start_time

    print(f'\nResults:')
    print(f'  - Number of samples: {len(dataset)}')
    print(f'  - Loading time: {load_time:.2f} seconds ({load_time/60:.2f} minutes)')
    print(f'  - Time per sample: {load_time / len(dataset) * 1000:.4f} ms')

    return load_time, len(dataset)


def main():
    args = parse_args()

    # Initialize MMPose registry
    from mmpose.utils import register_all_modules
    register_all_modules()

    results = {}

    # Test H5 cached dataset
    h5_time, n_samples = test_h5cached_dataset(
        args.data_root,
        rebuild_cache=args.rebuild_cache
    )
    results['h5_cached'] = h5_time

    # Optionally test original dataset
    if args.test_original:
        orig_time, _ = test_original_dataset(args.data_root)
        results['original'] = orig_time

        # Print comparison
        print('\n' + '=' * 60)
        print('Performance Comparison')
        print('=' * 60)
        print(f'Original dataset:  {orig_time:.2f} seconds ({orig_time/60:.2f} minutes)')
        print(f'H5 cached dataset: {h5_time:.2f} seconds')
        print(f'Speedup: {orig_time / h5_time:.1f}x faster')
    else:
        print('\n' + '=' * 60)
        print('Summary')
        print('=' * 60)
        print(f'H5 cached dataset: {h5_time:.2f} seconds for {n_samples} samples')
        print(f'\nTo compare with original dataset, run with --test-original flag')
        print('(Warning: original dataset loading can take 5-10+ minutes)')


if __name__ == '__main__':
    main()
