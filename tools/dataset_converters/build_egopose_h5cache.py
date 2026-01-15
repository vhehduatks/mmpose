#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Build H5 Cache for EgoPose Dataset

This script pre-processes all JSON annotations in the EgoPose dataset
and saves them to a single HDF5 file for fast loading during training.

Usage:
    python tools/dataset_converters/build_egopose_h5cache.py \
        --data-root F:/ego_cam_dataset/Train \
        --output F:/ego_cam_dataset/Train/annotations_cache.h5

    # With segmentation and depth support:
    python tools/dataset_converters/build_egopose_h5cache.py \
        --data-root F:/ego_cam_dataset/Train \
        --output F:/ego_cam_dataset/Train/annotations_cache.h5 \
        --with-seg-depth

    # Parallel processing (faster):
    python tools/dataset_converters/build_egopose_h5cache.py \
        --data-root F:/ego_cam_dataset/Train \
        --num-workers 8

Expected speedup:
    - Original: ~5-10 minutes to load 65k samples
    - With cache: ~3-5 seconds to load 65k samples
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

# Add mmpose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mmpose.datasets.datasets.body3d.config import config


CM_TO_M = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build H5 cache for EgoPose dataset')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset (e.g., F:/ego_cam_dataset/Train)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output H5 file path. Default: {data-root}/annotations_cache.h5')
    parser.add_argument('--with-seg-depth', action='store_true',
                        help='Include segmentation and depth paths in cache')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers for processing')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Chunk size for parallel processing')
    return parser.parse_args()


def preprocess_hmd_data(p3d: np.ndarray) -> np.ndarray:
    """Preprocess HMD data from 3D keypoints.

    Args:
        p3d: (16, 3) array of 3D keypoints

    Returns:
        (9,) array of HMD features
    """
    p3d = np.array(p3d)

    head = p3d[0]
    right_hand = p3d[7]
    left_hand = p3d[4]

    midpoint = (right_hand + left_hand) / 2
    z_axis = midpoint - head
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        z_axis = np.array([0, 0, 1])

    hand_vector = right_hand - left_hand
    x_axis = np.cross(z_axis, hand_vector)
    x_norm = np.linalg.norm(x_axis)
    if x_norm > 1e-6:
        x_axis = x_axis / x_norm
    else:
        x_axis = np.array([1, 0, 0])

    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    right_local = np.dot(rotation_matrix.T, (right_hand - head))
    left_local = np.dot(rotation_matrix.T, (left_hand - head))

    hand_distance = np.linalg.norm(right_local - left_local)
    right_distance = np.linalg.norm(right_local)
    left_distance = np.linalg.norm(left_local)

    return np.concatenate([
        right_local, left_local,
        [hand_distance, right_distance, left_distance]
    ]).astype(np.float32)


def parse_single_sample(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Parse a single JSON annotation file.

    Args:
        json_path: Path to JSON file

    Returns:
        Tuple of (keypoints_2d, keypoints_3d, hmd_info, action)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    joint_names = {j['name'].replace('mixamorig:', ''): jid
                   for jid, j in enumerate(data['joints'])}
    p2d_orig = np.array(data['pts2d_fisheye']).T
    p3d_orig = np.array(data['pts3d_fisheye']).T
    action = data['action']

    p2d = np.empty([len(config.skel), 2], dtype=np.float32)
    p3d = np.empty([len(config.skel), 3], dtype=np.float32)

    for jid, j in enumerate(config.skel.keys()):
        p2d[jid] = p2d_orig[joint_names[j]]
        p3d[jid] = p3d_orig[joint_names[j]]

    p3d /= CM_TO_M
    hmd = preprocess_hmd_data(p3d)

    return p2d, p3d, hmd, action


def parse_batch(args: Tuple[List[str], List[int]]) -> Dict:
    """Parse a batch of JSON files.

    Args:
        args: Tuple of (json_paths, indices)

    Returns:
        Dict with parsed data
    """
    json_paths, indices = args
    results = {
        'indices': indices,
        'keypoints': [],
        'keypoint3d': [],
        'hmd_info': [],
        'actions': []
    }

    for json_path in json_paths:
        try:
            p2d, p3d, hmd, action = parse_single_sample(json_path)
            results['keypoints'].append(p2d)
            results['keypoint3d'].append(p3d)
            results['hmd_info'].append(hmd)
            results['actions'].append(action)
        except Exception as e:
            print(f'Error parsing {json_path}: {e}')
            # Add zeros for failed samples
            results['keypoints'].append(np.zeros((16, 2), dtype=np.float32))
            results['keypoint3d'].append(np.zeros((16, 3), dtype=np.float32))
            results['hmd_info'].append(np.zeros(9, dtype=np.float32))
            results['actions'].append('unknown')

    return results


def index_directory(path: str, root_dirs: List[str]) -> Dict[str, List[str]]:
    """Recursively index all files in directory.

    Args:
        path: Root directory path
        root_dirs: List of subdirectory names to look for (e.g., ['rgba', 'json'])

    Returns:
        Dict mapping root_dir names to file path lists
    """
    indexed_paths = {k: [] for k in root_dirs}

    try:
        sub_dirs = sorted(next(os.walk(path))[1])
    except StopIteration:
        return indexed_paths

    if set(root_dirs) <= set(sub_dirs):
        for sub_dir in root_dirs:
            d_path = os.path.join(path, sub_dir)
            files = sorted([
                os.path.join(d_path, f)
                for f in os.listdir(d_path)
                if os.path.isfile(os.path.join(d_path, f))
            ])
            indexed_paths[sub_dir] = files
        return indexed_paths

    for sub_dir in sub_dirs:
        sub_indexed = index_directory(os.path.join(path, sub_dir), root_dirs)
        for r_dir in root_dirs:
            indexed_paths[r_dir].extend(sub_indexed[r_dir])

    return indexed_paths


def build_cache(data_root: str, output_path: str,
                with_seg_depth: bool = False,
                num_workers: int = 1,
                chunk_size: int = 1000):
    """Build H5 cache for EgoPose dataset.

    Args:
        data_root: Root directory of dataset
        output_path: Output H5 file path
        with_seg_depth: Whether to include seg/depth paths
        num_workers: Number of parallel workers
        chunk_size: Chunk size for parallel processing
    """
    # Determine which directories to index
    root_dirs = ['rgba', 'json']
    if with_seg_depth:
        root_dirs.extend(['depth', 'objectId'])

    print(f'Indexing dataset files from {data_root}...')
    index = index_directory(data_root, root_dirs)

    n_samples = len(index['json'])
    print(f'Found {n_samples} samples')

    if n_samples == 0:
        print('No samples found! Check your data_root path.')
        return

    # Verify file counts match
    for key in root_dirs:
        if len(index[key]) != n_samples:
            print(f'Warning: {key} has {len(index[key])} files, expected {n_samples}')

    # Pre-allocate arrays
    img_paths = index['rgba']
    keypoints = np.empty((n_samples, 1, 16, 2), dtype=np.float32)
    keypoint3d = np.empty((n_samples, 1, 16, 3), dtype=np.float32)
    hmd_info = np.empty((n_samples, 1, 9), dtype=np.float32)
    actions = []

    if with_seg_depth:
        depth_paths = index['depth']
        seg_paths = index['objectId']

    print(f'Parsing annotations with {num_workers} worker(s)...')

    if num_workers > 1:
        # Parallel processing
        chunks = []
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunks.append((
                index['json'][i:end_idx],
                list(range(i, end_idx))
            ))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(parse_batch, chunk): chunk[1][0]
                      for chunk in chunks}

            for future in tqdm(as_completed(futures), total=len(chunks),
                             desc='Processing chunks'):
                result = future.result()
                for i, idx in enumerate(result['indices']):
                    keypoints[idx, 0] = result['keypoints'][i]
                    keypoint3d[idx, 0] = result['keypoint3d'][i]
                    hmd_info[idx, 0] = result['hmd_info'][i]

                actions.extend(result['actions'])
    else:
        # Sequential processing with progress bar
        for idx, json_path in enumerate(tqdm(index['json'], desc='Parsing')):
            try:
                p2d, p3d, hmd, action = parse_single_sample(json_path)
                keypoints[idx, 0] = p2d
                keypoint3d[idx, 0] = p3d
                hmd_info[idx, 0] = hmd
                actions.append(action)
            except Exception as e:
                print(f'Error parsing {json_path}: {e}')
                keypoints[idx, 0] = np.zeros((16, 2), dtype=np.float32)
                keypoint3d[idx, 0] = np.zeros((16, 3), dtype=np.float32)
                hmd_info[idx, 0] = np.zeros(9, dtype=np.float32)
                actions.append('unknown')

    print(f'Saving cache to {output_path}...')

    with h5py.File(output_path, 'w') as hf:
        # String arrays
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('img_paths', data=img_paths, dtype=dt)
        hf.create_dataset('actions', data=actions, dtype=dt)

        if with_seg_depth:
            hf.create_dataset('depth_paths', data=depth_paths, dtype=dt)
            hf.create_dataset('seg_paths', data=seg_paths, dtype=dt)

        # Numeric arrays with compression
        hf.create_dataset('keypoints', data=keypoints, dtype=np.float32,
                         compression='gzip', compression_opts=4)
        hf.create_dataset('keypoint3d', data=keypoint3d, dtype=np.float32,
                         compression='gzip', compression_opts=4)
        hf.create_dataset('hmd_info', data=hmd_info, dtype=np.float32,
                         compression='gzip', compression_opts=4)

        # Metadata
        hf.attrs['n_samples'] = n_samples
        hf.attrs['version'] = '1.0'
        hf.attrs['has_depth'] = with_seg_depth
        hf.attrs['has_seg'] = with_seg_depth

    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Cache built successfully!')
    print(f'  - Samples: {n_samples}')
    print(f'  - File size: {file_size:.2f} MB')
    print(f'  - Location: {output_path}')


def main():
    args = parse_args()

    # Determine output path
    output_path = args.output
    if output_path is None:
        suffix = '_segdepth' if args.with_seg_depth else ''
        output_path = os.path.join(args.data_root, f'annotations_cache{suffix}.h5')

    # Build cache
    build_cache(
        data_root=args.data_root,
        output_path=output_path,
        with_seg_depth=args.with_seg_depth,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size
    )


if __name__ == '__main__':
    main()
