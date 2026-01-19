#!/usr/bin/env python
"""
Build H5 Cache for EgoPose Dataset with Images

This script pre-processes all annotations AND images into a single HDF5 file
for fast loading during testing. Images are preprocessed to 256x256.

Usage:
    python tools/dataset_converters/build_egopose_h5cache_with_images.py \
        --data-root /mnt/sdb2/xr_egopose_full/TestSet \
        --output /home/hyeonghwan/h5cache/test_cache_with_images.h5 \
        --num-workers 8
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Add mmpose to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mmpose.datasets.datasets.body3d.config import config

CM_TO_M = 100
IMG_SIZE = 256  # Preprocessed image size


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build H5 cache with images for EgoPose dataset')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output H5 file path')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Chunk size for parallel processing')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size to resize to')
    return parser.parse_args()


def preprocess_hmd_data(p3d: np.ndarray) -> np.ndarray:
    """Preprocess HMD data from 3D keypoints."""
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


def load_and_preprocess_image(img_path: str, img_size: int = 256) -> np.ndarray:
    """Load and preprocess image to target size with center crop."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    # Center crop to square (1000x800 -> 800x800, then resize)
    if h != w:
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        img = img[top:top+min_dim, left:left+min_dim]

    # Resize to target size
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    return img


def parse_single_sample(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Parse a single JSON annotation file."""
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


def process_batch(args: Tuple[List[str], List[str], List[int], int]) -> Dict:
    """Process a batch of samples (images + annotations)."""
    json_paths, img_paths, indices, img_size = args

    results = {
        'indices': indices,
        'keypoints': [],
        'keypoint3d': [],
        'hmd_info': [],
        'actions': [],
        'images': []
    }

    for json_path, img_path in zip(json_paths, img_paths):
        try:
            p2d, p3d, hmd, action = parse_single_sample(json_path)
            img = load_and_preprocess_image(img_path, img_size)

            results['keypoints'].append(p2d)
            results['keypoint3d'].append(p3d)
            results['hmd_info'].append(hmd)
            results['actions'].append(action)
            results['images'].append(img)
        except Exception as e:
            print(f'Error processing {json_path}: {e}')
            results['keypoints'].append(np.zeros((16, 2), dtype=np.float32))
            results['keypoint3d'].append(np.zeros((16, 3), dtype=np.float32))
            results['hmd_info'].append(np.zeros(9, dtype=np.float32))
            results['actions'].append('unknown')
            results['images'].append(np.zeros((img_size, img_size, 3), dtype=np.uint8))

    return results


def index_directory(path: str, root_dirs: List[str]) -> Dict[str, List[str]]:
    """Recursively index all files in directory."""
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


def build_cache_with_images(data_root: str, output_path: str,
                            num_workers: int = 4,
                            chunk_size: int = 500,
                            img_size: int = 256):
    """Build H5 cache with preprocessed images."""

    print(f'Indexing dataset files from {data_root}...')
    index = index_directory(data_root, ['rgba', 'json'])

    n_samples = len(index['json'])
    print(f'Found {n_samples} samples')

    if n_samples == 0:
        print('No samples found!')
        return

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f'Processing with {num_workers} workers...')
    print(f'Image size: {img_size}x{img_size}')

    # Create H5 file with chunked datasets for memory efficiency
    # No compression for faster I/O on NVMe SSD
    with h5py.File(output_path, 'w') as hf:
        # Pre-create datasets
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('img_paths', (n_samples,), dtype=dt)
        hf.create_dataset('actions', (n_samples,), dtype=dt)
        hf.create_dataset('keypoints', (n_samples, 1, 16, 2), dtype=np.float32,
                         chunks=(min(100, n_samples), 1, 16, 2))
        hf.create_dataset('keypoint3d', (n_samples, 1, 16, 3), dtype=np.float32,
                         chunks=(min(100, n_samples), 1, 16, 3))
        hf.create_dataset('hmd_info', (n_samples, 1, 9), dtype=np.float32,
                         chunks=(min(100, n_samples), 1, 9))
        # Images dataset - no compression for maximum read speed
        hf.create_dataset('images', (n_samples, img_size, img_size, 3), dtype=np.uint8,
                         chunks=(10, img_size, img_size, 3))

        # Process in chunks
        chunks = []
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunks.append((
                index['json'][i:end_idx],
                index['rgba'][i:end_idx],
                list(range(i, end_idx)),
                img_size
            ))

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_batch, chunk): chunk[2][0]
                          for chunk in chunks}

                for future in tqdm(as_completed(futures), total=len(chunks),
                                 desc='Processing chunks'):
                    result = future.result()
                    for i, idx in enumerate(result['indices']):
                        hf['img_paths'][idx] = index['rgba'][idx]
                        hf['actions'][idx] = result['actions'][i]
                        hf['keypoints'][idx, 0] = result['keypoints'][i]
                        hf['keypoint3d'][idx, 0] = result['keypoint3d'][i]
                        hf['hmd_info'][idx, 0] = result['hmd_info'][i]
                        hf['images'][idx] = result['images'][i]
        else:
            for chunk in tqdm(chunks, desc='Processing chunks'):
                result = process_batch(chunk)
                for i, idx in enumerate(result['indices']):
                    hf['img_paths'][idx] = index['rgba'][idx]
                    hf['actions'][idx] = result['actions'][i]
                    hf['keypoints'][idx, 0] = result['keypoints'][i]
                    hf['keypoint3d'][idx, 0] = result['keypoint3d'][i]
                    hf['hmd_info'][idx, 0] = result['hmd_info'][i]
                    hf['images'][idx] = result['images'][i]

        # Metadata
        hf.attrs['n_samples'] = n_samples
        hf.attrs['version'] = '2.0'
        hf.attrs['has_images'] = True
        hf.attrs['img_size'] = img_size

    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024 * 1024)
    print(f'Cache built successfully!')
    print(f'  - Samples: {n_samples}')
    print(f'  - File size: {file_size:.2f} GB')
    print(f'  - Location: {output_path}')


def main():
    args = parse_args()
    build_cache_with_images(
        data_root=args.data_root,
        output_path=args.output,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        img_size=args.img_size
    )


if __name__ == '__main__':
    main()
