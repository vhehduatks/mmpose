#!/usr/bin/env python
"""
Extract preprocessed images from H5 cache to disk.

This script extracts 256x256 preprocessed images from the H5 cache file
and saves them as regular image files for fast loading during testing.

Usage:
    python tools/dataset_converters/extract_h5cache_images.py \
        --h5-cache /home/hyeonghwan/h5cache/test_cache_with_images.h5 \
        --output-dir /home/hyeonghwan/h5cache/test_images_256 \
        --num-workers 8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract images from H5 cache to disk')
    parser.add_argument('--h5-cache', type=str, required=True,
                        help='Path to H5 cache file with images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for extracted images')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'jpg'],
                        help='Output image format')
    return parser.parse_args()


def save_image_batch(args):
    """Save a batch of images to disk."""
    indices, images, output_dir, img_format = args
    saved_paths = []

    for idx, img in zip(indices, images):
        # Convert RGB to BGR for OpenCV
        img_bgr = img[:, :, ::-1]

        # Create output path
        output_path = os.path.join(output_dir, f'{idx:06d}.{img_format}')

        # Save image
        if img_format == 'jpg':
            cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(output_path, img_bgr)

        saved_paths.append(output_path)

    return saved_paths


def extract_images(h5_cache: str, output_dir: str, num_workers: int = 8,
                   img_format: str = 'png'):
    """Extract all images from H5 cache to disk."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading H5 cache from {h5_cache}...')

    with h5py.File(h5_cache, 'r') as hf:
        n_samples = hf.attrs['n_samples']
        img_size = hf.attrs.get('img_size', 256)

        print(f'Found {n_samples} images ({img_size}x{img_size})')
        print(f'Extracting to {output_dir}...')

        # Process in batches for memory efficiency
        batch_size = 1000
        all_paths = []

        for start_idx in tqdm(range(0, n_samples, batch_size), desc='Extracting'):
            end_idx = min(start_idx + batch_size, n_samples)

            # Load batch of images
            images = hf['images'][start_idx:end_idx]
            indices = list(range(start_idx, end_idx))

            # Save images
            for idx, img in zip(indices, images):
                # Convert RGB to BGR for OpenCV
                img_bgr = img[:, :, ::-1]
                output_path = os.path.join(output_dir, f'{idx:06d}.{img_format}')

                if img_format == 'jpg':
                    cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(output_path, img_bgr)

                all_paths.append(output_path)

    print(f'Extracted {len(all_paths)} images to {output_dir}')

    # Calculate total size
    total_size = sum(os.path.getsize(p) for p in all_paths[:100]) * len(all_paths) / 100
    print(f'Estimated total size: {total_size / (1024**3):.2f} GB')

    return all_paths


def main():
    args = parse_args()
    extract_images(
        h5_cache=args.h5_cache,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        img_format=args.format
    )


if __name__ == '__main__':
    main()
