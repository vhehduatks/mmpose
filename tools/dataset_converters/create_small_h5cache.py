"""
Create a small H5 cache dataset for quick testing.

Usage:
    python tools/dataset_converters/create_small_h5cache.py \
        --src F:/egodataset_cache/h5cache/train_cache_with_images.h5 \
        --dst F:/egodataset_cache/h5cache/train_small_500.h5 \
        --num-samples 500
"""
import argparse
import h5py
import numpy as np
from tqdm import tqdm


def create_small_cache(src_path: str, dst_path: str, num_samples: int, seed: int = 42):
    """
    Extract a small subset from H5 cache.

    Args:
        src_path: Source H5 cache file path
        dst_path: Destination H5 cache file path
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    with h5py.File(src_path, 'r') as src:
        total_samples = src['images'].shape[0]
        print(f"Source file: {src_path}")
        print(f"Total samples: {total_samples}")
        print(f"Extracting: {num_samples} samples")

        # Random sampling
        if num_samples >= total_samples:
            indices = np.arange(total_samples)
        else:
            indices = np.random.choice(total_samples, num_samples, replace=False)
            indices = np.sort(indices)

        print(f"Selected indices: {indices[:5]}...{indices[-5:]}")

        with h5py.File(dst_path, 'w') as dst:
            # Copy each dataset with selected indices
            for key in tqdm(src.keys(), desc="Copying datasets"):
                data = src[key]

                if key in ['actions', 'img_paths']:
                    # String/object arrays - need special handling
                    selected = [data[i] for i in indices]
                    dt = h5py.special_dtype(vlen=str)
                    dst.create_dataset(key, data=selected, dtype=dt)
                else:
                    # Numeric arrays
                    selected = data[indices]
                    dst.create_dataset(key, data=selected, dtype=data.dtype)

                print(f"  {key}: {data.shape} -> {dst[key].shape}")

            # Copy attributes and add n_samples
            actual_samples = len(indices)
            dst.attrs['n_samples'] = actual_samples

            # Copy other attributes from source if they exist
            for attr_key in src.attrs.keys():
                if attr_key != 'n_samples':
                    dst.attrs[attr_key] = src.attrs[attr_key]

    print(f"\nCreated: {dst_path}")
    print(f"Samples: {num_samples}")


def main():
    parser = argparse.ArgumentParser(description="Create small H5 cache for testing")
    parser.add_argument('--src', type=str, required=True, help='Source H5 cache file')
    parser.add_argument('--dst', type=str, required=True, help='Destination H5 cache file')
    parser.add_argument('--num-samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    create_small_cache(args.src, args.dst, args.num_samples, args.seed)


if __name__ == '__main__':
    main()
