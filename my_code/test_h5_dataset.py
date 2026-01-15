"""Test script for H5Mo2Cap2Dataset"""

import time
import sys

print("=" * 60)
print("Testing H5Mo2Cap2Dataset")
print("=" * 60)

# Test 1: Import test
print("\n[1/5] Testing imports...")
try:
    from mmpose.datasets.datasets.body3d import H5Mo2Cap2Dataset, H5Mo2Cap2Dataset_Lazy
    from mmpose.datasets.transforms import LoadImageFromH5
    print("  OK - All imports successful!")
except Exception as e:
    print(f"  FAILED - Import error: {e}")
    sys.exit(1)

# Test 2: First load (builds cache)
print("\n[2/5] Testing first load (builds cache)...")
data_root = r'F:\mo2cap2_dataset\training_data'

try:
    start_time = time.time()
    dataset = H5Mo2Cap2Dataset(
        data_root=data_root,
        data_mode='topdown',
        pipeline=[],  # Empty pipeline for now
        sample_interval=1,  # Use ALL data
        input_size=(256, 256),
    )
    load_time = time.time() - start_time

    print(f"  OK - First load (cache build) in {load_time:.2f} seconds")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Chunk files: {len(dataset.chunk_files)}")
    del dataset
except Exception as e:
    print(f"  FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Second load (from cache)
print("\n[3/5] Testing second load (from cache)...")
try:
    start_time = time.time()
    dataset = H5Mo2Cap2Dataset(
        data_root=data_root,
        data_mode='topdown',
        pipeline=[],
        sample_interval=1,
        input_size=(256, 256),
    )
    load_time = time.time() - start_time

    print(f"  OK - Cache load in {load_time:.2f} seconds")
    print(f"  Total samples: {len(dataset)}")
except Exception as e:
    print(f"  FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Sample data access
print("\n[4/5] Testing sample data access...")
try:
    sample = dataset[0]
    print(f"  OK - Sample keys: {list(sample.keys())}")
    print(f"  - keypoints shape: {sample['keypoints'].shape}")
    print(f"  - keypoint3d shape: {sample['keypoint3d'].shape}")
    print(f"  - hmd_info shape: {sample['hmd_info'].shape}")
    print(f"  - h5_chunk_path: {sample['h5_chunk_path']}")
    print(f"  - h5_local_idx: {sample['h5_local_idx']}")
except Exception as e:
    print(f"  FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: LoadImageFromH5 transform
print("\n[5/5] Testing LoadImageFromH5 transform...")
try:
    import numpy as np

    transform = LoadImageFromH5(to_float32=False)

    # Get a sample and apply transform
    sample = dataset[0]
    result = transform.transform(sample)

    img = result['img']
    print(f"  OK - Image loaded successfully")
    print(f"  - Image shape: {img.shape}")
    print(f"  - Image dtype: {img.dtype}")
    print(f"  - Image range: [{img.min()}, {img.max()}]")

    # Save a test image
    import cv2
    test_img_path = r'C:\Users\user\Documents\GitHub\mmpose\my_code\test_h5_image.png'
    cv2.imwrite(test_img_path, img)
    print(f"  - Test image saved to: {test_img_path}")

except Exception as e:
    print(f"  FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
