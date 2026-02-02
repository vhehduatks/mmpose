"""
Measure model size and inference FPS for Cascaded V2b model.
Run on the training server with GPU and PyTorch installed.

Usage:
    python my_code/measure_model_v2b.py
"""

import torch
import time
import os
from mmengine.config import Config
from mmpose.apis import init_model

def main():
    # Load V2b model
    config_path = 'my_code/custom_config/HMD_xregopose_cascaded_both_from_ground_v2b_full_config.py'
    checkpoint_path = 'work_dirs/HMD_xregopose_cascaded_both_from_ground_v2b_full/best_xregopose_Full Body_All_mpjpe_epoch_19.pth'

    print("Loading model...")
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== Model Size ===")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # Breakdown by component
    print(f"\n=== Parameter Breakdown ===")
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Backbone (ResNet-101): {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    print(f"Head (Cascaded Enhanced): {head_params:,} ({head_params/1e6:.2f}M)")

    # Head submodule breakdown
    print(f"\n=== Head Breakdown ===")
    head = model.head
    for name, module in head.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"  {name}: {params:,} ({params/1e6:.2f}M)")

    # Checkpoint file size
    ckpt_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\nCheckpoint File Size: {ckpt_size:.2f} MB")

    # Inference FPS measurement
    print(f"\n=== Inference Speed (Batch Size 1) ===")
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 256, 256).cuda()

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model.forward(dummy_input, data_samples=None, mode='tensor')

    # Measure
    num_iterations = 200
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model.forward(dummy_input, data_samples=None, mode='tensor')

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    fps = num_iterations / elapsed
    latency = elapsed / num_iterations * 1000

    print(f"Iterations: {num_iterations}")
    print(f"Total Time: {elapsed:.3f}s")
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {latency:.2f} ms/frame")

    # Test with larger batch sizes
    for batch_size in [8, 16, 32]:
        try:
            dummy_input = torch.randn(batch_size, 3, 256, 256).cuda()

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model.forward(dummy_input, data_samples=None, mode='tensor')

            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(50):
                with torch.no_grad():
                    _ = model.forward(dummy_input, data_samples=None, mode='tensor')

            torch.cuda.synchronize()
            end_time = time.time()

            elapsed = end_time - start_time
            throughput = (50 * batch_size) / elapsed

            print(f"\nBatch Size {batch_size}:")
            print(f"Throughput: {throughput:.2f} samples/sec")
        except RuntimeError as e:
            print(f"\nBatch Size {batch_size}: OOM - {e}")
            break

    # Compare with V1 if available
    v1_ckpt = 'work_dirs/HMD_xregopose_cascaded_both_from_ground_full/best_xregopose_Full Body_All_mpjpe_epoch_10.pth'
    if os.path.exists(v1_ckpt):
        print(f"\n=== V1 Comparison ===")
        v1_size = os.path.getsize(v1_ckpt) / (1024 * 1024)
        print(f"V1 Checkpoint Size: {v1_size:.2f} MB")
        print(f"V2b Checkpoint Size: {ckpt_size:.2f} MB")
        print(f"Size Reduction: {(v1_size - ckpt_size) / v1_size * 100:.1f}%")

if __name__ == '__main__':
    main()
