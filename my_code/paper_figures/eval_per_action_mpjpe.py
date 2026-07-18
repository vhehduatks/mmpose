"""Per-fine-action MPJPE evaluation for the Kinect dataset.

Builds the val_dataloader from each model's training config (so frame
filtering — `use_2d_visible=True` and friends — matches training-val exactly)
and runs inference through mmpose's standard `data_preprocessor` +
`mode='predict'` path. This produces predictions that are byte-for-byte
compatible with `CustomxRegoposeMetric.process` and reproduces the training-val
Full/Upper/Lower MPJPE numbers exactly.

Crucially, this script aggregates MPJPE by the **raw** action label
(`Dancing1`, `Gaming-Archery`, `Workout-BicelCurl`, …) instead of routing
through `xr_egopose_evaluate.map_action_name`, which collapses these to nine
coarse categories (`Dancing`, `Gaming`, `Workout`, …). The output is therefore
directly usable for the per-action paper table.

Joint indexing (UPPER / LOWER) follows
`mmpose/evaluation/metrics/custom_xr_egopose_metric.py:227`.

Why this exists: `scan_session_mpjpe.py` and the inference helpers in
`generate_fig7_qualitative.py` are visualization-grade. They skip the
`use_2d_visible` filter, so they include ~800 tracking-failure frames from the
Val split that the training-val pipeline drops. Including those frames inflates
the OVERALL MPJPE by ~3 mm uniformly across models. This script avoids that by
going through the official dataset class and metric pipeline.

Examples
--------
# Single model
python my_code/paper_figures/eval_per_action_mpjpe.py \
    --config my_code/custom_config/HMD_kinect_v5_flag_cascaded_image_only_10ep_config.py \
    --checkpoint 'work_dirs/.../best_*.pth' \
    --name CascadedImageOnly \
    --output /tmp/per_action_image_only.csv

# Smoke test (first 5 batches only)
python my_code/paper_figures/eval_per_action_mpjpe.py \
    --config <cfg> --checkpoint <ckpt> --name <name> --output <csv> --limit 5
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# Joint indices: matches CustomxRegoposeMetric (custom_xr_egopose_metric.py:227)
UPPER = [0, 1, 2, 3, 4, 5, 6, 7]
LOWER = [8, 9, 10, 11, 12, 13, 14, 15]


def parse_args():
    p = argparse.ArgumentParser(
        description='Per-fine-action MPJPE evaluation (matches training val).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help='Training config of the model to evaluate.')
    p.add_argument('--checkpoint', required=True,
                   help='Checkpoint to load (any best_*.pth or epoch_*.pth).')
    p.add_argument('--name', required=True,
                   help='Column-name prefix written into the CSV header.')
    p.add_argument('--output', required=True,
                   help='Output CSV path (one row per action + an OVERALL row).')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke-test: stop after N batches.')
    return p.parse_args()


def main():
    args = parse_args()
    init_default_scope('mmpose')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = '/tmp/mmpose_eval_workdir'
    cfg.model.train_cfg = None
    # Configs trained with DDP set launcher='pytorch'; force single-process
    # so we can run from a plain `python` invocation.
    cfg.launcher = 'none'

    # Build dataloader and model via mmengine.Runner so the config's
    # val_pipeline + dataset (use_2d_visible=True etc) is honored exactly.
    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device).eval()

    val_loader = runner.val_dataloader

    # Aggregate per-action errors.
    action_full = defaultdict(list)
    action_upper = defaultdict(list)
    action_lower = defaultdict(list)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if args.limit is not None and i >= args.limit:
                break
            data = model.data_preprocessor(data, training=False)
            outputs = model(data['inputs'], data['data_samples'], mode='predict')
            for sample in outputs:
                pred = sample.pred_instances.keypoint_3d  # (1, 16, 3)
                gt = sample.gt_instance_labels.keypoint3d  # (1, 16, 3)
                if hasattr(pred, 'cpu'):
                    pred = pred.cpu().numpy()
                if hasattr(gt, 'cpu'):
                    gt = gt.cpu().numpy()
                pred = np.asarray(pred).reshape(16, 3)
                gt = np.asarray(gt).reshape(16, 3)
                err = np.linalg.norm(pred - gt, axis=-1) * 1000.0  # m → mm
                action = sample.get('action', 'Unknown')
                if isinstance(action, bytes):
                    action = action.decode()
                action_full[str(action)].append(float(err.mean()))
                action_upper[str(action)].append(float(err[UPPER].mean()))
                action_lower[str(action)].append(float(err[LOWER].mean()))

    # One row per action + an OVERALL summary row.
    rows = []
    overall_full, overall_upper, overall_lower = [], [], []
    for action in sorted(action_full):
        full = np.array(action_full[action])
        upper = np.array(action_upper[action])
        lower = np.array(action_lower[action])
        rows.append({
            'action': action,
            'n_frames': len(full),
            f'{args.name}_full': float(full.mean()),
            f'{args.name}_upper': float(upper.mean()),
            f'{args.name}_lower': float(lower.mean()),
        })
        overall_full.extend(full.tolist())
        overall_upper.extend(upper.tolist())
        overall_lower.extend(lower.tolist())
    rows.append({
        'action': 'OVERALL',
        'n_frames': len(overall_full),
        f'{args.name}_full': float(np.mean(overall_full)),
        f'{args.name}_upper': float(np.mean(overall_upper)),
        f'{args.name}_lower': float(np.mean(overall_lower)),
    })

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"OVERALL ({len(overall_full)} frames): "
          f"full={np.mean(overall_full):.2f} "
          f"upper={np.mean(overall_upper):.2f} "
          f"lower={np.mean(overall_lower):.2f}")


if __name__ == '__main__':
    main()
