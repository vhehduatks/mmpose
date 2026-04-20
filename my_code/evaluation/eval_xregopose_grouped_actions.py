"""
Evaluate xR-EgoPose model with official 9-category action grouping.

Uses MMPose's dataset and model infrastructure directly — loads the
H5CachedEgoposeDataset via the saved config, runs inference, then
groups the 43 individual actions into 9 official categories.

Usage:
    python my_code/evaluation/eval_xregopose_grouped_actions.py
"""

import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate, default_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmpose.registry import MODELS, DATASETS

# ── Official xR-EgoPose 9-category grouping ──────────────────────────────

GROUP_NAMES = [
    'Gesticulating', 'Reacting', 'Greeting', 'Talking',
    'UpperStretching', 'Gaming', 'LowerStretching', 'Patting', 'Walking',
]

_ACT_TO_GID = {
    'anim_Clip1': 8, 'Opening_A_Lid': 0, 'Dribble': 5, 'Boxing': 5,
    'Standing_Arguing__1_': 3, 'Happy': 3, 'Plotting': 3, 'Counting': 4,
    'Standing_Arguing': 0, 'Standing_2H_Cast_Spell_01': 4, 'Shooting_Gun': 5,
    'Two_Hand_Spell_Casting': 0, 'Shaking_Hands_2': 2, 'Hands_Forward_Gesture': 2,
    'Rifle_Punch': 1, 'Baseball_Umpire': 5, 'Angry_Gesture': 0, 'Waving_Gesture': 0,
    'Taunt_Gesture': 0, 'Golf_Putt_Failure': 5, 'Rejected': 1, 'Shake_Fist': 2,
    'Revealing_Dice': 5, 'Golf_Putt_Failure__1_': 5, 'No': 3, 'Angry_Point': 1,
    'Agreeing': 3, 'Sitting_Thumbs_Up': 6, 'Standing_Thumbs_Up': 4, 'Patting': 7,
    'Petting': 7, 'Petting_Animal': 7, 'Taking_Punch': 0,
    'Standing_1H_Magic_Attack_01': 4, 'Talking': 3, 'Standing_Greeting': 2,
    'Happy_Hand_Gesture': 0, 'Dismissing_Gesture': 1, 'Strong_Gesture': 1,
    'Pointing_Gesture': 1, 'Golf_Putt_Victory': 5, 'Pointing': 0,
    'Thinking': 4, 'Loser': 1, 'Reaching_Out': 3, 'Crazy_Gesture': 0,
    'Golf_Putt_Victory__1_': 5, 'Insult': 3, 'Arm_Gesture': 0,
    'Beckoning': 1, 'Charge': 5, 'Weight_Shift_Gesture': 8,
    'Pain_Gesture': 1, 'Fist_Pump': 0, 'Terrified': 1, 'Surprised': 1,
    'Clapping': 1, 'Rallying': 1, 'Hand_Raising': 0, 'Sitting_Disapproval': 6,
    'Quick_Formal_Bow': 2, 'Counting__1_': 0, 'Tpose_Take_001': 4,
    'upper_stretching': 4, 'lower_stretching': 6, 'walking': 8,
}
ACT_TO_GROUP = {k: GROUP_NAMES[v] for k, v in _ACT_TO_GID.items()}

UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 15]
LOWER_BODY = [8, 9, 10, 11, 12, 13, 14]


def strip_action(raw):
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode('utf-8')
    s = re.findall(r'_mixamo_com.*', raw)
    if s:
        raw = raw.replace(s[0], '')
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='/mnt/dataset_vol/xr_egodataset_best/'
                   'HMD_xregopose_cascaded_both_from_ground_v3_full/'
                   'HMD_xregopose_cascaded_both_from_ground_v3_full_config.py')
    p.add_argument('--checkpoint', default='/mnt/dataset_vol/xr_egodataset_best/'
                   'HMD_xregopose_cascaded_both_from_ground_v3_full/'
                   'best_xregopose_Full Body_All_mpjpe_epoch_8.pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', default='my_code/custom_config/PER_ACTION_RESULTS.md')
    args = p.parse_args()

    init_default_scope('mmpose')
    cfg = Config.fromfile(args.config)

    # Build model
    print('Building model...')
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    model.to(args.device)
    model.eval()
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    metainfo = dict(from_file='mmpose/datasets/datasets/body3d/egopose_info.py')
    model.dataset_meta = parse_pose_metainfo(metainfo)

    # Build test dataset & dataloader from the saved val config
    print('Building dataset...')
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    print(f'Dataset size: {len(dataset)}')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=pseudo_collate,
    )

    # Run inference
    print(f'Running inference ({len(dataset)} samples)...')
    all_pred_3d = []
    all_gt_3d = []
    all_actions = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(f'  batch {batch_idx}/{len(dataloader)}')

        with torch.no_grad():
            results = model.test_step(batch)

        data_samples_list = batch['data_samples']
        for i, r in enumerate(results):
            # Predicted 3D
            pred_3d = r.pred_instances.keypoint_3d.cpu().numpy()
            if pred_3d.ndim == 3:
                pred_3d = pred_3d[0]
            all_pred_3d.append(pred_3d)

            # GT 3D
            gt_3d = data_samples_list[i].gt_instance_labels.keypoint3d.cpu().numpy()
            if gt_3d.ndim == 3:
                gt_3d = gt_3d[0]
            all_gt_3d.append(gt_3d)

            # Action
            action = data_samples_list[i].metainfo.get('action', '')
            all_actions.append(strip_action(action))

    all_pred_3d = np.array(all_pred_3d)
    all_gt_3d = np.array(all_gt_3d)
    N = len(all_pred_3d)
    print(f'Inference done: {N} samples')

    # Compute per-sample MPJPE (mm)
    errors_per_joint = np.sqrt(np.sum((all_pred_3d - all_gt_3d) ** 2, axis=-1)) * 1000
    full_errors = errors_per_joint.mean(axis=1)
    upper_errors = errors_per_joint[:, UPPER_BODY].mean(axis=1)
    lower_errors = errors_per_joint[:, LOWER_BODY].mean(axis=1)

    # Group by 9 categories
    group_full = defaultdict(list)
    group_upper = defaultdict(list)
    group_lower = defaultdict(list)
    indiv_full = defaultdict(list)

    for i in range(N):
        act = all_actions[i]
        indiv_full[act].append(full_errors[i])
        group = ACT_TO_GROUP.get(act)
        if group:
            group_full[group].append(full_errors[i])
            group_upper[group].append(upper_errors[i])
            group_lower[group].append(lower_errors[i])

    overall_full = float(np.mean(full_errors))
    overall_upper = float(np.mean(upper_errors))
    overall_lower = float(np.mean(lower_errors))

    # Print results
    print(f'\n{"="*70}')
    print(f'xR-EgoPose Official 9-Category Evaluation')
    print(f'{"="*70}')
    print(f'Overall: Full={overall_full:.2f}  Upper={overall_upper:.2f}  Lower={overall_lower:.2f}')
    print(f'\n{"Category":<20s} {"Full":>8s} {"Upper":>8s} {"Lower":>8s} {"N":>8s}')
    print('-' * 50)

    rows = []
    for gn in GROUP_NAMES:
        if gn in group_full:
            f = float(np.mean(group_full[gn]))
            u = float(np.mean(group_upper[gn]))
            l = float(np.mean(group_lower[gn]))
            n = len(group_full[gn])
            print(f'{gn:<20s} {f:8.2f} {u:8.2f} {l:8.2f} {n:8d}')
            rows.append((gn, f, u, l, n))
    print('-' * 50)
    print(f'{"All":<20s} {overall_full:8.2f} {overall_upper:8.2f} {overall_lower:8.2f} {N:8d}')

    # Write markdown
    md = [
        '# xR-EgoPose Per-Action Results (Official 9-Category Grouping)',
        '',
        f'> Checkpoint: `{os.path.basename(args.checkpoint)}`',
        f'> Test set: {N:,} samples, 43 actions grouped into 9 categories',
        f'> Evaluation protocol: Official xR-EgoPose grouping',
        '',
        '## Grouped Results (MPJPE in mm)',
        '',
        '| Category | Full Body | Upper Body | Lower Body | Samples |',
        '|----------|-----------|------------|------------|---------|',
    ]
    for gn, f, u, l, n in rows:
        md.append(f'| {gn} | {f:.2f} | {u:.2f} | {l:.2f} | {n:,} |')
    md.append(f'| **All** | **{overall_full:.2f}** | **{overall_upper:.2f}** | **{overall_lower:.2f}** | **{N:,}** |')

    md.extend([
        '',
        '## Individual Action Results (sorted by MPJPE)',
        '',
        '| Action | Group | MPJPE (mm) | Samples |',
        '|--------|-------|------------|---------|',
    ])
    for act, errs in sorted(indiv_full.items(), key=lambda x: np.mean(x[1])):
        g = ACT_TO_GROUP.get(act, '?')
        md.append(f'| {act} | {g} | {np.mean(errs):.2f} | {len(errs):,} |')

    # Paper table row
    md.extend([
        '',
        '## Paper Table Row',
        '',
        '| Method | Gaming | Gestic. | Greeting | LowerStr. | Patting | Reacting | Talking | UpperStr. | Walking | All |',
        '|--------|--------|---------|----------|-----------|---------|----------|---------|-----------|---------|-----|',
    ])
    paper = '| **Ours** |'
    for gn in ['Gaming', 'Gesticulating', 'Greeting', 'LowerStretching',
               'Patting', 'Reacting', 'Talking', 'UpperStretching', 'Walking']:
        v = float(np.mean(group_full[gn])) if gn in group_full else 0
        paper += f' {v:.1f} |'
    paper += f' **{overall_full:.1f}** |'
    md.append(paper)
    md.append('')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(md))
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
