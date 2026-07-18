"""
Scan a dataset root and compute per-session mean MPJPE for one or more trained
models. Prints a table, saves a CSV, and optionally a JSON blob.

Default models match the web GUI's LHF vs GBH comparison
(see generate_fig7_qualitative.py for the config/checkpoint defaults).

Examples
--------
# Zero-config: LHF + GBH on the Val split
python my_code/paper_figures/scan_session_mpjpe.py \
    --data-root /mnt/dataset_vol/kinect_v5_split/Val

# Explicit model set (repeatable --model NAME CONFIG CKPT HMD_MODE)
python my_code/paper_figures/scan_session_mpjpe.py \
    --data-root /mnt/dataset_vol/kinect_v5_split/Val \
    --model Single my_code/custom_config/...stage1_only_ground_info_10ep_config.py \
            work_dirs/.../best.pth hmd_12 \
    --model Cascaded my_code/custom_config/...cascaded_ground_info_10ep_config.py \
            work_dirs/.../best.pth hmd_12 \
    --output stage_cmp.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_fig7_qualitative import (  # noqa: E402
    REPO_ROOT,
    CONFIG_LHF, CKPT_LHF, CONFIG_GBH, CKPT_GBH,
    load_all_frames, load_model, build_pipeline,
    run_inference, compute_mpjpe,
)


DEFAULT_DATA_ROOT = '/mnt/dataset_vol/kinect_v5_split/Val'

# (name, config, ckpt, hmd_mode); hmd_mode in {'hmd_9', 'hmd_12'}
DEFAULT_MODELS = [
    ('LHF', CONFIG_LHF, CKPT_LHF, 'hmd_9'),
    ('GBH', CONFIG_GBH, CKPT_GBH, 'hmd_12'),
]

HMD_CHOICES = ('hmd_9', 'hmd_12', 'hmd_zero_9', 'hmd_zero_12')
METRIC_KEYS = ('full', 'upper', 'lower')


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Per-session mean MPJPE scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                   help=f'Dataset root (default: {DEFAULT_DATA_ROOT})')
    p.add_argument('--model', nargs=4, action='append', default=None,
                   metavar=('NAME', 'CONFIG', 'CKPT', 'HMD'),
                   help="Model spec: NAME CONFIG CKPT HMD  "
                        "(HMD is 'hmd_9' or 'hmd_12'). Repeatable. "
                        "If omitted, uses LHF + GBH defaults.")
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default=None,
                   help='CSV output path (default: auto-named in current dir)')
    p.add_argument('--output-json', default=None,
                   help='Optional JSON output path')
    p.add_argument('--limit', type=int, default=None,
                   help='Max frames per session (smoke test)')
    p.add_argument('--no-progress', action='store_true',
                   help='Suppress inline progress output')
    return p.parse_args()


def parse_model_specs(raw):
    specs = raw if raw else [list(m) for m in DEFAULT_MODELS]
    out = []
    seen = set()
    for name, cfg, ckpt, hmd in specs:
        if hmd not in HMD_CHOICES:
            raise SystemExit(
                f"Model {name!r}: HMD must be one of {HMD_CHOICES}, got {hmd!r}")
        if name in seen:
            raise SystemExit(f"Duplicate model name: {name!r}")
        seen.add(name)
        out.append(dict(name=name, config=cfg, ckpt=ckpt, hmd=hmd))
    return out


# ── Grouping / scanning ──────────────────────────────────────────────────

def discover_frames(data_root):
    """Load frames from data_root, handling both layouts:

      flat   : <root>/<session>/synced_data.csv
      nested : <root>/<subject>/<session>/synced_data.csv     (e.g. kinect_v5_split/Val)

    In nested mode, session ids are prefixed with the subject dir
    (e.g. ``022336/Dancing1_20260223_092417``) to stay unique across subjects.
    """
    if not os.path.isdir(data_root):
        print(f'ERROR: data-root not a directory: {data_root}', file=sys.stderr)
        return []

    children = [c for c in sorted(os.listdir(data_root))
                if os.path.isdir(os.path.join(data_root, c))]

    flat_hits = sum(
        1 for c in children
        if os.path.isfile(os.path.join(data_root, c, 'synced_data.csv')))
    if flat_hits > 0:
        print(f'  layout: flat ({flat_hits} sessions)')
        return load_all_frames(data_root)

    all_frames = []
    nested_subjects = 0
    for subj in children:
        subj_dir = os.path.join(data_root, subj)
        sess_hits = sum(
            1 for s in os.listdir(subj_dir)
            if os.path.isdir(os.path.join(subj_dir, s))
            and os.path.isfile(os.path.join(subj_dir, s, 'synced_data.csv')))
        if sess_hits == 0:
            continue
        nested_subjects += 1
        subj_frames = load_all_frames(subj_dir)
        for fr in subj_frames:
            fr['session'] = f"{subj}/{fr['session']}"
        all_frames.extend(subj_frames)

    if nested_subjects == 0:
        print('  no session dirs (synced_data.csv) found under data-root.',
              file=sys.stderr)
    else:
        print(f'  layout: nested ({nested_subjects} subjects)')
    return all_frames


def _action_from_session(sess):
    """Extract action name from a possibly-prefixed session id."""
    basename = sess.rsplit('/', 1)[-1]
    # Session naming is typically <Action>_<YYYYMMDD>_<HHMMSS>
    trimmed = '_'.join(basename.split('_')[:-2])
    return trimmed or basename


def group_by_session(frames):
    """Preserve load order while grouping by session."""
    order, by_sess = [], {}
    for fr in frames:
        s = fr['session']
        if s not in by_sess:
            order.append(s)
            by_sess[s] = []
        by_sess[s].append(fr)
    return order, by_sess


def scan_sessions(models, sessions, session_frames, pipeline, device,
                  limit=None, show_progress=True):
    """Runs inference for every (frame, model) pair and accumulates metrics.

    Returns a list of rows (one per session) — each a dict with keys
    `session`, `action`, `n_frames`, and `<MODEL>_{full,upper,lower}`.
    A final weighted OVERALL row is appended.
    """
    total = sum(min(len(session_frames[s]), limit or 10 ** 9) for s in sessions)
    done = 0
    rows = []

    for sess in sessions:
        frames = session_frames[sess]
        if limit is not None:
            frames = frames[:limit]
        if not frames:
            continue

        accum = {m['name']: {k: 0.0 for k in METRIC_KEYS} for m in models}

        for fr in frames:
            gt = fr['p3d']
            for m in models:
                if m['hmd'].startswith('hmd_zero_'):
                    dim = int(m['hmd'].rsplit('_', 1)[-1])
                    hmd_vec = np.zeros(dim, dtype=np.float32)
                else:
                    hmd_vec = fr[m['hmd']]
                pred = run_inference(m['loaded'], pipeline, fr,
                                     hmd_vec, device)
                full, up, lo = compute_mpjpe(pred, gt)
                a = accum[m['name']]
                a['full'] += float(full)
                a['upper'] += float(up)
                a['lower'] += float(lo)
            done += 1
            if show_progress and (done % 20 == 0 or done == total):
                sys.stdout.write(
                    f"\r  progress: {done}/{total} frames "
                    f"({len(rows)}/{len(sessions)} sessions done)   ")
                sys.stdout.flush()

        n = len(frames)
        action = _action_from_session(sess)
        row = dict(session=sess, action=action, n_frames=n)
        for m in models:
            for key in METRIC_KEYS:
                row[f"{m['name']}_{key}"] = accum[m['name']][key] / n
        rows.append(row)

    if show_progress:
        sys.stdout.write('\n')
        sys.stdout.flush()

    if rows:
        rows.append(_weighted_overall(rows, models))
    return rows


def _weighted_overall(rows, models):
    tot_n = sum(r['n_frames'] for r in rows)
    overall = dict(session='__overall__', action='OVERALL', n_frames=tot_n)
    for m in models:
        for key in METRIC_KEYS:
            col = f"{m['name']}_{key}"
            overall[col] = sum(r[col] * r['n_frames'] for r in rows) / tot_n
    return overall


def add_pairwise_deltas(rows, models):
    """Add A_minus_B_{metric} columns comparing each model against the first."""
    if len(models) < 2:
        return rows
    base = models[0]['name']
    for m in models[1:]:
        for key in METRIC_KEYS:
            col = f"{base}_minus_{m['name']}_{key}"
            for r in rows:
                r[col] = r[f"{base}_{key}"] - r[f"{m['name']}_{key}"]
    return rows


# ── Output ───────────────────────────────────────────────────────────────

def print_table(rows, models):
    header = ['Session', 'N']
    for m in models:
        header += [f"{m['name']}-F", f"{m['name']}-L"]
    base = models[0]['name'] if len(models) >= 2 else None
    if base:
        for m in models[1:]:
            header += [f"Δ({base}-{m['name']}).F", f"Δ({base}-{m['name']}).L"]

    def fmt_row(r):
        cells = [r['action'][:28], str(r['n_frames'])]
        for m in models:
            name = m['name']
            cells += [f"{r[name + '_full']:.1f}",
                      f"{r[name + '_lower']:.1f}"]
        if base:
            for m in models[1:]:
                name = m['name']
                dF = r[base + '_full'] - r[name + '_full']
                dL = r[base + '_lower'] - r[name + '_lower']
                cells += [f"{dF:+.1f}", f"{dL:+.1f}"]
        return cells

    body = [fmt_row(r) for r in rows]
    cols = list(zip(header, *body)) if body else [(h,) for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]

    def line(cells):
        return '  '.join(
            (str(c).ljust(w) if i == 0 else str(c).rjust(w))
            for i, (c, w) in enumerate(zip(cells, widths)))

    print()
    print(line(header))
    print('-' * (sum(widths) + 2 * (len(widths) - 1)))
    for b in body:
        print(line(b))
    print()


def csv_fieldnames(models):
    fields = ['session', 'action', 'n_frames']
    for m in models:
        for key in METRIC_KEYS:
            fields.append(f"{m['name']}_{key}")
    if len(models) >= 2:
        base = models[0]['name']
        for m in models[1:]:
            for key in METRIC_KEYS:
                fields.append(f"{base}_minus_{m['name']}_{key}")
    return fields


def write_csv(path, rows, models):
    fields = csv_fieldnames(models)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, '')
                out[k] = round(v, 2) if isinstance(v, float) else v
            w.writerow(out)


def write_json(path, rows, models, data_root):
    payload = dict(
        generated_at=datetime.now().isoformat(timespec='seconds'),
        data_root=data_root,
        models=[{k: m[k] for k in ('name', 'config', 'ckpt', 'hmd')}
                for m in models],
        rows=[{k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in r.items()}
              for r in rows],
    )
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    models = parse_model_specs(args.model)

    print(f'Data root : {args.data_root}')
    print(f'Device    : {args.device}')
    print(f'Models    :')
    for m in models:
        print(f"  - {m['name']:<10} hmd={m['hmd']}")
        print(f"      config: {m['config']}")
        print(f"      ckpt  : {m['ckpt']}")
    print()

    print('Loading frames...')
    frames = discover_frames(args.data_root)
    if not frames:
        print('ERROR: no frames found. Check --data-root.', file=sys.stderr)
        return 1
    sessions, by_sess = group_by_session(frames)
    print(f'  {len(frames)} frames across {len(sessions)} sessions.\n')

    # Load models FIRST — load_model() runs init_default_scope('mmpose'), which
    # is what registers the custom transforms that build_pipeline() relies on.
    print('Loading models...')
    for m in models:
        if not os.path.isfile(m['config']):
            print(f"  WARNING: config not found: {m['config']}", file=sys.stderr)
        if not os.path.isfile(m['ckpt']):
            print(f"  WARNING: checkpoint not found: {m['ckpt']}", file=sys.stderr)
        print(f"  {m['name']} ...")
        m['loaded'], _ = load_model(m['config'], m['ckpt'], args.device)
    print()

    pipeline = build_pipeline()

    print('Scanning sessions...')
    rows = scan_sessions(
        models, sessions, by_sess, pipeline, args.device,
        limit=args.limit, show_progress=not args.no_progress)
    if not rows:
        print('No sessions processed.', file=sys.stderr)
        return 1

    rows = add_pairwise_deltas(rows, models)
    print_table(rows, models)

    out_csv = args.output or f'session_mpjpe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
    write_csv(out_csv, rows, models)
    print(f'CSV  saved: {out_csv}')

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or '.',
                    exist_ok=True)
        write_json(args.output_json, rows, models, args.data_root)
        print(f'JSON saved: {args.output_json}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
