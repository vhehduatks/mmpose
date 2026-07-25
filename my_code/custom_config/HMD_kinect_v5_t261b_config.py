"""Task 26.1b — 26.1a + RESTORE the global K/V tokens arm A deleted (z, pose,
kin). Merged into arm A's cross-attn alongside the 2 controller tokens, bias
columns 0 for the globals. Judged vs 26.1a (61.57) — same frozen base, same
5-ep recipe, one variable: whether A's 3.5 mm shortfall vs pesens (58.08) is
the missing global context rather than the ego-cam frame. See ours_ec_modules.py.
"""

_base_ = ['./HMD_kinect_v5_t261a_config.py']

model = dict(head=dict(restore_globals=True))
work_dir = 'work_dirs/t261b'
