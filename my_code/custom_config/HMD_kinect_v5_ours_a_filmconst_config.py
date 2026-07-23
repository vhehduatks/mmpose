"""Task 22a control — same FiLM MLPs fed a LEARNED CONSTANT instead of
gravity: modulation capacity without per-frame gravity information.
22a must beat THIS, not 57.70."""

_base_ = ['./HMD_kinect_v5_ours_a_film_config.py']

model = dict(head=dict(film_mode='const'))
