"""Task 42 / Arm A (IEEE VR 2027) — FROZEN hmd12 injection into external backbones.

mmpose-side plumbing for the single adapter defined in the egoallo repo
(`ours/frame_adapt/inject_adapter.py`, imported via PYTHONPATH). Nothing here is
tuned per host; the recipe is in `egoallo/ours/BASELINE_SELECTION.md` §3.

Registered here (custom_imports precedent: ours_a_modules.py):

  OursSplitHMD12 (transform) — after GenerateTarget (+ EnhanceHMDInfo on xR):
      results['hmd12']    = the 12-dim [LHF9, GBH3] vector  -> packed to gt_instance_labels
      results['hmd_info'] = zeros(1, hmd_dim)               -> the host head keeps its
                                                               image-only (zeroed) HMD input
    so the signal reaches the network ONLY through the adapter.

  OursInjectBaselinel1        — xR-EgoPose (Tome) head + adapter on feats[-1]
  OursInjectCascadedEnhanced  — our single-stage image-only head + adapter (control)

Both wrappers add `adapter(hmd12)` as a spatial-constant bias to the last backbone
feature map and then call the unchanged base head. Zero-init => identical to the
baseline at step 0.
"""

import os
import sys

import numpy as np
import torch

from mmpose.registry import MODELS, TRANSFORMS
from mmpose.models.heads.heatmap_heads.custom_egopose_baselinel1_head import (
    CustomxRegoposeBaselinel1)
from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced)

_EGOALLO_FA = os.environ.get(
    "EGOALLO_FRAME_ADAPT",
    "/home/hyeonghwan/github/egoallo/ours/frame_adapt")
if _EGOALLO_FA not in sys.path:
    sys.path.insert(0, _EGOALLO_FA)
from inject_adapter import HMD12_DIM, InjectAdapter  # noqa: E402


@TRANSFORMS.register_module()
class OursSplitHMD12:
    """Move the 12-dim hmd_info to `hmd12` (packed) and zero `hmd_info` (hmd_dim)."""

    def __init__(self, hmd_dim: int = 9):
        self.hmd_dim = hmd_dim

    def __call__(self, results: dict) -> dict:
        h = np.asarray(results["hmd_info"], dtype=np.float32).reshape(1, -1)
        assert h.shape[1] == HMD12_DIM, f"expected 12-dim hmd_info, got {h.shape}"
        results["hmd12"] = h
        results["hmd_info"] = np.zeros((1, self.hmd_dim), dtype=np.float32)
        table = dict(results.get("label_mapping_table", {}))
        table["hmd12"] = "hmd12"
        results["label_mapping_table"] = table
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(hmd_dim={self.hmd_dim})"


class _InjectMixin:
    """Adds the frozen adapter; `inject_stats` in {quest, xr} selects the Train stats."""

    def __init__(self, *args, inject_stats: str = "quest", **kwargs):
        super().__init__(*args, **kwargs)
        c = self.in_channels
        c = c[-1] if isinstance(c, (list, tuple)) else c
        self.inject = InjectAdapter(int(c), stats=inject_stats)

    def _inject(self, feats, batch_data_samples):
        hmd12 = torch.stack([
            torch.as_tensor(d.gt_instance_labels.hmd12) for d in batch_data_samples
        ]).reshape(len(batch_data_samples), HMD12_DIM)
        feats = list(feats) if isinstance(feats, (list, tuple)) else [feats]
        f = feats[-1]
        feats[-1] = self.inject.inject(f, hmd12.to(f.device))
        return tuple(feats)

    def predict(self, feats, batch_data_samples, test_cfg={}):
        return super().predict(self._inject(feats, batch_data_samples),
                               batch_data_samples, test_cfg)

    def loss(self, feats, batch_data_samples, train_cfg={}):
        return super().loss(self._inject(feats, batch_data_samples),
                            batch_data_samples, train_cfg)


@MODELS.register_module()
class OursInjectBaselinel1(_InjectMixin, CustomxRegoposeBaselinel1):
    pass


@MODELS.register_module()
class OursInjectCascadedEnhanced(_InjectMixin,
                                 CustomEgoposeCascadedRefinementHead_enhanced):
    pass
