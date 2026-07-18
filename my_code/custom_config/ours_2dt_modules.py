"""Task 11 — Ours-2Dt: inertially-aligned temporal 2D (new module file; no
tracked mmpose files are edited — registered via custom_imports).

Stage-1 experiment: use HMD ego-motion to align PAST 2D evidence to the
current pixel grid and fuse it into heatmap estimation. For pure camera
rotation the past->current pixel correspondence is EXACT and learning-free
from the HMD relative rotation (infinite homography):

    H(t-1 -> t) = K · R_rel · K^-1,   R_rel = R(cam2world(t))^T · R(cam2world(t-1))

11.1 pilot: the previous frame's decoded 2D keypoints (frozen-baseline stage-1
outputs, kp2d caches) are warped by H and rendered as per-joint Gaussian prior
maps at heatmap resolution (47x47, sigma matching the GT encoding), then
concatenated as 16 extra input channels to the stage-1 head's final 1x1 conv.
The pretrained final_layer weight loads into the leading input channels and
the prior channels are zero-padded, so zero_prior=True reproduces the baseline
EXACTLY (GATE 11b). `warp_mode="identity"` is the MANDATORY control (same
prior channels, H=I): it separates "inertial alignment" from "any temporal
prior" (the Ours-G lesson).

Causal; warm-up = 1 frame (fid 0 -> prior_mask=0 -> zero prior, single-frame
semantics). Backbone frozen (FreezeBackboneHook); the whole head (stage 1 +
stage 2) fine-tunes from the headline checkpoint.

Codebase rules honored: .reshape() only, new files only.
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mmengine.hooks import Hook
from mmpose.codecs.custom_mo2cap2_msra_heatmap import Custom_mo2cap2_MSRAHeatmap
from mmpose.datasets.datasets.body3d.custom_kinect_egopose_dataset import (
    KinectEgoposeDataset)
from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced)
from mmpose.registry import DATASETS, HOOKS, KEYPOINT_CODECS, MODELS

from my_code.custom_config.ours_t_modules import K_CX, K_CY, K_FX, K_FY

# Original image size the calibrated K is expressed in; heatmap grid covers
# the full image (EgoImageResize full-image stretch + MSRA stride mapping:
# heatmap cell coordinate = u_px * HEATMAP_SIZE / IMG_W).
IMG_W, IMG_H = 1920.0, 1080.0
HEATMAP_SIZE = 47


def warp_kp2d_homography(kp2d, rot_prev, rot_cur):
    """Warp previous-frame pixels by the infinite homography K R_rel K^-1.

    Args:
        kp2d: (B,16,2) px @1920x1080 (previous frame)
        rot_prev, rot_cur: (B,3,3) cam2world rotations at t-1 and t

    Returns:
        warped (B,16,2) px, valid (B,16) bool (in front of camera after warp)
    """
    r_rel = rot_cur.transpose(-1, -2) @ rot_prev
    x = (kp2d[..., 0] - K_CX) / K_FX
    y = (kp2d[..., 1] - K_CY) / K_FY
    d = torch.stack([x, y, torch.ones_like(x)], dim=-1)          # (B,16,3)
    d2 = torch.einsum("bij,bkj->bki", r_rel, d)
    valid = d2[..., 2] > 1e-6
    z = d2[..., 2].clamp_min(1e-6)
    u = d2[..., 0] / z * K_FX + K_CX
    v = d2[..., 1] / z * K_FY + K_CY
    return torch.stack([u, v], dim=-1), valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@DATASETS.register_module()
class KinectEgopose2DtDataset(KinectEgoposeDataset):
    """Paper loader + previous-frame 2D (frozen-baseline kp2d cache) and the
    two cam2world rotations needed for the homography warp.

    Extra per-sample fields (numpy, packed by Ours2DtCodec's label mapping):
        prior_kp2d     (1,16,2) previous frame's cached stage-1 2D, px
        prior_rot_prev (1,3,3)  egocam_left cam2world rotation at t-1
        prior_rot_cur  (1,3,3)  egocam_left cam2world rotation at t
        prior_mask     (1,)     1.0 iff fid>0 and cache + poses available
    """

    def __init__(self, *, frame_export_root: str,
                 kp2d_cache_name: str = "kp2d_ours_pilot", **kwargs):
        self.frame_export_root = Path(frame_export_root)
        self.kp2d_cache_name = kp2d_cache_name
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        pat = re.compile(r"frame_(\d+)\.jpg$")
        per_session = {}
        n_masked = 0
        zero_kp = np.zeros((1, 16, 2), dtype=np.float32)
        eye3 = np.eye(3, dtype=np.float32)[None]

        for d in data_list:
            img = d["img_path"]
            m = pat.search(img)
            assert m, img
            fid = int(m.group(1))
            sess_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
            participant = os.path.basename(os.path.dirname(sess_dir))
            session = os.path.basename(sess_dir)
            key = (participant, session)
            if key not in per_session:
                act = self.frame_export_root / participant / "actions" / session
                kp_npz = act / "cache" / self.kp2d_cache_name / "joints_2D.npz"
                pose_l = act / "on_device_poses" / "egocam_left.npz"
                if kp_npz.is_file() and pose_l.is_file():
                    per_session[key] = (
                        np.load(kp_npz)["kp2d"].astype(np.float32),
                        np.load(pose_l)["rotations"].astype(np.float32),
                    )
                else:
                    per_session[key] = None

            entry = per_session[key]
            if entry is None or fid < 1 or fid >= entry[1].shape[0]:
                d["prior_kp2d"] = zero_kp
                d["prior_rot_prev"] = eye3
                d["prior_rot_cur"] = eye3
                d["prior_mask"] = np.zeros(1, dtype=np.float32)
                n_masked += 1
            else:
                kp2d, rot = entry
                d["prior_kp2d"] = kp2d[fid - 1][None]
                d["prior_rot_prev"] = rot[fid - 1][None]
                d["prior_rot_cur"] = rot[fid][None]
                d["prior_mask"] = np.ones(1, dtype=np.float32)
        print(f"[KinectEgopose2DtDataset] {len(data_list)} samples, "
              f"{n_masked} masked (warm-up / uncached), "
              f"cache={self.kp2d_cache_name}")
        return data_list


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

@KEYPOINT_CODECS.register_module()
class Ours2DtCodec(Custom_mo2cap2_MSRAHeatmap):
    label_mapping_table = dict(
        Custom_mo2cap2_MSRAHeatmap.label_mapping_table,
        prior_kp2d="prior_kp2d",
        prior_rot_prev="prior_rot_prev",
        prior_rot_cur="prior_rot_cur",
        prior_mask="prior_mask",
    )


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

@MODELS.register_module()
class OursPrior2DHead(CustomEgoposeCascadedRefinementHead_enhanced):
    """Cascaded head + warped-keypoint prior channels into the stage-1
    final 1x1 conv.

    The pretrained final_layer weight loads into the first 256 input
    channels; the 16 prior channels are zero-initialized, so at init (and
    whenever the prior is zeroed by mask or `zero_prior`) the head is
    EXACTLY the pretrained baseline.
    """

    def __init__(self, *args, zero_prior: bool = False,
                 warp_mode: str = "homography", prior_sigma: float = 3.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert warp_mode in ("homography", "identity"), warp_mode
        self.zero_prior = zero_prior
        # "identity": mandatory control — same prior channels, H=I (no
        # inertial alignment). Separates "inertial alignment" from "any
        # temporal prior".
        self.warp_mode = warp_mode
        self.prior_sigma = prior_sigma

        # rebuild the final 1x1 conv with room for the prior channels;
        # zero-init the new input channels so prior=0 is exactly baseline.
        base = self.final_layer
        assert isinstance(base, nn.Conv2d) and base.kernel_size == (1, 1)
        self._base_final_in = base.in_channels
        self.final_layer = nn.Conv2d(
            base.in_channels + self.out_channels, base.out_channels,
            kernel_size=1)
        with torch.no_grad():
            self.final_layer.weight[:, :base.in_channels] = base.weight
            self.final_layer.weight[:, base.in_channels:].zero_()
            self.final_layer.bias.copy_(base.bias)

        cells = torch.arange(HEATMAP_SIZE, dtype=torch.float32)
        gy, gx = torch.meshgrid(cells, cells, indexing="ij")
        self.register_buffer("_grid_x", gx, persistent=False)
        self.register_buffer("_grid_y", gy, persistent=False)

        self._register_load_state_dict_pre_hook(self._pad_final_layer)

    def _pad_final_layer(self, state_dict, prefix, *args):
        """Load a baseline checkpoint: pad final_layer.weight with zero
        input channels for the prior maps."""
        key = prefix + "final_layer.weight"
        w = state_dict.get(key)
        if w is not None and w.shape[1] == self._base_final_in:
            pad = w.new_zeros(w.shape[0], self.out_channels, 1, 1)
            state_dict[key] = torch.cat([w, pad], dim=1)

    def _compute_priors(self, batch_data_samples, device):
        """(B,16,47,47) Gaussian prior maps from the warped previous-frame
        2D keypoints (matches the GT MSRA heatmap coordinate convention)."""
        labels = [d.gt_instance_labels for d in batch_data_samples]
        kp2d = torch.cat([l.prior_kp2d for l in labels]).to(device).float()
        rp = torch.cat([l.prior_rot_prev for l in labels]).to(device).float()
        rc = torch.cat([l.prior_rot_cur for l in labels]).to(device).float()
        mask = torch.cat([l.prior_mask for l in labels]).to(device).float()
        with torch.no_grad():
            if self.warp_mode == "homography":
                kp, valid = warp_kp2d_homography(kp2d, rp, rc)
            else:
                kp, valid = kp2d, torch.ones_like(kp2d[..., 0], dtype=torch.bool)
            u = kp[..., 0] * (HEATMAP_SIZE / IMG_W)             # (B,16) cells
            v = kp[..., 1] * (HEATMAP_SIZE / IMG_H)
            dx = self._grid_x.reshape(1, 1, HEATMAP_SIZE, HEATMAP_SIZE) \
                - u[..., None, None]
            dy = self._grid_y.reshape(1, 1, HEATMAP_SIZE, HEATMAP_SIZE) \
                - v[..., None, None]
            pri = torch.exp(-(dx * dx + dy * dy)
                            / (2.0 * self.prior_sigma ** 2))
            pri = pri * valid[..., None, None].float()
            pri = pri * mask.reshape(-1, 1, 1, 1)
            if self.zero_prior:
                pri = pri * 0.0
        return pri

    def _forward_with_prior(self, feats, priors):
        x = self.deconv_layers(feats[-1])
        x = self.add_deconv_layers(x)
        x = self.conv_layers(x)
        return self.final_layer(torch.cat([x, priors], dim=1))

    # predict() and loss() are copied from the parent with the single change
    # that the stage-1 forward receives the prior channels (the parent
    # hard-codes self.forward(feats)). flip_test is unused in our configs.

    def predict(self, feats, batch_data_samples, test_cfg={}):
        assert not test_cfg.get("flip_test", False), \
            "flip_test unsupported with prior channels"
        priors = self._compute_priors(batch_data_samples, feats[-1].device)
        batch_heatmaps = self._forward_with_prior(feats, priors)
        preds, _ = self.decode(batch_heatmaps, batch_data_samples,
                               backbone_feat=feats[-1])
        if test_cfg.get("output_heatmaps", False):
            from mmengine.structures import PixelData
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields
        return preds

    def loss(self, feats, batch_data_samples, train_cfg={}):
        from mmpose.evaluation.functional import pose_pck_accuracy
        from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
            preprocess_hmd_data_batch)
        from mmpose.utils.tensor_utils import to_numpy

        backbone_feat = feats[-1]
        priors = self._compute_priors(batch_data_samples, backbone_feat.device)
        pred_fields = self._forward_with_prior(feats, priors)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        gt_keypoint_3d = torch.cat([
            d.gt_instance_labels.keypoint3d for d in batch_data_samples])
        HMD_info = torch.cat([
            d.gt_instance_labels.hmd_info for d in batch_data_samples])

        z = self.encoder(pred_fields.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)
        coarse_pose = self.pose_decoder(z_plus_hmd).reshape(-1, 16, 3)

        loss_pose_l2norm = self.loss_pose_l2norm_module(
            coarse_pose, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(
            coarse_pose, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(coarse_pose, gt_keypoint_3d)
        loss_kpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses = dict(
            loss_pose_l2norm=torch.mean(loss_pose_l2norm),
            loss_cosine_similarity=torch.mean(loss_cosine),
            loss_limb_length=torch.mean(loss_limb),
            loss_kpt=loss_kpt)

        if self.use_auxiliary_decoders:
            recon_heatmap = self.heatmap_decoder(z_plus_hmd)
            hmd_recon = preprocess_hmd_data_batch(coarse_pose)
            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmap, gt_heatmaps, keypoint_weights)
            loss_hmd = self.loss_hmd_module(
                hmd_recon.to(torch.double), HMD_info[:, :9].to(torch.double))
            losses.update(loss_heatmap_recon=loss_heatmap_recon,
                          loss_hmd=loss_hmd)

        if self.use_refinement:
            refined_pose = self.refine(coarse_pose, pred_fields, backbone_feat,
                                       z, hmd_info=HMD_info)
            loss_refined = self.loss_pose_l2norm_refined_module(
                refined_pose, gt_keypoint_3d)
            loss_bone = self.loss_bone_length_module(
                refined_pose, gt_keypoint_3d)
            loss_sym = self.loss_symmetry_module(refined_pose)
            losses.update(loss_pose_l2norm_refined=torch.mean(loss_refined),
                          loss_bone_length=torch.mean(loss_bone),
                          loss_symmetry=torch.mean(loss_sym))

        if train_cfg.get("compute_acc", True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)
            losses.update(acc_pose=torch.tensor(
                avg_acc, device=gt_heatmaps.device))

        self.hm_iteration += 1
        return losses


# ---------------------------------------------------------------------------
# Freeze hook (backbone only — the whole head retrains in Task 11)
# ---------------------------------------------------------------------------

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Freeze the backbone only: no grads AND no BatchNorm running-stat
    updates (re-applied every epoch since the loop calls model.train())."""

    def _backbone(self, runner):
        model = runner.model
        model = model.module if hasattr(model, "module") else model
        return model.backbone

    def before_train(self, runner):
        m = self._backbone(runner)
        m.requires_grad_(False)
        n = sum(p.numel() for p in m.parameters())
        runner.logger.info(f"[FreezeBackboneHook] froze {n / 1e6:.1f}M params")

    def before_train_epoch(self, runner):
        self._backbone(runner).eval()
