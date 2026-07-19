"""Task 15.2 — feed the STF1-fused coarse to the frozen stage 2 (new module
file; no tracked mmpose files are edited — registered via custom_imports).

The fused coarse comes from the frame-env STF1 (cache-playback over the
coarse_ours cache) as pred_ec16 trees; this dataset attaches it per sample
(fused_coarse (1,16,3) ego-cam + fused_mask), and the head substitutes it
for the pose_decoder's coarse at the stage-2 input — weights untouched, so
the pass isolates "does a better coarse propagate through the frozen
refinement". Warm-up frames (no STF window) keep the original coarse.

Codebase rules honored: .reshape() only, new files only.
"""

import os
import re
from pathlib import Path

import numpy as np
import torch

from mmengine.structures import InstanceData
from mmpose.codecs.custom_mo2cap2_msra_heatmap import Custom_mo2cap2_MSRAHeatmap
from mmpose.datasets.datasets.body3d.custom_kinect_egopose_dataset import (
    KinectEgoposeDataset)
from mmpose.models.heads.heatmap_heads.custom_egopose_cascaded_refinement_head_enhanced import (  # noqa: E501
    CustomEgoposeCascadedRefinementHead_enhanced, preprocess_hmd_data_batch)
from mmpose.registry import DATASETS, KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy


@DATASETS.register_module()
class KinectEgoposeFusedDataset(KinectEgoposeDataset):
    """Paper loader + per-sample fused-coarse lookup from a pred_ec16 tree."""

    def __init__(self, *, fused_root: str, **kwargs):
        self.fused_root = Path(fused_root)
        super().__init__(**kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        pat = re.compile(r"frame_(\d+)\.jpg$")
        per_session = {}
        zero = np.zeros((1, 16, 3), dtype=np.float32)
        n_have = 0
        for d in data_list:
            img = d["img_path"]
            fid = int(pat.search(img).group(1))
            sess_dir = os.path.dirname(os.path.dirname(os.path.dirname(img)))
            participant = os.path.basename(os.path.dirname(sess_dir))
            session = os.path.basename(sess_dir)
            key = (participant, session)
            if key not in per_session:
                f = self.fused_root / participant / session / "pred_ec16.npz"
                if f.is_file():
                    z = np.load(f)
                    per_session[key] = dict(
                        zip(z["frame_ids"].tolist(),
                            z["joints_ec"].astype(np.float32)))
                else:
                    per_session[key] = None
            entry = per_session[key]
            kp = entry.get(fid) if entry else None
            if kp is None:
                d["fused_coarse"] = zero
                d["fused_mask"] = np.zeros(1, dtype=np.float32)
            else:
                d["fused_coarse"] = kp[None]
                d["fused_mask"] = np.ones(1, dtype=np.float32)
                n_have += 1
        print(f"[KinectEgoposeFusedDataset] {len(data_list)} samples, "
              f"{n_have} with fused coarse ({self.fused_root})")
        return data_list


@KEYPOINT_CODECS.register_module()
class OursFusedCodec(Custom_mo2cap2_MSRAHeatmap):
    label_mapping_table = dict(
        Custom_mo2cap2_MSRAHeatmap.label_mapping_table,
        fused_coarse="fused_coarse",
        fused_mask="fused_mask",
    )


@MODELS.register_module()
class OursFusedCoarseHead(CustomEgoposeCascadedRefinementHead_enhanced):
    """Parent decode with the coarse swapped for the fused coarse (masked)."""

    def decode(self, batch_outputs, batch_data_samples, backbone_feat=None):
        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args,)
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f"The decoder has not been set in {self.__class__.__name__}.")

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode)
            if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                batch_scores, batch_visibility = batch_scores
            else:
                batch_visibility = [None] * len(batch_keypoints)
        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints, batch_scores, batch_visibility = [], [], []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)
                batch_keypoints.append(keypoints)

        labels = [d.gt_instance_labels for d in batch_data_samples]
        HMD_info = torch.cat([l.hmd_info for l in labels])
        device = batch_outputs.device
        fused = torch.cat(
            [l.fused_coarse for l in labels]).to(device).float()
        fmask = torch.cat(
            [l.fused_mask for l in labels]).to(device).float()

        z = self.encoder(batch_outputs.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)
        batch_3d_keypoints = self.pose_decoder(z_plus_hmd)

        if self.use_auxiliary_decoders:
            generated_heatmaps = self.heatmap_decoder(z_plus_hmd)
            hmd_recons = preprocess_hmd_data_batch(batch_3d_keypoints)
        else:
            generated_heatmaps = None
            hmd_recons = None

        coarse_pose = batch_3d_keypoints.reshape(-1, 16, 3)
        # Task 15.2 substitution: STF1-fused coarse where available.
        coarse_pose = torch.where(
            fmask.reshape(-1, 1, 1) > 0, fused, coarse_pose)

        if self.use_refinement and backbone_feat is not None:
            output_3d = self.refine(
                coarse_pose, batch_outputs, backbone_feat, z,
                hmd_info=HMD_info)
        else:
            output_3d = coarse_pose

        preds = []
        for i, (keypoints, kp3d, scores, visibility) in enumerate(zip(
                batch_keypoints, output_3d, batch_scores, batch_visibility)):
            kp3d = kp3d.unsqueeze(dim=0)
            pred_kwargs = dict(keypoints=keypoints, keypoint_scores=scores,
                               keypoint_3d=kp3d)
            if generated_heatmaps is not None:
                pred_kwargs["generated_heatmap"] = \
                    generated_heatmaps[i].unsqueeze(dim=0)
            if hmd_recons is not None:
                pred_kwargs["hmd_recon"] = hmd_recons[i].unsqueeze(dim=0)
            pred = InstanceData(**pred_kwargs)
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds, output_3d

    def loss(self, feats, batch_data_samples, train_cfg={}):
        """Task 15.3c — parent loss with the SUBSTITUTED coarse feeding
        stage 2 (stage-1 losses stay on the original coarse; run with
        FreezeStage1Hook so only stage 2 trains)."""
        from mmpose.evaluation.functional import pose_pck_accuracy

        backbone_feat = feats[-1]
        pred_fields = self.forward(feats)

        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        labels = [d.gt_instance_labels for d in batch_data_samples]
        keypoint_weights = torch.cat([l.keypoint_weights for l in labels])
        gt_keypoint_3d = torch.cat([l.keypoint3d for l in labels])
        HMD_info = torch.cat([l.hmd_info for l in labels])
        device = pred_fields.device
        fused = torch.cat([l.fused_coarse for l in labels]).to(device).float()
        fmask = torch.cat([l.fused_mask for l in labels]).to(device).float()

        z = self.encoder(pred_fields.to(torch.float32))
        hmd_info_ = self.hmd_linear(HMD_info.to(torch.float32))
        z_plus_hmd = self._fuse(z, hmd_info_)
        coarse_orig = self.pose_decoder(z_plus_hmd).reshape(-1, 16, 3)

        loss_pose_l2norm = self.loss_pose_l2norm_module(
            coarse_orig, gt_keypoint_3d)
        loss_cosine = self.loss_cosine_similarity_module(
            coarse_orig, gt_keypoint_3d)
        loss_limb = self.loss_limb_length_module(coarse_orig, gt_keypoint_3d)
        loss_kpt = self.loss_module(pred_fields, gt_heatmaps, keypoint_weights)

        losses = dict(
            loss_pose_l2norm=torch.mean(loss_pose_l2norm),
            loss_cosine_similarity=torch.mean(loss_cosine),
            loss_limb_length=torch.mean(loss_limb),
            loss_kpt=loss_kpt)

        if self.use_auxiliary_decoders:
            recon_heatmap = self.heatmap_decoder(z_plus_hmd)
            hmd_recon = preprocess_hmd_data_batch(coarse_orig)
            loss_heatmap_recon = self.loss_heatmap_recon_module(
                recon_heatmap, gt_heatmaps, keypoint_weights)
            loss_hmd = self.loss_hmd_module(
                hmd_recon.to(torch.double), HMD_info[:, :9].to(torch.double))
            losses.update(loss_heatmap_recon=loss_heatmap_recon,
                          loss_hmd=loss_hmd)

        coarse_pose = torch.where(
            fmask.reshape(-1, 1, 1) > 0, fused, coarse_orig.detach())
        refined_pose = self.refine(
            coarse_pose, pred_fields, backbone_feat, z, hmd_info=HMD_info)
        loss_refined = self.loss_pose_l2norm_refined_module(
            refined_pose, gt_keypoint_3d)
        loss_bone = self.loss_bone_length_module(refined_pose, gt_keypoint_3d)
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
