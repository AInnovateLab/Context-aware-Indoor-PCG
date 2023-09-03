from typing import List, Optional, Tuple

import evaluate
import torch
from evaluate.info import EvaluationModuleInfo
from more_itertools import batched

from datasets import Features, Sequence, Value


class ChamferDistance(evaluate.Metric):
    def _info(self) -> EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description="Chamfer distance between pairwise point clouds, averaged over batch. "
            "Make sure the batch size are the same.",
            citation="",
            inputs_description="",
            features=Features(
                {
                    "predictions": Sequence(Sequence(Value("float"))),
                    "references": Sequence(Sequence(Value("float"))),
                }
            ),
            reference_urls=None,
        )

    def _compute(self, predictions, references, batch_size=128):
        """
        Args:
            predictions: 3d array-like predicted point clouds of shape (n_samples, P, C).
                The first 3 channels are x, y, z coordinates.
            references: 3d array-like reference point clouds of shape (n_samples, P, C)
                The first 3 channels are x, y, z coordinates.
            batch_size: batch size for computing chamfer distance
        """
        assert len(predictions) == len(references)
        distances: List[torch.Tensor] = list()
        feat_diffs: List[torch.Tensor] = list()
        for b_preds, b_refs in zip(
            batched(predictions, batch_size), batched(references, batch_size)
        ):
            b_preds = torch.tensor(b_preds)
            b_refs = torch.tensor(b_refs)
            dist, feat_diff = self.chamfer_distance(b_preds, b_refs)
            distances.append(dist)
            feat_diffs.append(feat_diff)

        distances = torch.concat(distances, dim=0)
        feat_diffs = torch.concat(feat_diffs, dim=0)
        return {
            "distance": float(distances.mean().item()),
            "feat_diff": float(feat_diffs.mean().item()),
        }

    @staticmethod
    def chamfer_distance(
        preds: torch.Tensor, refs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            preds: (B, P1, C)
            refs: (B, P2, C)
        """
        print(preds.shape)
        assert preds.shape[0] == refs.shape[0]
        assert preds.shape[2] == refs.shape[2]
        assert preds.shape[2] >= 3
        assert refs.shape[2] >= 3
        B, P1, C = preds.shape
        _, P2, C = refs.shape

        # Compute coord diff from each point in preds to each point in refs
        preds_coord = preds[:, :, :3]  # (B, P1, 3)
        refs_coord = refs[:, :, :3]  # (B, P2, 3)
        coord_diff = preds_coord[:, :, None, :].repeat(1, 1, P2, 1) - refs_coord[
            :, None, :, :
        ].repeat(
            1, P1, 1, 1
        )  # (B, P1, P2, 3)

        distance = torch.sum(coord_diff**2, dim=-1)  # (B, P1, P2)
        distance_preds, dist_idx_preds = distance.min(dim=2)  # (B, P1)
        distance_refs, dist_idx_refs = distance.min(dim=1)  # (B, P2)
        ret_dist = distance_preds.sum(dim=-1) + distance_refs.sum(dim=-1)  # (B,)

        # Compute feat diff
        if C == 3:
            ret_feat_diff = torch.zeros_like(ret_dist)
        else:
            preds_feat = preds[:, :, 3:]  # (B, P1, C-3)
            refs_feat = refs[:, :, 3:]  # (B, P2, C-3)
            # select
            dist_idx_preds = dist_idx_preds[:, :, None].repeat(1, 1, C - 3)
            dist_idx_refs = dist_idx_refs[:, :, None].repeat(1, 1, C - 3)
            min_dist_preds_feat = torch.gather(preds_feat, 1, dist_idx_preds)  # (B, P1, C-3)
            min_dist_refs_feat = torch.gather(refs_feat, 1, dist_idx_refs)  # (B, P2, C-3)
            # compute diff
            feat_diff_preds = preds_feat - min_dist_preds_feat  # (B, P1, C-3)
            feat_diff_refs = refs_feat - min_dist_refs_feat  # (B, P2, C-3)
            # l2
            feat_diff_preds = torch.sum(feat_diff_preds**2, dim=-1)  # (B, P1)
            feat_diff_refs = torch.sum(feat_diff_refs**2, dim=-1)  # (B, P2)

            ret_feat_diff = feat_diff_preds.sum(dim=-1) + feat_diff_refs.sum(dim=-1)  # (B,)

        return ret_dist, ret_feat_diff
