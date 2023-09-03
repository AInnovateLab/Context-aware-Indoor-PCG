import evaluate
import torch
from evaluate.info import EvaluationModuleInfo

import datasets


class ChamferDistance(evaluate.Metric):
    def _info(self) -> EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description="Accuracy with ignore label",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=None,
        )

    def _compute(self, predictions, references):
        predictions = torch.Tensor(predictions)
        references = torch.Tensor(references)

        # TO Do: batch
        batch = predictions.shape[0]
        predictions = predictions

        num_points_p, num_feat_p = predictions.shape
        num_points_r, num_feat_r = references.shape
        expanded_predictions = predictions.repeat(1, num_points_r, 1)
        expanded_references = torch.reshape(
            torch.unsqueeze(references, 2).repeat(1, 1, num_points_p, 1), (batch, -1, num_feat_r)
        )

        distance = (expanded_predictions - expanded_references).pow(2).sum(dim=1)
        distance = distance.reshape(batch, num_points_p, num_points_r)
        distance_1 = torch.min(distance, dim=1)

        distance = distance.transpose(1, 2)
        distance_2 = torch.min(distance, dim=1)
        distance = distance_1.values.sum(dim=1) + distance_2.values.sum(dim=1)
        return distance
