import evaluate
import numpy as np
from evaluate.info import EvaluationModuleInfo

import datasets


class LocEstimateWithTopN(evaluate.Metric):
    def _info(self) -> EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description="Distance between predicted and reference points and the L1 difference of estimated radius.",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float"), length=4),
                    "references": datasets.Sequence(datasets.Value("float"), length=4),
                }
            ),
            reference_urls=None,
        )

    def _compute(self, predictions, references):
        """
        Args:
            predictions: 2d array-like predicted labels of shape (n_samples, # of choices, 4)
            references: 2d array-like reference labels of shape (n_samples, 4)
        """
        predictions = np.array(predictions)
        references = np.array(references)
        assert predictions.shape[0] == references.shape[0]
        assert predictions.shape[2] == references.shape[1]
        assert predictions.ndim == 3
        for i in range(predictions.shape[1]):
            dist = np.linalg.norm(predictions[:, i, :3] - references[:, i, :3], ord=2, axis=1)
            dist = np.mean(dist)
            radius_diff = np.abs(predictions[:, 3] - references[:, 3])
            radius_diff = np.mean(radius_diff)
            if i == 0:
                dist_min = dist
                radius_diff_min = radius_diff
                idx_of_min = i
            if dist < dist_min:
                dist_min = dist
                radius_diff_min = radius_diff
                idx_of_min = i

        return {
            "dist": float(dist_min),
            "radius_diff": float(radius_diff_min),
            "idx_of_min": int(idx_of_min),
        }
