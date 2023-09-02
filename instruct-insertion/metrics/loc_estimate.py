import evaluate
import numpy as np
from evaluate.info import EvaluationModuleInfo

import datasets


class LocEstimate(evaluate.Metric):
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
            predictions: 1d array-like predicted labels of shape (n_samples, 4)
            references: 1d array-like reference labels of shape (n_samples, 4)
        """
        predictions = np.array(predictions)
        references = np.array(references)
        dist = np.linalg.norm(predictions[:, :3] - references[:, :3], ord=2, axis=1)
        dist = np.mean(dist)
        radius_diff = np.abs(predictions[:, 3] - references[:, 3])
        radius_diff = np.mean(radius_diff)
        assert predictions.shape == references.shape
        return {
            "dist": float(dist),
            "radius_diff": float(radius_diff),
        }
