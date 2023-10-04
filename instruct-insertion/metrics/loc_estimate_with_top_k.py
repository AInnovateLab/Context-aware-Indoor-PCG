import evaluate
import numpy as np
from evaluate.info import EvaluationModuleInfo

import datasets


class LocEstimateWithTopN(evaluate.Metric):
    def __init__(self, *args, topk=5, **kwargs):
        self.topk = topk
        super().__init__(*args, **kwargs)

    def _info(self) -> EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description="Top-k, Distance between predicted and reference points and the L1 difference of estimated radius.",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Array2D((self.topk, 4), "float32"),
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

        dist = np.linalg.norm(
            predictions[:, :, :3] - references[:, None, :3], ord=2, axis=2
        )  # (n_samples, # of choices)
        dist_min = dist.min(axis=1)  # (n_samples,)
        radius_diff = np.abs(
            predictions[:, :, 3] - references[:, None, 3]
        )  # (n_samples, # of choices)
        radius_diff_min = radius_diff.min(axis=1)  # (n_samples,)
        return {
            "dist": float(dist_min.mean()),
            "radius_diff": float(radius_diff_min.mean()),
        }
