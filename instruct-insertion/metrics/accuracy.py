from typing import Optional

import evaluate
import numpy as np
from evaluate.info import EvaluationModuleInfo
from sklearn.metrics import accuracy_score

import datasets


class AccuracyWithIgnoredLabel(evaluate.Metric):
    def __init__(self, *args, ignore_label: Optional[int] = None, **kwargs):
        self.ignore_label = ignore_label
        super().__init__(*args, **kwargs)

    def _info(self) -> EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description="Accuracy with ignored label",
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

    def _compute(
        self, predictions, references, normalize=True, sample_weight=None, ignore_label=None
    ):
        """
        Args:
            predictions: 1d array-like predicted labels of shape (n_samples,)
            references: 1d array-like reference labels of shape (n_samples,)
            normalize(bool, optional): If False, return the number of correctly classified samples.
                Otherwise, return the fraction of correctly classified samples.
            sample_weight(optional): array-like of shape (n_samples,), Sample weights.
            ignore_label(int, optional): label to ignore in references.
        """
        predictions = np.array(predictions)
        references = np.array(references)
        assert predictions.shape == references.shape
        assert predictions.ndim == 1
        ignore_label = ignore_label or self.ignore_label
        if ignore_label is not None:
            # Remove ignore label
            ignore_index = np.where(references == ignore_label)
            predictions = np.delete(predictions, ignore_index)
            references = np.delete(references, ignore_index)
            if sample_weight is not None:
                sample_weight = np.delete(sample_weight, ignore_index)
        return {
            "accuracy": float(
                accuracy_score(
                    references, predictions, normalize=normalize, sample_weight=sample_weight
                )
            )
        }
