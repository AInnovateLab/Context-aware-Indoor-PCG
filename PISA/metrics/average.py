import evaluate
from evaluate.info import EvaluationModuleInfo

import datasets


class Average(evaluate.Measurement):
    def __init__(self, *args, data_type: str = "float32", **kwargs):
        self.data_type = data_type
        super().__init__(*args, **kwargs)

    def _info(self) -> EvaluationModuleInfo:
        return evaluate.MetricInfo(
            description="Compute average of a list of numbers.",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value(self.data_type),
                }
            ),
            reference_urls=None,
        )

    def _compute(self, predictions):
        """
        Args:
            predictions: 1d array-like predicted number of shape (n_samples,)
        """
        return {"average": float(sum(predictions) / len(predictions))}
