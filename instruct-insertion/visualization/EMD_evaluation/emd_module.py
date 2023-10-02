import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../.."))
from openpoints.cpp.emd.emd import earth_mover_distance


def emd_eval(prediction, references):
    """
    prediction: (N, 3)
    references: (B, N, 3)
    """
    d = 0
    emd = earth_mover_distance()
    for reference in references:
        d += emd.forward(
            xyz1=prediction[None, :], xyz2=reference[None, :], transpose=False
        )  # p1: B x N1 x 3, p2: B x N2 x 3
    return d / references.shape[0]


if __name__ == "__main__":
    import torch

    pred = torch.randn(1024, 3).to("cuda")
    refs = torch.randn(10, 1024, 3).to("cuda")
    print(emd_eval(pred, refs))
