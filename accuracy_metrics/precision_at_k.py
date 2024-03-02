"""
Compute Precision at k.

For Datasets.py-QA task a,
measure the precision of the first k results returned by the system.

"""

import numpy as np


def precision_at_k(y_true, y_score, k):
    """
    Computes the precision at k.

    Parameters:
    - y_true: np.array, true binary labels in range {0, 1} or {-1, 1}.
    - y_score: np.array, predicted scores.
    - k: int, the number of top-scored items to consider for precision calculation.

    Returns:
    - precision: float, the precision at k.
    """
    # Sort scores and corresponding truth values
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, sorted_indices[:k])

    # Calculate precision
    precision = np.mean(y_true_sorted)
    return precision
