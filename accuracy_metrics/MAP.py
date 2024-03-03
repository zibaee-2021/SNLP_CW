"""
Compute Mean Average Precision.

For Datasets.py task a,
evaluate the quality of the ranked list of documents returned by a system.
It takes into account both the order of the documents and the precision at each cut-off point in the list.


The example snippet below gives MAP score for given set of queries, true labels, and predicted scores.
(This should be adjusted based on the data to match the expected format.)
"""

import numpy as np


def average_precision(y_true, y_score, k=None):
    """
    Computes the average precision at k.

    Parameters:
    - y_true: np.array, true binary labels in range {0, 1} or {-1, 1}.
    - y_score: np.array, predicted scores.
    - k: int, the maximum number of predicted elements.

    Returns:
    - average_precision: float, the average precision at k.
    """

    # If k is None or greater than the number of elements, use all elements
    if k is None or k > len(y_true):
        k = len(y_true)

    # Sort scores and corresponding truth values
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, sorted_indices[:k])

    # Calculate precision at each rank
    precisions = [np.mean(y_true_sorted[:i + 1]) for i in range(k) if y_true_sorted[i]]

    # If no relevant documents, return 0
    if not precisions:
        return 0

    # Calculate average precision
    average_precision = np.mean(precisions)
    return average_precision


def mean_average_precision(y_true, y_scores, k=None):
    """
    Computes the mean average precision at k.

    Parameters:
    - y_true: List of arrays, true binary labels for each query in range {0, 1} or {-1, 1}.
    - y_scores: List of arrays, predicted scores for each query.
    - k: int, the maximum number of predicted elements for each query.

    Returns:
    - mean_average_precision: float, the mean average precision at k.
    """

    # Calculate average precision for each query and take the mean
    return np.mean([average_precision(yt, ys, k) for yt, ys in zip(y_true, y_scores)])


# Example usage:
y_true = [
    np.array([1, 0, 0, 1, 1]), # true labels for query 1
    np.array([0, 1, 0, 0, 1]), # true labels for query 2
    # Add more queries as needed
]

y_scores = [
    np.array([0.9, 0.1, 0.2, 0.8, 0.75]), # predicted scores for query 1
    np.array([0.2, 0.8, 0.1, 0.4, 0.85]), # predicted scores for query 2
    # Add more queries as needed
]

k = 5 # you can adjust k or leave it as None to consider all predictions
map_score = mean_average_precision(y_true, y_scores, k)
print(f"MAP: {map_score}")
