"""
Compute recall.

For BioASQ-QA task a,
measure the fraction of relevant documents that are successfully retrieved.

Recall (or Sensitivity) measures the fraction of all relevant instances that are retrieved.


In example snippet below, `y_true` represents the true labels (1 for relevant items, 0 for irrelevant),
`y_score` represents the predicted scores or probabilities (usually the output of your model),
and `k` is the number of top-scored items to consider in the calculation of Precision at k and Recall.

Remember to validate the inputs and adjust these functions as necessary to fit the specific requirements of your application.
"""
import numpy as np
import precision_at_k as pak


def recall(y_true, y_score, k=None):
    """
    Computes the recall.

    Parameters:
    - y_true: np.array, true binary labels in range {0, 1} or {-1, 1}.
    - y_score: np.array, predicted scores.
    - k: int, the number of top-scored items to consider for recall calculation.
      If k is None, consider all items.

    Returns:
    - recall: float, the recall.
    """
    if k is not None:
        # Sort scores and corresponding truth values
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = np.take(y_true, sorted_indices[:k])
    else:
        y_true_sorted = y_true

    # Calculate recall
    recall = np.sum(y_true_sorted) / np.sum(y_true)
    return recall

# Example data
y_true = np.array([1, 0, 0, 1, 1]) # true labels
y_score = np.array([0.9, 0.1, 0.2, 0.8, 0.75]) # predicted scores

k = 3 # top 3 items

# Calculate Precision at k
p_at_k = pak.precision_at_k(y_true, y_score, k)
print(f'Precision at {k}: {p_at_k}')

# Calculate Recall
rec = recall(y_true, y_score, k)
print(f'Recall: {rec}')

