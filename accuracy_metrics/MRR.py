"""
Compute Mean Reciprocal Rank.

For BioASQ.py-QA task b, factoid questions,
evaluate the ranking of the correct answers provided by the system.

Mean Reciprocal Rank (MRR) is a statistical measure used to evaluate the performance of a system that returns
a ranked list of responses to queries.
It's particularly useful in scenarios like information retrieval and question answering systems
where you are interested in the rank of the first correct answer.

In the example snippet below, `y_true` represents the true labels for each query (1 for the correct answer,
0 for others), and `y_scores` represents the predicted scores for each query.
The function calculates the reciprocal rank for each query and then computes the mean of these values to get the MRR.

Remember that MRR considers only the rank of the first correct answer, making it particularly suited for
situations where the most important thing is to return a correct answer as high in the list as possible.
"""
import numpy as np


def mean_reciprocal_rank(y_true, y_scores):
    """
    Computes the Mean Reciprocal Rank (MRR).

    Parameters:
    - y_true: List of arrays, true binary labels for each query in range {0, 1} or {-1, 1}.
    - y_scores: List of arrays, predicted scores for each query.

    Returns:
    - mrr: float, the mean reciprocal rank.
    """
    rr_list = []
    for true, scores in zip(y_true, y_scores):
        # Sort scores in descending order and get the indices
        sorted_indices = np.argsort(scores)[::-1]

        # Find index of first relevant item (first item with label '1')
        for rank, idx in enumerate(sorted_indices, 1):
            if true[idx] == 1:
                rr_list.append(1 / rank)
                break

    # Compute mean of reciprocal ranks
    mrr = np.mean(rr_list) if rr_list else 0
    return mrr

# Example usage:
y_true = [
    np.array([0, 0, 1]), # true labels for query 1 (correct answer at rank 3)
    np.array([1, 0, 0]), # true labels for query 2 (correct answer at rank 1)
    # Add more queries as needed
]

y_scores = [
    np.array([0.1, 0.2, 0.9]), # predicted scores for query 1
    np.array([0.9, 0.1, 0.2]), # predicted scores for query 2
    # Add more queries as needed
]

mrr = mean_reciprocal_rank(y_true, y_scores)
print(f"MRR: {mrr}")
